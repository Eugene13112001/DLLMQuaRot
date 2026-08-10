"""Verify the adapter against a real checkpoint BEFORE launching a long job.

This code was written from the papers and model cards; it could not be executed
against LLaDA weights on the machine it was authored on.  Every assumption it
makes about the architecture is checked here, cheaply, so a mismatch costs a
minute instead of six hours.

    python scripts/selfcheck.py --model GSAI-ML/LLaDA-1.5 --model-type llada
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dllmquant.algos.ia_aq import InteractionCollector  # noqa: E402
from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.models.llada import VALUE_BEARING_NAMES  # noqa: E402
from dllmquant.modules import wrap_linears  # noqa: E402
from dllmquant.rotate import apply_quarot  # noqa: E402

OK, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"

# torch.OutOfMemoryError since 2.5; the cuda-namespaced one before that.
_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def _print_gpu_memory(label: str = "") -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    held = torch.cuda.memory_allocated()
    print(f"  {label}GPU: {free / 2**30:.2f} GB free of {total / 2**30:.1f} GB, "
          f"{held / 2**30:.1f} GB held by this process")


def representative_block(adapter):
    """The first block that has experts, else block 0.

    Blocks are not interchangeable in an MoE: LLaDA2.0 replaces the first
    ``first_k_dense_replace`` layers with dense MLPs, so block 0 carries no
    experts at all.  Checking it would exercise 3 linears and say nothing
    about the 768 that hold almost all of the model's weights.
    """
    import torch.nn as nn

    for i, block in enumerate(adapter.blocks):
        for name, module in block.named_modules():
            if name.split(".")[-1] == "experts" and isinstance(module, nn.ModuleList):
                return i, block
    return 0, adapter.blocks[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rotate", action="store_true",
                    help="also verify the QuaRot rotation plan and invariance")
    ap.add_argument("--allow-untested-transformers", action="store_true",
                    help="skip the transformers major-version guard")
    ap.add_argument("--device-map", default="",
                    help="only for models too big for one GPU (e.g. 'auto'); "
                         "LLaDA's remote code breaks accelerate's device-map "
                         "inference, so leave it empty unless you need it")
    args = ap.parse_args()

    cfg = DLLMQuantConfig(
        model_path=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map or None,
        allow_untested=args.allow_untested_transformers,
    )
    failures = 0
    skipped: list[str] = []

    import transformers
    print(f"torch {torch.__version__} · transformers {transformers.__version__} · "
          f"cuda {torch.cuda.is_available()} "
          f"({torch.cuda.device_count()} device(s))")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU 0: {torch.cuda.get_device_name(0)}, "
              f"{free / 2**30:.1f} GB free of {total / 2**30:.1f} GB")

    print(f"\n=== loading {args.model} ===")
    adapter = build_adapter(cfg)
    try:
        adapter.load()
        print(OK, adapter.describe())
        # The headroom that matters is what is left *after* the weights land,
        # and on a shared node it is not what was free at launch.
        _print_gpu_memory("after load: ")
    except Exception:
        print(FAIL, "load/validate raised:")
        traceback.print_exc()
        return 1

    # 1 -- mask token round-trips ------------------------------------------
    print("\n=== 1. mask token ===")
    try:
        tok = adapter.tokenizer.convert_ids_to_tokens(adapter.mask_id)
        print(OK, f"mask_id {adapter.mask_id} decodes to {tok!r}")
    except Exception as exc:
        print(WARN, f"could not decode mask_id {adapter.mask_id}: {exc}")

    # 2 -- the sampler actually denoises ------------------------------------
    print("\n=== 2. denoising loop ===")
    tmas = TMASConfig(gen_length=32, block_length=16, steps=8, n_samples=4, n_prompts=1)
    prompt = adapter.encode_prompts(load_prompts(1), max_len=64)[0]
    try:
        snaps = adapter.trace_generation(prompt, tmas)
        ratios = [s.mask_ratio for s in snaps]
        print(OK, f"{len(snaps)} snapshots, mask ratio {ratios[0]:.2f} -> {ratios[-1]:.2f}")
        if ratios[0] <= ratios[-1]:
            print(FAIL, "mask ratio did not decrease -- no tokens were committed")
            failures += 1
        conf = torch.cat([s.confidence[s.mask] for s in snaps if s.mask.any()])
        if conf.numel() and (conf.min() < 0 or conf.max() > 1.0001):
            print(FAIL, f"confidence outside [0,1]: {conf.min():.3f}..{conf.max():.3f}")
            failures += 1
        else:
            print(OK, f"masked-token confidence in [{conf.min():.3f}, {conf.max():.3f}]")
    except Exception:
        print(FAIL, "trace_generation raised:")
        traceback.print_exc()
        return 1

    # 3 -- generation is not gibberish --------------------------------------
    print("\n=== 3. generation sanity ===")
    try:
        out = adapter.generate(prompt, tmas)
        text = adapter.tokenizer.decode(out[0, prompt.shape[-1]:], skip_special_tokens=True)
        print(OK, f"completion: {text[:120]!r}")
        if adapter.mask_id in out[0, prompt.shape[-1]:].tolist():
            print(FAIL, "output still contains mask tokens")
            failures += 1
    except Exception:
        print(FAIL, "generate raised:")
        traceback.print_exc()
        failures += 1

    # 4 -- the attention probe fires ----------------------------------------
    block_idx, block = representative_block(adapter)
    print(f"\n=== 4. IA-AQ attention probe (block {block_idx}) ===")
    try:
        probe = adapter.make_probe(block)
        captured = None
        ids = prompt.unsqueeze(0).to(next(adapter.model.parameters()).device)
        with probe:
            adapter.model(ids, **adapter.forward_kwargs(ids))
            # parts are released on __exit__, so read them inside the scope.
            if probe.parts is not None:
                captured = (
                    probe.parts.attn_probs.detach().clone(),
                    tuple(probe.parts.value_states.shape),
                )
        if captured is None:
            print(FAIL, "probe never fired -- the hook is on the wrong module")
            failures += 1
        else:
            a, v_shape = captured
            rowsum = a.sum(-1)
            print(OK, f"attn {tuple(a.shape)}, V {v_shape}")
            if not torch.allclose(rowsum, torch.ones_like(rowsum), atol=1e-2):
                print(FAIL, f"attention rows do not sum to 1 (got {rowsum.mean():.3f})")
                failures += 1
            else:
                print(OK, "attention rows sum to 1")
            if not probe.rope_applied:
                print(WARN, "rotary embeddings were NOT applied; IA-AQ importance "
                            "ranking is approximate (see LLaDAAttentionProbe docstring)")
    except Exception:
        print(FAIL, "probe raised:")
        traceback.print_exc()
        failures += 1

    # NOTE ON ORDER: the QuaRot check must run BEFORE the wrapping check.
    # wrap_linears mutates block 0 in place, and rotation_plan scans the real
    # module tree -- checking rotation afterwards would inspect a model this
    # script itself had already modified.

    # 5 -- QuaRot rotation plan ----------------------------------------------
    if args.rotate:
        print("\n=== 5. QuaRot ===")
        try:
            plan = adapter.rotation_plan()
            print(OK, f"plan: {len(plan.embeddings)} embeddings, "
                      f"{len(plan.input_linears)} residual readers, "
                      f"{len(plan.output_linears)} residual writers, "
                      f"{len(plan.norm_groups)} norms, "
                      f"{len(plan.head_pairs)} head rotations")
            n_blocks = len(adapter.blocks)
            if len(plan.head_pairs) != n_blocks:
                print(WARN, f"{len(plan.head_pairs)} head rotations for "
                            f"{n_blocks} blocks -- some blocks were missed")

            cfg.rotation.enabled = True
            report = apply_quarot(adapter, cfg)
            print(report.summary())
            if report.decisive:
                print(OK, "rotation is invariant at every checked mask ratio")
            else:
                print(WARN, "invariance is not decisive in this precision -- "
                            "routing moved as well, and that alone explains a "
                            "change of this size. Verify once with "
                            "--dtype float32.")
                skipped.append("5 (invariance, undecidable at this precision)")
        except Exception:
            print(FAIL, "rotation raised:")
            traceback.print_exc()
            print(FAIL, "rotation mutates weights in place, so it stopped "
                        "part-way through and the model is now inconsistent. "
                        "Checks below run on that model -- read them as noise, "
                        "not as findings.")
            failures += 1

    # 6 -- linears can be wrapped and the block still runs -------------------
    print("\n=== 6. QuantLinear wrapping ===")
    replaced = None  # distinct from "wrapped nothing", which is a real failure
    try:
        replaced = wrap_linears(
            block, cfg.weight, cfg.activation, skip=cfg.skip,
            prefix=f"blocks.{block_idx}",
        )
        # Counted by leaf name: an MoE block has 773 of them and the flat list
        # is unreadable. The counts also carry information -- 257 of each
        # expert projection is 256 experts plus the shared one.
        kinds = Counter(n.split(".")[-1] for n in replaced)
        shape = ", ".join(
            f"{k} x{v}" if v > 1 else k for k, v in sorted(kinds.items())
        )
        print(OK, f"wrapped {len(replaced)} linears: {shape}")
        if not replaced:
            print(FAIL, "nothing was wrapped -- skip_patterns are too broad")
            failures += 1
    except _OOM:
        # Not an adapter problem, and calling it one sends you to read the
        # wrong code.  On a shared node the card can fill up between the load
        # and this line -- it did, mid-run, while checks 1-4 were passing.
        print(WARN, "out of memory while wrapping: the GPU filled up, the "
                    "adapter is not implicated. Free memory below, then rerun.")
        _print_gpu_memory()
        skipped.append("6 (QuantLinear wrapping)")
    except Exception:
        print(FAIL, "wrap_linears raised:")
        traceback.print_exc()
        failures += 1

    # 7 -- value projection is reachable for IA-AQ ---------------------------
    print("\n=== 7. value projection ===")
    if replaced is None:
        print(WARN, "skipped: nothing was wrapped to look at")
        skipped.append("7 (value projection)")
    else:
        leaves = {n.split(".")[-1] for n in replaced}
        v_like = leaves & set(VALUE_BEARING_NAMES)
        if v_like:
            print(OK, f"IA-AQ will attach to {sorted(v_like)}")
        else:
            print(FAIL,
                  f"no value projection among {sorted(leaves)}; "
                  "IA-AQ would be a no-op")
            failures += 1

    # 8 -- MoE expert coverage ----------------------------------------------
    if args.model_type == "llada2_moe":
        print("\n=== 8. expert coverage ===")
        try:
            adapter.coverage.attach(adapter.model)
            ids = prompt.unsqueeze(0).to(next(adapter.model.parameters()).device)
            for _ in range(2):
                adapter.model(ids, **adapter.forward_kwargs(ids))
            adapter.coverage.detach()
            # Threshold 1, not the default: two forwards of one short prompt
            # route a handful of tokens per expert, so every expert would be
            # "starved" and the warning would mean nothing. What this check
            # answers is whether routing is observable at all -- the budget
            # question needs a real calibration run.
            print(adapter.coverage.report(min_tokens=1))
            print("  (one prompt, and routing is deterministic, so most experts "
                  "seeing nothing here is expected -- what this check answers "
                  "is whether routing is observable at all. Note the spread "
                  "between median and max: usage is concentrated, so a "
                  "calibration budget sized off the mean will still starve "
                  "the median expert.)")
        except Exception:
            print(WARN, "expert coverage probe failed:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} check(s) FAILED -- fix the adapter before quantizing")
    elif skipped:
        # "all checks passed" over a run that skipped two of them is how an
        # unverified adapter gets launched into a six-hour job.
        print(f"no failures, but {len(skipped)} check(s) never ran: "
              f"{', '.join(skipped)}")
        print("the adapter is not cleared until they do")
    else:
        print("all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
