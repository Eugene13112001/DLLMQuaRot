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

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from dllmquant.algos.ia_aq import InteractionCollector  # noqa: E402
from dllmquant.calib.prompts import load_prompts  # noqa: E402
from dllmquant.config import DLLMQuantConfig, TMASConfig  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.modules import wrap_linears  # noqa: E402
from dllmquant.rotate import apply_quarot  # noqa: E402

OK, FAIL, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rotate", action="store_true",
                    help="also verify the QuaRot rotation plan and invariance")
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
    )
    failures = 0

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
    print("\n=== 4. IA-AQ attention probe ===")
    block = adapter.blocks[0]
    try:
        probe = adapter.make_probe(block)
        with probe:
            adapter.model(prompt.unsqueeze(0).to(next(adapter.model.parameters()).device))
        if probe.parts is None:
            print(FAIL, "probe never fired -- the hook is on the wrong module")
            failures += 1
        else:
            a = probe.parts.attn_probs
            v = probe.parts.value_states
            rowsum = a.sum(-1)
            print(OK, f"attn {tuple(a.shape)}, V {tuple(v.shape)}")
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

    # 5 -- linears can be wrapped and the block still runs -------------------
    print("\n=== 5. QuantLinear wrapping ===")
    try:
        replaced = wrap_linears(
            block, cfg.weight, cfg.activation, skip=cfg.skip, prefix="blocks.0"
        )
        print(OK, f"wrapped {len(replaced)} linears: {sorted(n.split('.')[-1] for n in replaced)}")
        if not replaced:
            print(FAIL, "nothing was wrapped -- skip_patterns are too broad")
            failures += 1
    except Exception:
        print(FAIL, "wrap_linears raised:")
        traceback.print_exc()
        failures += 1

    # 6 -- value projection is reachable for IA-AQ ---------------------------
    print("\n=== 6. value projection ===")
    leaves = {n.split(".")[-1] for n in replaced} if replaced else set()
    v_like = leaves & {"v_proj", "wv", "value", "att_proj", "qkv_proj", "Wqkv"}
    if v_like:
        print(OK, f"IA-AQ will attach to {sorted(v_like)}")
    else:
        print(FAIL, f"no value projection among {sorted(leaves)}; IA-AQ would be a no-op")
        failures += 1

    # 7 -- QuaRot rotation plan ----------------------------------------------
    if args.rotate:
        print("\n=== 7. QuaRot ===")
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
            print(OK, "rotation is invariant at every checked mask ratio")
        except Exception:
            print(FAIL, "rotation raised:")
            traceback.print_exc()
            failures += 1

    # 8 -- MoE expert coverage ----------------------------------------------
    if args.model_type == "llada2_moe":
        print("\n=== 8. expert coverage ===")
        try:
            adapter.coverage.attach(adapter.model)
            for _ in range(2):
                adapter.model(prompt.unsqueeze(0).to(
                    next(adapter.model.parameters()).device))
            adapter.coverage.detach()
            print(adapter.coverage.report())
        except Exception:
            print(WARN, "expert coverage probe failed:")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"{failures} check(s) FAILED -- fix the adapter before quantizing")
    else:
        print("all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
