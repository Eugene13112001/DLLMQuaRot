"""Run the ablation matrix and print one table.

Answers the question the individual scripts do not: how much quality did each
component actually buy, and where.  Every configuration is quantized from the
same FP16 weights, measured against the same held-out trajectory states, and
optionally scored on GSM8K.

    python scripts/compare.py --model GSAI-ML/LLaDA-1.5 --out out/compare.json

Fidelity is measured for every row (cheap, one forward per probe state);
GSM8K only with --gsm8k, because it costs a full generation per problem.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import List, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from dllmquant.calib.prompts import mix_prompts  # noqa: E402
from dllmquant.config import (  # noqa: E402
    CGQConfig,
    DLLMQuantConfig,
    IAAQConfig,
    QuantConfig,
    RotationConfig,
    TMASConfig,
)
from dllmquant.eval import ReferenceProbes, evaluate_gsm8k, measure_fidelity  # noqa: E402
from dllmquant.models import build_adapter  # noqa: E402
from dllmquant.modules import wrap_linears  # noqa: E402
from dllmquant.pipeline import DLLMQuantPipeline  # noqa: E402
from dllmquant.quantizers import quantize_weight_rtn  # noqa: E402
from dllmquant.report import round_floats, sibling_csv, write_csv  # noqa: E402


def recipes(args) -> list[dict]:
    """Nine rows: two anchors, the method, three ablations, two rotated,
    and the combination.

    The ablations are leave-one-out rather than add-one-in, deliberately.  The
    paper's claim is that the three components are complementary plug-ins, so
    what matters is each one's marginal contribution *in the presence of the
    others* -- which is what removing it measures.  Adding one to a bare
    baseline would answer a different question.
    """
    rows = [
        # --- anchors: the best and worst achievable at this bit width -------
        {"name": "FP16", "kind": "fp"},
        {"name": "RTN", "kind": "rtn"},
        # --- the paper's method ---------------------------------------------
        {"name": "DLLMQuant (full)", "kind": "dllm"},
        # --- leave-one-out: what each component is worth ---------------------
        {"name": "  -TMAS (uniform calib)", "kind": "dllm", "uniform": True},
        {"name": "  -IA-AQ", "kind": "dllm", "no_ia_aq": True},
        {"name": "  -CGQ weights (= GPTQ)", "kind": "dllm", "no_cgq_w": True},
        # --- rotation: the paper's third baseline, and the combination -------
        # QuaRot alone, twice. Neither row uses any DLLMQuant component; they
        # differ only in where GPTQ's calibration comes from, which is the one
        # thing about QuaRot that does not transfer to a DLLM unchanged.
        {"name": "QuaRot (text calib)", "kind": "dllm", "rotate": True,
         "no_ia_aq": True, "no_cgq_w": True, "calib": "text"},
        {"name": "QuaRot (trajectory calib)", "kind": "dllm", "rotate": True,
         "no_ia_aq": True, "no_cgq_w": True, "uniform": True},
        # Everything at once -- not "QuaRot", the combination.
        {"name": "DLLMQuant + QuaRot", "kind": "dllm", "rotate": True},
    ]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["name"].strip() in keep]
    return rows


def build_cfg(args, row: dict) -> DLLMQuantConfig:
    cgq = CGQConfig()
    if row.get("no_cgq_w"):
        cgq = CGQConfig(unmasked_weight=1.0, masked_base=1.0, beta=0.0)

    return DLLMQuantConfig(
        model_path=args.model,
        model_type=args.model_type,
        dtype=args.dtype,
        device=args.device,
        weight=QuantConfig(
            n_bits=args.w_bits, granularity="per_channel", mse_search=True
        ),
        activation=(
            QuantConfig(n_bits=args.a_bits, granularity="per_group",
                        group_size=args.a_group_size, dynamic=True)
            if args.a_group_size > 0
            else QuantConfig(n_bits=args.a_bits, granularity="per_token")
        ),
        tmas=TMASConfig(
            n_samples=args.nsamples,
            n_prompts=args.nprompts,
            uniform=bool(row.get("uniform")),
            mode=row.get("calib", "trajectory"),
        ),
        cgq=cgq,
        ia_aq=IAAQConfig(enabled=not row.get("no_ia_aq"), n_bits=args.a_bits),
        rotation=RotationConfig(enabled=bool(row.get("rotate"))),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-type", default="llada", choices=["llada", "llada2_moe"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--a-group-size", type=int, default=-1,
                    help="one activation scale per N channels instead of per "
                         "token; the second lever against channel outliers")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--nprompts", type=int, default=32)
    ap.add_argument("--n-probes", type=int, default=16,
                    help="held-out trajectory states for the fidelity metric")
    ap.add_argument("--gsm8k", action="store_true", help="also run GSM8K (slow)")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--only", default="", help="comma-separated row names")
    ap.add_argument("--out", default="", help="JSON path; a .csv is written beside it")
    ap.add_argument("--csv", default="", help="override the CSV path")
    args = ap.parse_args()

    rows = recipes(args)
    results = []
    reference = None

    for row in rows:
        print(f"\n{'=' * 70}\n{row['name']}\n{'=' * 70}")
        t0 = time.time()

        cfg = build_cfg(args, row)
        adapter = build_adapter(cfg)
        adapter.load()
        device = next(adapter.model.parameters()).device

        # Held-out probe states: the same ones for every row, and never drawn
        # from the calibration prompts.
        probe_prompts = adapter.encode_prompts(
            mix_prompts([("gsm8k", args.n_probes)], seed=12345),
            max_len=cfg.tmas.seq_len,
        )
        probe_cfg = TMASConfig(gen_length=32, block_length=16, steps=8)
        snaps = []
        for p in probe_prompts[: args.n_probes]:
            snaps.extend(adapter.trace_generation(p, probe_cfg))

        probes = ReferenceProbes(seed=0).capture(adapter.model, snaps, device)

        entry = {"name": row["name"], "kind": row["kind"]}

        if row["kind"] == "rtn":
            n = 0
            for bi, block in enumerate(adapter.blocks):
                for layer in wrap_linears(
                    block, cfg.weight, cfg.activation,
                    skip=cfg.skip, prefix=f"blocks.{bi}",
                ).values():
                    layer.set_weight(
                        quantize_weight_rtn(layer.weight.float(), cfg.weight)
                    )
                    n += 1
            entry["layers"] = n
        elif row["kind"] == "dllm":
            prompts = adapter.encode_prompts(
                mix_prompts([("gsm8k", cfg.tmas.n_prompts)], seed=0),
                max_len=cfg.tmas.seq_len,
            )
            report = DLLMQuantPipeline(cfg, adapter).run(prompts, verbose=False)
            entry["mean_proxy_loss"] = (
                sum(l.proxy_loss for l in report.layers) / max(len(report.layers), 1)
            )
            entry["layers"] = len(report.layers)

        fid = measure_fidelity(adapter.model, probes, device)
        entry["fidelity"] = {
            "top1": fid.overall["top1"],
            "kl": fid.overall["kl"],
            "conf_delta": fid.overall["conf_delta"],
            "per_bucket": fid.per_bucket,
        }
        print(fid.summary())

        if args.gsm8k:
            res = evaluate_gsm8k(adapter, n_samples=args.n_eval, verbose=False)
            entry["gsm8k"] = res.accuracy
            print(res.summary())

        entry["seconds"] = time.time() - t0
        results.append(entry)

        if row["kind"] == "fp":
            reference = entry

        del adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(results, reference, args)

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(results, indent=2), encoding="utf-8")
        csv_path = write_csv(
            args.csv or sibling_csv(p), flatten_results(results)
        )
        print(f"\nresults -> {p}\n         -> {csv_path}")
    return 0


def flatten_results(results: Sequence[dict]) -> List[dict]:
    """One row per recipe, with the per-bucket columns spread out flat."""
    buckets = _buckets(results)
    rows = []
    for e in results:
        f = e["fidelity"]
        row = {
            "recipe": e["name"].strip(),
            "top1": f["top1"],
            "kl": f["kl"],
            "conf_delta": f["conf_delta"],
            "mean_proxy_loss": e.get("mean_proxy_loss", ""),
            "layers": e.get("layers", ""),
            "gsm8k": e.get("gsm8k", ""),
            "seconds": e.get("seconds", ""),
        }
        for b in buckets:
            m = f["per_bucket"].get(b)
            row[f"top1_mask_{b}"] = m["top1"] if m else ""
            row[f"kl_mask_{b}"] = m["kl"] if m else ""
        rows.append(round_floats(row))
    return rows


def print_table(results, reference, args) -> None:
    print(f"\n\n{'=' * 96}")
    print(f"W{args.w_bits}A{args.a_bits} on {args.model}")
    print("=" * 96)

    head = f"{'recipe':26} {'top-1':>8} {'KL':>9} {'drift':>8} {'proxy':>9}"
    if args.gsm8k:
        head += f" {'GSM8K':>8} {'Δ':>7}"
    print(head)
    print("-" * 96)

    base_acc = reference.get("gsm8k") if reference else None
    for e in results:
        f = e["fidelity"]
        line = (
            f"{e['name']:26} {100 * f['top1']:7.2f}% {f['kl']:9.4f} "
            f"{f['conf_delta']:+8.4f} "
            f"{e.get('mean_proxy_loss', float('nan')):9.5f}"
        )
        if args.gsm8k and "gsm8k" in e:
            line += f" {100 * e['gsm8k']:7.2f}%"
            if base_acc is not None:
                line += f" {100 * (e['gsm8k'] - base_acc):+6.2f}"
        print(line)

    print("-" * 96)
    print("top-1  : agreement with FP16 argmax at masked positions (higher better)")
    print("KL     : KL(FP16 || quantized) at masked positions (lower better)")
    print("drift  : mean change in max probability (nearer zero better)")
    print("proxy  : mean CGQ layer loss; not comparable across recipes, only within")

    # Where the damage sits along the trajectory.
    print(f"\n{'recipe':26} " + " ".join(f"{b:>12}" for b in _buckets(results)))
    for e in results:
        cells = []
        for b in _buckets(results):
            m = e["fidelity"]["per_bucket"].get(b)
            cells.append(f"{100 * m['top1']:11.2f}%" if m else " " * 12)
        print(f"{e['name']:26} " + " ".join(cells))
    print("(top-1 agreement by mask ratio: leftmost = start of generation)")


def _buckets(results) -> list[str]:
    seen = set()
    for e in results:
        seen |= set(e["fidelity"]["per_bucket"])
    return sorted(seen, reverse=True)


if __name__ == "__main__":
    raise SystemExit(main())
