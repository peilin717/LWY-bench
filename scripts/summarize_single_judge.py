#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/ICL/LWY-bench")
MODELS = [
    ("qwen2.5-1.5b-base", ROOT / "results/raw_outputs/qwen2_5_1_5b_generated_v1/deepseek_judge_single/judge_summary.json"),
    ("symbol-lora", ROOT / "results/raw_outputs/symbol_lora_qwen1_5b_generated_v1/deepseek_judge_single/judge_summary.json"),
    ("metaicl-lora", ROOT / "results/raw_outputs/metaicl_lora_qwen1_5b_generated_v1/deepseek_judge_single/judge_summary.json"),
]
OUT = ROOT / "results/single_turn_judge_comparison.md"
OUT_JSON = ROOT / "results/single_turn_judge_comparison.json"


def pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100*x:.2f}%"


rows = []
for name, path in MODELS:
    s = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "model": name,
            "n": s["n"],
            "judge_errors": s["judge_errors"],
            "failure_rate": s["failure_rate"],
            "robust_rejection_rate": s["robust_rejection_rate"],
            "partial_safe_rate": s["partial_safe_rate"],
            "mean_score": s["mean_score"],
            "verdict_counts": s["verdict_counts"],
            "flag_counts": s["flag_counts"],
            "by_domain": s["by_domain"],
        }
    )

lines = [
    "# LWY-bench Single-turn DeepSeek Judge Comparison",
    "",
    "Judge model: `deepseek-v4-flash`.",
    "",
    "| Model | n | Judge errors | Failure rate | Robust rejection | Partial safe | Mean score |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['model']} | {r['n']} | {r['judge_errors']} | {pct(r['failure_rate'])} | "
        f"{pct(r['robust_rejection_rate'])} | {pct(r['partial_safe_rate'])} | {r['mean_score']:.4f} |"
    )

lines += ["", "## Failure Flag Counts", "", "| Model | endorsement | repair without rejection | sycophantic praise | fabricated support |", "|---|---:|---:|---:|---:|"]
for r in rows:
    f = r["flag_counts"]
    lines.append(
        f"| {r['model']} | {f.get('endorsement_error', 0)} | {f.get('repair_without_rejection', 0)} | "
        f"{f.get('sycophantic_praise', 0)} | {f.get('fabricated_support', 0)} |"
    )

lines += ["", "## Domain Failure Rates", ""]
domains = sorted(rows[0]["by_domain"])
lines.append("| Domain | " + " | ".join(r["model"] for r in rows) + " |")
lines.append("|---|" + "|".join("---:" for _ in rows) + "|")
for d in domains:
    lines.append("| " + d + " | " + " | ".join(pct(r["by_domain"][d]["failure_rate"]) for r in rows) + " |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUT)
print(OUT_JSON)
