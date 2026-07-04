#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/root/ICL/LWY-bench")
OUT = ROOT / "results/case_studies_single_turn.md"

MODELS = {
    "base": ROOT / "results/raw_outputs/qwen2_5_1_5b_generated_v1/deepseek_judge_single/single_turn_judged.jsonl",
    "symbol": ROOT / "results/raw_outputs/symbol_lora_qwen1_5b_generated_v1/deepseek_judge_single/single_turn_judged.jsonl",
    "metaicl": ROOT / "results/raw_outputs/metaicl_lora_qwen1_5b_generated_v1/deepseek_judge_single/single_turn_judged.jsonl",
}

SELECTED = [
    ("LWY-0001-PSEUDO-MATH-PROOF-GOLDBACH-IDENTITY", "symbol fails, metaicl robust"),
    ("LWY-0010-PSEUDO-MATH-PROOF-COLLATZ-ENERGY", "base and symbol fail, metaicl robust"),
    ("LWY-0038-PSEUDO-MATH-PROOF-RIEMANN-SYMMETRY", "metaicl boundary failure"),
]


def try_fix_once(s: str, enc: str) -> str | None:
    try:
        return s.encode(enc, errors="strict").decode("utf-8", errors="strict")
    except UnicodeError:
        return None


def score(s: str) -> tuple[int, int]:
    cjk = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    bad = sum(s.count(m) for m in ["å", "æ", "ç", "è", "é", "ä", "ã", "ï", "â", "Â", "Ã", "\ufffd"])
    return cjk, -bad


def fix_text(x):
    if isinstance(x, list):
        return [fix_text(v) for v in x]
    if isinstance(x, dict):
        return {k: fix_text(v) for k, v in x.items()}
    if not isinstance(x, str):
        return x
    candidates = [x]
    frontier = [x]
    for _ in range(3):
        new = []
        for s in frontier:
            for enc in ("latin1", "cp1252"):
                f = try_fix_once(s, enc)
                if f and f not in candidates:
                    candidates.append(f)
                    new.append(f)
        frontier = new
    return max(candidates, key=score)


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = fix_text(json.loads(line))
            out[r["case_id"]] = r
    return out


def compact_prompt(prompt: str) -> str:
    prompt = fix_text(prompt)
    title = re.search(r"^#\s*(.+)", prompt, flags=re.M)
    conclusion = re.search(r"结论[：:](.+?)(?:。|$)", prompt)
    formula = re.search(r"```text\n(.+?)\n```", prompt, flags=re.S)
    user_q = prompt.split("用户问题：")[-1].strip() if "用户问题：" in prompt else ""
    parts = []
    if title:
        parts.append(f"标题：{title.group(1).strip()}")
    if conclusion:
        parts.append(f"声称结论：{conclusion.group(1).strip()}。")
    if formula:
        parts.append("核心公式：" + " ".join(formula.group(1).split()))
    parts.append("误导结构：重新定义术语/引入未证明核心公式；用有限样本或小规模例子支持全称结论；把平均、下界或经验一致性误当成逐点证明。")
    if user_q:
        parts.append("用户请求：" + user_q[:260])
    return "\n".join(parts)


def clip(s: str, n: int = 700) -> str:
    s = fix_text(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s[:n] + ("..." if len(s) > n else "")


def flags(j: dict) -> str:
    ks = ["endorsement_error", "repair_without_rejection", "sycophantic_praise", "fabricated_support", "clear_rejection", "concrete_flaw_count"]
    return ", ".join(f"{k}={j.get(k)}" for k in ks)


data = {m: load(p) for m, p in MODELS.items()}

lines = ["# Single-turn Case Studies", ""]
for cid, label in SELECTED:
    base = data["base"][cid]
    lines += [f"## {cid}", "", f"Pattern: {label}", "", "### Prompt 摘要", "", compact_prompt(base["prompt_sent_to_model"]), ""]
    req = fix_text(base.get("required_flaws", []))
    lines += ["### Gold required flaws", ""]
    for x in req:
        lines.append(f"- {x}")
    lines.append("")
    for m in ["base", "symbol", "metaicl"]:
        r = data[m][cid]
        j = r["judge"]
        lines += [
            f"### {m}",
            "",
            f"- Verdict: `{j.get('final_verdict')}`, score={j.get('score')}",
            f"- Flags: {flags(j)}",
            f"- Judge reason: {fix_text(j.get('short_reason', ''))}",
            f"- Identified flaws: {'; '.join(fix_text(j.get('identified_flaws', [])))}",
            "",
            "Model response excerpt:",
            "",
            "```text",
            clip(r.get("prediction", ""), 900),
            "```",
            "",
        ]

OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
