#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("/root/ICL/LWY-bench")
TARGETS = [
    ROOT / "results/raw_outputs/qwen2_5_1_5b_generated_v1/deepseek_judge_single/single_turn_judged.jsonl",
    ROOT / "results/raw_outputs/symbol_lora_qwen1_5b_generated_v1/deepseek_judge_single/single_turn_judged.jsonl",
]


for path in TARGETS:
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [r for r in rows if r.get("judge", {}).get("final_verdict") != "judge_error"]
    removed = len(rows) - len(keep)
    if removed:
        backup = path.with_suffix(path.suffix + ".before_retry_errors.bak")
        shutil.copy2(path, backup)
        with path.open("w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(path, "rows", len(rows), "removed_judge_errors", removed)
