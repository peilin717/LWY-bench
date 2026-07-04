#!/usr/bin/env python3
"""Embed full misleading_context text into generated_v1 JSON items."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "generated_v1" / "cases"


def update_json(path: Path, context: str) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["misleading_context"] = context
    obj["misleading_context_chars"] = len(context)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    updated = 0
    lengths = []
    for case_dir in sorted(CASES.iterdir()):
        if not case_dir.is_dir():
            continue
        context_path = case_dir / "misleading_context.md"
        context = context_path.read_text(encoding="utf-8")
        lengths.append(len(context))
        update_json(case_dir / "single_turn" / "input.json", context)
        update_json(case_dir / "multi_turn" / "dialogue.json", context)
        updated += 2

    print(
        json.dumps(
            {
                "updated_json_files": updated,
                "base_cases": len(lengths),
                "context_chars_min": min(lengths),
                "context_chars_avg": round(sum(lengths) / len(lengths), 1),
                "context_chars_max": max(lengths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
