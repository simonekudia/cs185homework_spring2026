from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEFAULT_JUDGE_MODEL, JudgeConfig, grade_policy_submission, load_jsonl, load_public_data


def main() -> None:
    jsonl_path = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.70

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set.")

    thresholds = json.loads((Path(__file__).resolve().parent / "thresholds.json").read_text(encoding="utf-8"))
    judge_cfg = JudgeConfig(
        api_key=api_key,
        judge_model=thresholds.get("judge_model", DEFAULT_JUDGE_MODEL),
        reasoning_effort=str(thresholds.get("reasoning_effort", "none")),
        max_workers=int(os.environ.get("LOCAL_AUTOGRADER_MAX_WORKERS", "1")),
    )

    public = load_public_data()
    metrics = grade_policy_submission(public["part2_prompts"], public["part2_base"], load_jsonl(jsonl_path), judge_cfg)
    rate = metrics["policy_win_rate_pair_agree_usable"]

    print(f"file     = {jsonl_path}")
    print(f"win_rate = {rate:.4f}")
    print(f"threshold= {threshold:.4f}")
    print(f"usable   = {metrics['count_pair_agree_usable_rows']}")
    print(f"status   = {'PASS' if rate >= threshold else 'FAIL'}")


if __name__ == "__main__":
    main()