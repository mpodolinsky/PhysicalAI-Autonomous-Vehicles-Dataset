#!/home/michal/miniforge3/envs/physicalai-av/bin/python
"""Compares GPT's and Qwen's 3-step atomic-action sequences for each scene
(from generate_atomic_actions_physicalai.py / generate_atomic_actions_qwen.py)
and buckets each pair into one of three categories:

- "correct": all 3 steps match.
- "partial": the sequences agree on a leading, contiguous run of steps (from
  step 1) and disagree from some point on, with no further matches after
  that point -- e.g. steps 1-2 match, step 3 doesn't. Steps 1 and 2 matching
  is "fine" (a real partial match); steps 1 and 3 matching while step 2
  doesn't is NOT partial -- that's a discontinuity (the two sequences
  diverged and then coincidentally matched again later), and is bucketed as
  "incorrect" instead, same as a spurious match at step 2/3 with step 1
  already wrong.
- "incorrect": no matching leading run at all, or a discontinuous match as
  described above.

Scenes where either model's reply didn't parse (actions is null) are
reported separately and excluded from the three percentages.
"""

import argparse
import json
import os

ATOMIC_ACTIONS_DIR = os.path.dirname(__file__)
GPT_OUTPUT_DIR = os.path.join(ATOMIC_ACTIONS_DIR, "output")
QWEN_OUTPUT_DIR = os.path.join(ATOMIC_ACTIONS_DIR, "output", "by_model", "Qwen3-VL-8B-Instruct")
DEFAULT_REPORT_PATH = os.path.join(ATOMIC_ACTIONS_DIR, "output", "comparison_gpt_vs_qwen.json")


def classify(actions_a, actions_b):
    """Returns "correct", "partial", or "incorrect" for one pair of 3-step
    action sequences -- see module docstring for the exact rule."""
    matches = [a == b for a, b in zip(actions_a, actions_b)]

    prefix_len = 0
    while prefix_len < len(matches) and matches[prefix_len]:
        prefix_len += 1

    if prefix_len == len(matches):
        return "correct"

    trailing_match_after_break = any(matches[prefix_len + 1:])
    if trailing_match_after_break:
        return "incorrect"
    return "partial" if prefix_len >= 1 else "incorrect"


def load_actions(out_dir, clip_id):
    path = os.path.join(out_dir, clip_id, "actions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("actions")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-dir", default=GPT_OUTPUT_DIR, help=f"GPT output dir. Default: {GPT_OUTPUT_DIR}")
    parser.add_argument("--qwen-dir", default=QWEN_OUTPUT_DIR, help=f"Qwen output dir. Default: {QWEN_OUTPUT_DIR}")
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help=f"Where to write the JSON report. Default: {DEFAULT_REPORT_PATH}")
    return parser.parse_args()


def main():
    args = parse_args()

    gpt_clip_ids = {
        d for d in os.listdir(args.gpt_dir)
        if os.path.isdir(os.path.join(args.gpt_dir, d)) and os.path.exists(os.path.join(args.gpt_dir, d, "actions.json"))
    }
    qwen_clip_ids = {
        d for d in os.listdir(args.qwen_dir)
        if os.path.isdir(os.path.join(args.qwen_dir, d)) and os.path.exists(os.path.join(args.qwen_dir, d, "actions.json"))
    }
    common_clip_ids = sorted(gpt_clip_ids & qwen_clip_ids)
    if not common_clip_ids:
        raise RuntimeError(f"No clip_id has both a GPT and Qwen output ({args.gpt_dir} / {args.qwen_dir}).")

    per_clip = {}
    unparsed = []
    for clip_id in common_clip_ids:
        gpt_actions = load_actions(args.gpt_dir, clip_id)
        qwen_actions = load_actions(args.qwen_dir, clip_id)
        if gpt_actions is None or qwen_actions is None:
            unparsed.append(clip_id)
            continue
        category = classify(gpt_actions, qwen_actions)
        per_clip[clip_id] = {
            "gpt_actions": gpt_actions,
            "qwen_actions": qwen_actions,
            "category": category,
        }

    counts = {"correct": 0, "partial": 0, "incorrect": 0}
    for entry in per_clip.values():
        counts[entry["category"]] += 1
    num_compared = len(per_clip)

    percentages = {
        category: round(100 * count / num_compared, 1) if num_compared else None
        for category, count in counts.items()
    }

    print(f"Compared {num_compared} scene(s) with both GPT and Qwen parsed output "
          f"({len(unparsed)} skipped -- at least one side unparsed, out of {len(common_clip_ids)} total).")
    for category in ("correct", "partial", "incorrect"):
        print(f"  {category:<10} {counts[category]:>3} / {num_compared}  ({percentages[category]}%)")

    if unparsed:
        print(f"\nSkipped (unparsed on at least one side): {', '.join(unparsed)}")

    report = {
        "gpt_dir": args.gpt_dir,
        "qwen_dir": args.qwen_dir,
        "num_common_scenes": len(common_clip_ids),
        "num_compared": num_compared,
        "num_unparsed_skipped": len(unparsed),
        "counts": counts,
        "percentages": percentages,
        "unparsed_clip_ids": unparsed,
        "per_clip": per_clip,
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote full report to {args.report_path}")


if __name__ == "__main__":
    main()
