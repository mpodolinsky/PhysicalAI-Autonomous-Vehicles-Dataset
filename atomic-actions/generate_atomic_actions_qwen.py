#!/home/michal/miniforge3/envs/VLSM/bin/python
"""Local-inference counterpart to generate_atomic_actions_physicalai.py: asks
a Qwen3-VL model (instead of GPT) to compose a 3-step atomic-action sequence
(go straight / turn left / turn right / stop) that best approximates each
scene's actual 6.4s ego trajectory, and logs to the SAME W&B project
(physicalai-atomic-actions) so both models' runs live side by side.

Same prompt.txt, same scene set, same output JSON shape (clip_id/reasoning/
actions/motion/meta) and same W&B table columns as the GPT script -- only the
inference backend differs. Reads the viridis (dark blue -> green -> yellow) overlay.jpg already
cached under output/<clip_id>/overlay.jpg by generate_atomic_actions_physicalai.py
/ render_missing_overlays.py (this script's env has no cv2/physical_ai_av, so
it cannot render them itself -- run render_missing_overlays.py first if any
are missing).

No API cost for local inference, so unparsed replies are recorded (raw text
kept, actions/reasoning left null) and skipped rather than raising -- same
fail-soft convention as 06-Benchmark-VQA/benchmark_qwen.py, since a batch of
~100 local generations shouldn't die on one bad reply.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

import torch
import wandb
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 2048

ATOMIC_ACTIONS_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(os.path.dirname(ATOMIC_ACTIONS_DIR), "dataset")
OUTPUT_DIR = os.path.join(ATOMIC_ACTIONS_DIR, "output")
DEFAULT_PROMPT_FILE = os.path.join(ATOMIC_ACTIONS_DIR, "prompt.txt")
WANDB_PROJECT = "physicalai-atomic-actions"

ACTION_VOCAB = {"go straight", "turn left", "turn right", "stop"}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def list_scenes(max_scenes=None):
    clip_ids = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and os.path.exists(os.path.join(DATASET_DIR, d, "motion.json"))
    )
    if max_scenes:
        clip_ids = clip_ids[:max_scenes]
    return clip_ids


def load_system_prompt(prompt_file):
    with open(prompt_file) as f:
        return f.read()


def format_motion_text(motion):
    v = motion["velocity_mps"]
    a = motion["acceleration_mps2"]
    return (
        f"v=({v['vx']:+.1f}, {v['vy']:+.1f}, {v['vz']:+.1f}) m/s "
        f"(speed={motion['speed_kmh']:.0f} km/h)\n"
        f"a=({a['ax']:+.1f}, {a['ay']:+.1f}, {a['az']:+.1f}) m/s^2"
    )


def parse_actions(text):
    """Same parsing as generate_atomic_actions_physicalai.py's parse_actions,
    but returns (reasoning, actions, error) instead of raising -- see module
    docstring for why this script is fail-soft instead of fail-loud."""
    try:
        stripped = CODE_FENCE_RE.sub("", text.strip())
        data = json.loads(stripped)

        if "reasoning" not in data or "actions" not in data:
            raise ValueError("missing 'reasoning' or 'actions' key")

        reasoning = str(data["reasoning"]).strip()
        actions = data["actions"]
        if not isinstance(actions, list) or len(actions) != 3:
            raise ValueError(f"'actions' must be a list of exactly 3 items, got {actions!r}")

        normalized = [str(a).strip().lower() for a in actions]
        invalid = [a for a in normalized if a not in ACTION_VOCAB]
        if invalid:
            raise ValueError(f"invalid action(s) {invalid} (must be one of {sorted(ACTION_VOCAB)})")

        return reasoning, normalized, None
    except (json.JSONDecodeError, ValueError) as exc:
        return None, None, str(exc)


def build_output_json(clip_id, motion, reasoning, actions, error, model, timestamp, model_input, token_usage, processing_time_s):
    return {
        "clip_id": clip_id,
        "reasoning": reasoning,
        "actions": actions,
        "parse_error": error,
        "motion": motion,
        "meta": {
            "gpt_model": model,
            "timestamp": timestamp,
            "processing_time_s": processing_time_s,
            "model_input": model_input,
            "token_usage": token_usage,
            "seed": None,
            "system_fingerprint": None,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id. Default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-label scenes that already have an output JSON (default: skip them).",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Only process the first N scenes (default: all scenes in the dataset).",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Plain-text file containing the full system prompt. Default: {DEFAULT_PROMPT_FILE}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    clip_ids = list_scenes(args.max_scenes)
    if not clip_ids:
        raise RuntimeError(f"No scenes found in {DATASET_DIR}")
    system_prompt = load_system_prompt(args.prompt_file)
    print(f"Found {len(clip_ids)} scene(s) in {DATASET_DIR}.")

    model_slug = args.model.split("/")[-1]
    model_out_dir = os.path.join(OUTPUT_DIR, "by_model", model_slug)

    print(f"Loading {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
    ).to("cuda")

    run = wandb.init(
        project=WANDB_PROJECT,
        tags=["physicalai-av", "qwen"],
        config={
            "model": args.model,
            "num_scenes": len(clip_ids),
            "max_scenes": args.max_scenes,
            "overwrite": args.overwrite,
            "prompt_file": args.prompt_file,
            "source": "physicalai-egomotion",
        },
    )
    table = wandb.Table(columns=[
        "clip_id", "country", "image", "actions", "reasoning",
        "input_tokens", "output_tokens", "total_tokens", "processing_time_s",
    ])
    total_input_tokens = 0
    total_output_tokens = 0
    total_processing_time_s = 0.0
    num_labeled = 0
    num_skipped = 0
    num_unparsed = 0

    progress = tqdm(clip_ids, desc="Labeling", unit="scene")
    for clip_id in progress:
        out_dir = os.path.join(model_out_dir, clip_id)
        out_path = os.path.join(out_dir, "actions.json")
        if os.path.exists(out_path) and not args.overwrite:
            num_skipped += 1
            continue

        overlay_path = os.path.join(OUTPUT_DIR, clip_id, "overlay.jpg")
        if not os.path.exists(overlay_path):
            raise RuntimeError(
                f"No cached overlay for {clip_id} at {overlay_path} -- run render_missing_overlays.py "
                "(in the physicalai-av env) first."
            )
        with open(os.path.join(DATASET_DIR, clip_id, "motion.json")) as f:
            motion = json.load(f)

        motion_text = format_motion_text(motion)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": overlay_path},
                    {"type": "text", "text": f"{system_prompt}\n\nEgo motion at the start of the 6.4s window:\n{motion_text}"},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        t_start = time.time()
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        processing_time_s = round(time.time() - t_start, 3)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        token_usage = {
            "input_tokens": int(inputs.input_ids.shape[-1]),
            "output_tokens": int(generated_ids_trimmed[0].shape[-1]),
            "total_tokens": int(inputs.input_ids.shape[-1] + generated_ids_trimmed[0].shape[-1]),
        }
        reasoning, actions, error = parse_actions(raw_text)
        if error:
            num_unparsed += 1
            tqdm.write(f"{clip_id}: unparsed reply ({error}): {raw_text[:200]}")
            reasoning = raw_text
        else:
            tqdm.write(f"{clip_id}: actions={actions}")
        total_input_tokens += token_usage["input_tokens"]
        total_output_tokens += token_usage["output_tokens"]
        total_processing_time_s += processing_time_s

        model_input = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"<local file, see image_paths.overlay: {overlay_path}>"},
                    messages[0]["content"][1],
                ],
            },
        ]
        timestamp = datetime.now(timezone.utc).isoformat()
        output = build_output_json(
            clip_id, motion, reasoning, actions, error, args.model, timestamp, model_input, token_usage, processing_time_s
        )
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        table.add_data(
            clip_id,
            motion.get("country"),
            wandb.Image(overlay_path),
            ", ".join(actions) if actions else None,
            reasoning,
            token_usage["input_tokens"],
            token_usage["output_tokens"],
            token_usage["total_tokens"],
            processing_time_s,
        )
        num_labeled += 1
        progress.set_postfix(labeled=num_labeled, skipped=num_skipped, unparsed=num_unparsed)

    run.log({"actions": table})
    run.summary["total_input_tokens"] = total_input_tokens
    run.summary["total_output_tokens"] = total_output_tokens
    run.summary["total_tokens"] = total_input_tokens + total_output_tokens
    run.summary["num_labeled"] = num_labeled
    run.summary["num_skipped"] = num_skipped
    run.summary["num_unparsed"] = num_unparsed
    run.summary["total_processing_time_s"] = round(total_processing_time_s, 3)
    run.summary["avg_processing_time_s"] = round(total_processing_time_s / num_labeled, 3) if num_labeled else None
    run.finish()

    print(f"\nLabeled {num_labeled} scene(s) ({num_unparsed} unparsed), skipped {num_skipped} already-labeled. Output dir: {model_out_dir}")


if __name__ == "__main__":
    main()
