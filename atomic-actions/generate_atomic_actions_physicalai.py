#!/home/michal/miniforge3/envs/physicalai-av/bin/python
"""Ask a GPT vision model to compose a 3-step atomic-action sequence
(go straight / turn left / turn right / stop) that best approximates each
scene's actual 6.4s ego trajectory, for every scene in
PhysicalAI-EgoMotion/dataset, and log the results to W&B.

Each scene there (built by PhysicalAI-EgoMotion/tmp/build_dataset.py) has a
raw.jpg front-camera frame and a motion.json with the ego's instantaneous
velocity/acceleration at t_frame_actual_s. This script re-renders the
trajectory overlay itself (same viridis dark-blue -> green -> yellow
colormap as build_dataset.py's own overlay.jpg, but with bigger/translucent
dots and a fixed vmin/vmax=[0, 1] so color is comparable across scenes) and
caches the result under output/<clip_id>/overlay.jpg.
"""

import argparse
import copy
import glob
import json
import mimetypes
import os
import re
import time
from base64 import b64encode
from datetime import datetime, timezone

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import wandb
from physical_ai_av import calibration as pav_calibration
from scipy.spatial.transform import Rotation, RigidTransform

GPT_MODEL = "gpt-5.4"
GPT_MAX_TOKENS = 2048

PHYSICALAI_ROOT = "/home/michal/Documents/01-Projects/PhysicalAI-EgoMotion"
DATASET_DIR = os.path.join(PHYSICALAI_ROOT, "dataset")
DATA_DIR = os.path.join(PHYSICALAI_ROOT, "data")
EGO_OFF_DIR = os.path.join(DATA_DIR, "labels", "egomotion.offline", "extracted")
CAM_INTRINSICS_PATH = os.path.join(DATA_DIR, "calibration", "camera_intrinsics", "camera_intrinsics.chunk_0000.parquet")
SENSOR_EXTRINSICS_PATH = os.path.join(DATA_DIR, "calibration", "sensor_extrinsics", "sensor_extrinsics.chunk_0000.parquet")
FRONT_CAM = "camera_front_wide_120fov"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompt.txt")
WANDB_PROJECT = "physicalai-atomic-actions"

TRAJECTORY_CMAP = "viridis"

ACTION_VOCAB = {"go straight", "turn left", "turn right", "stop"}
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_intrinsics_df = pd.read_parquet(CAM_INTRINSICS_PATH)
_extrinsics_df = pd.read_parquet(SENSOR_EXTRINSICS_PATH)
_calib_cache: dict[tuple, tuple] = {}


def load_egomotion_offline(clip_id):
    return pd.read_parquet(os.path.join(EGO_OFF_DIR, f"{clip_id}.egomotion.offline.parquet"))


def get_camera_calibration(clip_id, camera_name=FRONT_CAM):
    key = (clip_id, camera_name)
    if key not in _calib_cache:
        cam_intrinsics = pav_calibration.CameraIntrinsics.from_intrinsics_df(_intrinsics_df.loc[clip_id])
        cam_extrinsics = pav_calibration.SensorExtrinsics.from_extrinsics_df(_extrinsics_df.loc[clip_id])
        _calib_cache[key] = (
            cam_intrinsics.camera_models[camera_name],
            cam_extrinsics.sensor_poses[camera_name],
        )
    return _calib_cache[key]


def project_path_to_camera(clip_id, frame_time_s, t_min, t_max, camera_name=FRONT_CAM):
    """Same projection as PhysicalAI-EgoMotion/tmp/build_dataset.py: ego path
    from t_min to t_max, viewed from the ego's own pose at frame_time_s."""
    ego_off = load_egomotion_offline(clip_id).sort_values("timestamp").reset_index(drop=True)
    times_s = ego_off["timestamp"].to_numpy() / 1e6
    cam_model, cam_pose_rig = get_camera_calibration(clip_id, camera_name)

    t_idx = int(np.argmin(np.abs(times_s - frame_time_s)))
    pose_world_rig = RigidTransform.from_components(
        rotation=Rotation.from_quat(ego_off.loc[t_idx, ["qx", "qy", "qz", "qw"]].to_numpy()),
        translation=ego_off.loc[t_idx, ["x", "y", "z"]].to_numpy(),
    )

    window_mask = (times_s >= t_min) & (times_s <= t_max)
    path_df = ego_off[window_mask]
    path_times = times_s[window_mask]

    world_points = path_df[["x", "y", "z"]].to_numpy()
    rig_points = pose_world_rig.inv().apply(world_points)
    camera_points = cam_pose_rig.inv().apply(rig_points)

    pixels = cam_model.ray2pixel(camera_points)
    in_front = camera_points[:, 2] > 0
    in_bounds = ~cam_model.is_out_of_bounds(pixels)
    mask = in_front & in_bounds

    t_norm = (path_times - t_min) / max(t_max - t_min, 1e-9)
    return pixels[mask], t_norm[mask]


def render_overlay(clip_id, motion, out_path):
    """Draws the viridis (dark blue -> green -> yellow) trajectory overlay on raw.jpg and saves it to
    out_path. Cached: skipped if out_path already exists."""
    if os.path.exists(out_path):
        return
    raw_path = os.path.join(DATASET_DIR, clip_id, "raw.jpg")
    frame = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    pixels, t_norm = project_path_to_camera(
        clip_id, motion["t_frame_actual_s"], motion["t_frame_actual_s"], motion["t_trajectory_end_s"]
    )

    h, w = frame.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame)
    if len(pixels):
        ax.scatter(pixels[:, 0], pixels[:, 1], c=t_norm, cmap=TRAJECTORY_CMAP, vmin=0, vmax=1, s=45, linewidths=0, alpha=0.6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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


def image_path_to_data_url(path):
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    encoded = b64encode(open(path, "rb").read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def redact_image_data_urls(messages):
    redacted = copy.deepcopy(messages)
    for message in redacted:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                mime_type = url.split(";")[0].removeprefix("data:") if url.startswith("data:") else "unknown"
                part["image_url"]["url"] = f"<base64 {mime_type} image, {len(url)} chars, see image_paths.overlay>"
    return redacted


def openai_chat_completion(messages, api_token, base_url, model, seed=None, max_retries=3):
    request_body = {"model": model, "messages": messages, "temperature": 0.2, "max_completion_tokens": GPT_MAX_TOKENS}
    if seed is not None:
        request_body["seed"] = seed

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=120,
            )
            if not response.ok:
                raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text[:500]}")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"].strip()
            usage = payload.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            fingerprint = payload.get("system_fingerprint")
            return content, token_usage, fingerprint
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"OpenAI API call failed after {max_retries} attempts: {last_error}")


def parse_actions(text):
    """Parse {"reasoning": str, "actions": [3 strings from ACTION_VOCAB]}.
    Fail-loud on malformed replies, same convention as the sibling nuScenes
    atomic-proposition script -- a bad reply on one of ~100 scenes should be
    surfaced, not silently dropped."""
    stripped = CODE_FENCE_RE.sub("", text.strip())
    data = json.loads(stripped)

    if "reasoning" not in data or "actions" not in data:
        raise ValueError(f"Missing 'reasoning' or 'actions' key in model reply:\n{text}")

    reasoning = str(data["reasoning"]).strip()
    actions = data["actions"]
    if not isinstance(actions, list) or len(actions) != 3:
        raise ValueError(f"'actions' must be a list of exactly 3 items, got {actions!r} in model reply:\n{text}")

    normalized = [str(a).strip().lower() for a in actions]
    invalid = [a for a in normalized if a not in ACTION_VOCAB]
    if invalid:
        raise ValueError(f"Invalid action(s) {invalid} (must be one of {sorted(ACTION_VOCAB)}) in model reply:\n{text}")

    return reasoning, normalized


def build_output_json(clip_id, motion, reasoning, actions, timestamp, model_input, token_usage, seed, fingerprint, processing_time_s):
    return {
        "clip_id": clip_id,
        "reasoning": reasoning,
        "actions": actions,
        "motion": motion,
        "meta": {
            "gpt_model": GPT_MODEL,
            "timestamp": timestamp,
            "processing_time_s": processing_time_s,
            "model_input": model_input,
            "token_usage": token_usage,
            "seed": seed,
            "system_fingerprint": fingerprint,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed passed to the OpenAI-compatible API for best-effort deterministic sampling.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_token = os.environ.get("PROMPT_UPSAMPLER_API_TOKEN")
    base_url = os.environ.get("PROMPT_UPSAMPLER_ENDPOINT_URL")
    if not api_token or not base_url:
        raise RuntimeError(
            "PROMPT_UPSAMPLER_API_TOKEN and PROMPT_UPSAMPLER_ENDPOINT_URL must be set in the environment."
        )

    clip_ids = list_scenes(args.max_scenes)
    if not clip_ids:
        raise RuntimeError(f"No scenes found in {DATASET_DIR}")
    system_prompt = load_system_prompt(args.prompt_file)
    print(f"Found {len(clip_ids)} scene(s) in {DATASET_DIR}.")

    run = wandb.init(
        project=WANDB_PROJECT,
        tags=["physicalai-av"],
        config={
            "gpt_model": GPT_MODEL,
            "num_scenes": len(clip_ids),
            "max_scenes": args.max_scenes,
            "overwrite": args.overwrite,
            "prompt_file": args.prompt_file,
            "seed": args.seed,
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

    for clip_id in clip_ids:
        out_dir = os.path.join(OUTPUT_DIR, clip_id)
        out_path = os.path.join(out_dir, "actions.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"--- {clip_id} --- already labeled, skipping (use --overwrite to redo)")
            num_skipped += 1
            continue

        print(f"\n--- {clip_id} ---")
        with open(os.path.join(DATASET_DIR, clip_id, "motion.json")) as f:
            motion = json.load(f)

        os.makedirs(out_dir, exist_ok=True)
        overlay_path = os.path.join(out_dir, "overlay.jpg")
        render_overlay(clip_id, motion, overlay_path)

        image_data_url = image_path_to_data_url(overlay_path)
        motion_text = format_motion_text(motion)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": f"Ego motion at the start of the 6.4s window:\n{motion_text}"},
                ],
            },
        ]

        timestamp = datetime.now(timezone.utc).isoformat()
        t_start = time.time()
        reply, token_usage, fingerprint = openai_chat_completion(messages, api_token, base_url, GPT_MODEL, seed=args.seed)
        processing_time_s = round(time.time() - t_start, 3)
        reasoning, actions = parse_actions(reply)
        print(f"  actions: {actions}")
        print(f"  reasoning: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")
        print(f"  tokens: input={token_usage['input_tokens']} output={token_usage['output_tokens']}  time: {processing_time_s}s")
        total_input_tokens += token_usage["input_tokens"] or 0
        total_output_tokens += token_usage["output_tokens"] or 0
        total_processing_time_s += processing_time_s

        model_input = redact_image_data_urls(messages)
        output = build_output_json(
            clip_id, motion, reasoning, actions, timestamp, model_input, token_usage, args.seed, fingerprint, processing_time_s
        )
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        table.add_data(
            clip_id,
            motion.get("country"),
            wandb.Image(overlay_path),
            ", ".join(actions),
            reasoning,
            token_usage["input_tokens"],
            token_usage["output_tokens"],
            token_usage["total_tokens"],
            processing_time_s,
        )
        num_labeled += 1

    run.log({"actions": table})
    run.summary["total_input_tokens"] = total_input_tokens
    run.summary["total_output_tokens"] = total_output_tokens
    run.summary["total_tokens"] = total_input_tokens + total_output_tokens
    run.summary["num_labeled"] = num_labeled
    run.summary["num_skipped"] = num_skipped
    run.summary["total_processing_time_s"] = round(total_processing_time_s, 3)
    run.summary["avg_processing_time_s"] = round(total_processing_time_s / num_labeled, 3) if num_labeled else None
    run.finish()

    print(f"\nLabeled {num_labeled} scene(s), skipped {num_skipped} already-labeled scene(s). Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
