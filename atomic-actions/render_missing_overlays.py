#!/home/michal/miniforge3/envs/physicalai-av/bin/python
"""Pre-renders the viridis (dark blue -> green -> yellow) trajectory overlay.jpg for every scene in
PhysicalAI-EgoMotion/dataset that doesn't already have one cached under
atomic-actions/output/<clip_id>/overlay.jpg.

Reuses render_overlay()/list_scenes() from generate_atomic_actions_physicalai.py
(this only imports that module and calls its rendering helper -- it never
calls main(), so no API token is required). Needed so generate_atomic_actions_qwen.py
(which runs in a separate conda env without cv2/physical_ai_av) can just read
pre-rendered overlays instead of rendering them itself.
"""
import json
import os

import generate_atomic_actions_physicalai as gap


def main():
    clip_ids = gap.list_scenes()
    rendered = 0
    skipped = 0
    for clip_id in clip_ids:
        out_dir = os.path.join(gap.OUTPUT_DIR, clip_id)
        overlay_path = os.path.join(out_dir, "overlay.jpg")
        if os.path.exists(overlay_path):
            skipped += 1
            continue
        with open(os.path.join(gap.DATASET_DIR, clip_id, "motion.json")) as f:
            motion = json.load(f)
        os.makedirs(out_dir, exist_ok=True)
        gap.render_overlay(clip_id, motion, overlay_path)
        rendered += 1
    print(f"Rendered {rendered} new overlay(s), {skipped} already cached. Total scenes: {len(clip_ids)}.")


if __name__ == "__main__":
    main()
