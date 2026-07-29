"""Builds a small image+trajectory dataset from the local PhysicalAI-Autonomous-Vehicles slice.

For every clip that has egomotion, egomotion.offline, a video and camera
calibration, writes a folder named after the clip_id (the "scene") containing:
  - overlay.jpg   : front-camera frame at t=4.0s with the 6.4s trajectory
                    (t=4.0 -> t=10.4) projected on top
  - raw.jpg       : the same frame, with nothing overlaid
  - motion.json   : instantaneous 3D velocity/acceleration at t=4.0s
"""
import glob
import json
import os
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, RigidTransform

from physical_ai_av import calibration as pav_calibration

BASE = "/home/michal/Documents/01-Projects/PhysicalAI-EgoMotion"
DATA = os.path.join(BASE, "data")
EGO_DIR = os.path.join(DATA, "labels", "egomotion", "extracted")
EGO_OFF_DIR = os.path.join(DATA, "labels", "egomotion.offline", "extracted")
CAM_ZIP = os.path.join(DATA, "camera", "camera_front_wide_120fov", "camera_front_wide_120fov.chunk_0000.zip")
CAM_DIR = os.path.join(DATA, "camera", "camera_front_wide_120fov", "extracted")
CAM_INTRINSICS_PATH = os.path.join(DATA, "calibration", "camera_intrinsics", "camera_intrinsics.chunk_0000.parquet")
SENSOR_EXTRINSICS_PATH = os.path.join(DATA, "calibration", "sensor_extrinsics", "sensor_extrinsics.chunk_0000.parquet")
FRONT_CAM = "camera_front_wide_120fov"

OUT_DIR = os.path.join(BASE, "dataset")
N_CLIPS = 100
T_FRAME = 4.0
T_END = 10.4
SEQ_CMAP = "viridis"

os.makedirs(CAM_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

meta = pd.read_parquet(os.path.join(DATA, "metadata", "data_collection.parquet"))
_intrinsics_df = pd.read_parquet(CAM_INTRINSICS_PATH)
_extrinsics_df = pd.read_parquet(SENSOR_EXTRINSICS_PATH)
_calib_cache: dict[str, tuple] = {}

with zipfile.ZipFile(CAM_ZIP) as z:
    cam_ids = {n.split("/")[-1].split(".")[0] for n in z.namelist()}

ego_ids = {os.path.basename(f).split(".")[0]
           for f in glob.glob(os.path.join(EGO_DIR, "*.egomotion.parquet"))}
ego_off_ids = {os.path.basename(f).split(".")[0]
               for f in glob.glob(os.path.join(EGO_OFF_DIR, "*.egomotion.offline.parquet"))}
intrinsics_ids = set(_intrinsics_df.index.get_level_values("clip_id").unique())
extrinsics_ids = set(_extrinsics_df.index.get_level_values("clip_id").unique())

eligible = sorted(ego_ids & ego_off_ids & cam_ids & intrinsics_ids & extrinsics_ids)
print(f"{len(eligible)} clips eligible (have egomotion, egomotion.offline, video and calibration)")
clip_ids = eligible[:N_CLIPS]


def load_egomotion(clip_id):
    return pd.read_parquet(os.path.join(EGO_DIR, f"{clip_id}.egomotion.parquet"))


def load_egomotion_offline(clip_id):
    return pd.read_parquet(os.path.join(EGO_OFF_DIR, f"{clip_id}.egomotion.offline.parquet"))


def extract_clip_video(clip_id):
    out_path = os.path.join(CAM_DIR, f"{clip_id}.mp4")
    if os.path.exists(out_path):
        return out_path
    with zipfile.ZipFile(CAM_ZIP) as z:
        matches = [n for n in z.namelist() if clip_id in n]
        with z.open(matches[0]) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


def grab_frame_at_time(video_path, t):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = int(np.clip(round(t * fps), 0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), idx / fps


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


def project_path_to_camera(clip_id, frame_time_s, camera_name=FRONT_CAM, t_min=None, t_max=None):
    ego_off = load_egomotion_offline(clip_id).sort_values("timestamp").reset_index(drop=True)
    times_s = ego_off["timestamp"].to_numpy() / 1e6
    cam_model, cam_pose_rig = get_camera_calibration(clip_id, camera_name)

    t_idx = int(np.argmin(np.abs(times_s - frame_time_s)))
    pose_world_rig = RigidTransform.from_components(
        rotation=Rotation.from_quat(ego_off.loc[t_idx, ["qx", "qy", "qz", "qw"]].to_numpy()),
        translation=ego_off.loc[t_idx, ["x", "y", "z"]].to_numpy(),
    )

    window_mask = np.ones_like(times_s, dtype=bool)
    if t_min is not None:
        window_mask &= times_s >= t_min
    if t_max is not None:
        window_mask &= times_s <= t_max
    path_df = ego_off[window_mask]
    path_times = times_s[window_mask]

    world_points = path_df[["x", "y", "z"]].to_numpy()
    rig_points = pose_world_rig.inv().apply(world_points)
    camera_points = cam_pose_rig.inv().apply(rig_points)

    pixels = cam_model.ray2pixel(camera_points)
    in_front = camera_points[:, 2] > 0
    in_bounds = ~cam_model.is_out_of_bounds(pixels)
    mask = in_front & in_bounds

    color_min = t_min if t_min is not None else path_times.min()
    color_max = t_max if t_max is not None else path_times.max()
    t_norm = (path_times - color_min) / max(color_max - color_min, 1e-9)
    return pixels[mask], t_norm[mask]


def get_instant_velocity_acceleration(clip_id, t):
    df = load_egomotion(clip_id)
    times = df["timestamp"].to_numpy() / 1e6
    idx = int(np.argmin(np.abs(times - t)))
    row = df.iloc[idx]
    velocity = row[["vx", "vy", "vz"]].to_numpy(dtype=float)
    acceleration = row[["ax", "ay", "az"]].to_numpy(dtype=float)
    return velocity, acceleration


def save_raw(frame, out_path):
    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def save_overlay(frame, pixels, t_norm, out_path):
    h, w = frame.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame)
    if len(pixels):
        ax.scatter(pixels[:, 0], pixels[:, 1], c=t_norm, cmap=SEQ_CMAP, vmin=0, vmax=1, s=45, linewidths=0, alpha=0.6)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


written = 0
skipped = 0
for cid in clip_ids:
    try:
        clip_dir = os.path.join(OUT_DIR, cid)
        os.makedirs(clip_dir, exist_ok=True)

        video_path = extract_clip_video(cid)
        frame, actual_t = grab_frame_at_time(video_path, T_FRAME)
        pixels, t_norm = project_path_to_camera(cid, actual_t, FRONT_CAM, t_min=T_FRAME, t_max=T_END)
        velocity, acceleration = get_instant_velocity_acceleration(cid, actual_t)
        speed_kmh = float(np.linalg.norm(velocity) * 3.6)

        save_raw(frame, os.path.join(clip_dir, "raw.jpg"))
        save_overlay(frame, pixels, t_norm, os.path.join(clip_dir, "overlay.jpg"))

        country = meta.loc[cid, "country"] if cid in meta.index else None
        motion = {
            "clip_id": cid,
            "country": country,
            "t_frame_requested_s": T_FRAME,
            "t_frame_actual_s": actual_t,
            "t_trajectory_end_s": T_END,
            "velocity_mps": {"vx": float(velocity[0]), "vy": float(velocity[1]), "vz": float(velocity[2])},
            "speed_kmh": speed_kmh,
            "acceleration_mps2": {"ax": float(acceleration[0]), "ay": float(acceleration[1]), "az": float(acceleration[2])},
        }
        with open(os.path.join(clip_dir, "motion.json"), "w") as f:
            json.dump(motion, f, indent=2)

        written += 1
    except Exception as e:
        skipped += 1
        print(f"skipping {cid}: {e}")

print(f"done: {written} scenes written to {OUT_DIR}, {skipped} skipped")
