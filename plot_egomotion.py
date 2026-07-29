"""Plot ego motion trajectories from a small sample of the NVIDIA
PhysicalAI-Autonomous-Vehicles dataset (labels/egomotion, chunk 0000).

Each clip's egomotion parquet gives x, y, z position (meters) in a local
frame with origin = ego vehicle position at timestamp 0 (timestamp in
microseconds, can be negative/positive around that origin), plus velocity,
acceleration and heading quaternion at 100 Hz.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
EGO_DIR = os.path.join(BASE, "data", "labels", "egomotion", "extracted")
PLOTS_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Single accent hue for sequential (time) encoding: dark teal ramp.
SEQ_CMAP = "viridis"
LINE_COLOR = "#1b6f7a"  # single categorical hue used where identity doesn't matter
GRID_COLOR = "#d9d9d9"
BG = "#ffffff"

meta = pd.read_parquet(os.path.join(BASE, "data", "metadata", "data_collection.parquet"))

files = sorted(glob.glob(os.path.join(EGO_DIR, "*.egomotion.parquet")))
print(f"Found {len(files)} clips")

# ---------- Small multiples: 16 individual trajectories, colored by time ----------
N_SMALL = 16
rng = np.random.default_rng(7)
sample_files = rng.choice(files, size=min(N_SMALL, len(files)), replace=False)

fig, axes = plt.subplots(4, 4, figsize=(14, 14), facecolor=BG)
fig.suptitle(
    "Ego motion trajectories — 16 sample clips (PhysicalAI-Autonomous-Vehicles)",
    fontsize=14,
    color="#222222",
)

for ax, f in zip(axes.flat, sample_files):
    df = pd.read_parquet(f)
    clip_id = os.path.basename(f).split(".")[0]
    country = meta.loc[clip_id, "country"] if clip_id in meta.index else "?"

    t = df["timestamp"].to_numpy()
    t_norm = (t - t.min()) / max(t.max() - t.min(), 1)
    sc = ax.scatter(df["x"], df["y"], c=t_norm, cmap=SEQ_CMAP, s=4, linewidths=0)
    ax.plot(df["x"], df["y"], color="#00000022", linewidth=0.8, zorder=0)
    ax.scatter([0], [0], color="#d64550", s=25, zorder=5, marker="o")  # t=0 origin

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{country}", fontsize=9, color="#444444")
    ax.tick_params(labelsize=6, colors="#888888")
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)

fig.text(
    0.5, 0.005,
    "Color = time within the 20s clip (dark→light = early→late). Red dot = position at t=0.",
    ha="center", fontsize=9, color="#666666",
)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])
out1 = os.path.join(PLOTS_DIR, "small_multiples_trajectories.png")
plt.savefig(out1, dpi=150, facecolor=BG)
plt.close(fig)
print("Saved", out1)

# ---------- Overlay: many trajectories recentered/rotated to a common start heading ----------
N_OVERLAY = 100
overlay_files = rng.choice(files, size=min(N_OVERLAY, len(files)), replace=False)

fig, ax = plt.subplots(figsize=(9, 9), facecolor=BG)
for f in overlay_files:
    df = pd.read_parquet(f)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    # Rotate so the initial heading (first two points) points "up" (+y),
    # to make forward/turning patterns comparable across clips.
    if len(x) > 1:
        dx0, dy0 = x[1] - x[0], y[1] - y[0]
        heading = np.arctan2(dy0, dx0)
        rot = np.pi / 2 - heading
        c, s = np.cos(rot), np.sin(rot)
        xr = c * x - s * y
        yr = s * x + c * y
    else:
        xr, yr = x, y
    ax.plot(xr, yr, color=LINE_COLOR, linewidth=0.7, alpha=0.35)

ax.scatter([0], [0], color="#d64550", s=30, zorder=5)
ax.set_aspect("equal", adjustable="datalim")
ax.set_title(
    f"Overlaid ego trajectories, {len(overlay_files)} clips\n"
    "(rotated to a common initial heading, aligned at t=0)",
    fontsize=12, color="#222222",
)
ax.set_xlabel("meters (lateral, relative to initial heading)", fontsize=9, color="#666666")
ax.set_ylabel("meters (forward, relative to initial heading)", fontsize=9, color="#666666")
ax.grid(True, color=GRID_COLOR, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)
plt.tight_layout()
out2 = os.path.join(PLOTS_DIR, "overlaid_trajectories.png")
plt.savefig(out2, dpi=150, facecolor=BG)
plt.close(fig)
print("Saved", out2)

# ---------- Speed profile alongside one example trajectory ----------
example_file = sample_files[0]
df = pd.read_parquet(example_file)
clip_id = os.path.basename(example_file).split(".")[0]
speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2) * 3.6  # m/s -> km/h
t_s = df["timestamp"] / 1e6

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=BG)

sc = axes[0].scatter(df["x"], df["y"], c=speed, cmap=SEQ_CMAP, s=6, linewidths=0)
axes[0].plot(df["x"], df["y"], color="#00000022", linewidth=0.8, zorder=0)
axes[0].scatter([0], [0], color="#d64550", s=30, zorder=5)
axes[0].set_aspect("equal", adjustable="datalim")
axes[0].set_title(f"Trajectory colored by speed\nclip {clip_id[:8]}…", fontsize=11)
axes[0].set_xlabel("x (m)")
axes[0].set_ylabel("y (m)")
cb = fig.colorbar(sc, ax=axes[0], shrink=0.8)
cb.set_label("speed (km/h)", fontsize=9)

axes[1].plot(t_s, speed, color=LINE_COLOR, linewidth=1.5)
axes[1].axvline(0, color="#d64550", linewidth=1, linestyle="--", alpha=0.7)
axes[1].set_title("Speed over time", fontsize=11)
axes[1].set_xlabel("time relative to clip anchor (s)")
axes[1].set_ylabel("speed (km/h)")
axes[1].grid(True, color=GRID_COLOR, linewidth=0.5)
for spine in axes[1].spines.values():
    spine.set_color(GRID_COLOR)

plt.tight_layout()
out3 = os.path.join(PLOTS_DIR, "example_trajectory_speed.png")
plt.savefig(out3, dpi=150, facecolor=BG)
plt.close(fig)
print("Saved", out3)
