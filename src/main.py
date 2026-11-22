import argparse
import os
import sys

# Ensure matplotlib uses a writable temp cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add current script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from animate import create_mechanism_animation
from jansen import BASE, NAMES, generate_trajectory
from optimizer import compute_metrics, optimize, resample_by_arclength
from trajectory import load_human_trajectory


def main():
    parser = argparse.ArgumentParser(description="Jansen Linkage Dimensional Synthesis for Human Gait")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "human_trajectory.csv"),
        help="Path to input human foot trajectory CSV file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="shape",
        choices=["shape", "synchronized", "balanced"],
        help="Synthesis mode: 'shape' (arc-length geometric path), 'synchronized' (crank timing), or 'balanced'"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        default=True,
        help="Generate animated GIF of the walking mechanism"
    )
    args = parser.parse_args()

    # 1. Load human trajectory from data file
    human = load_human_trajectory(args.input)

    # 2. Generate original Jansen baseline trajectory
    original_base = generate_trajectory(BASE, n=180)
    original_c = original_base - original_base.mean(axis=0)

    # 3. Back-calculate / Optimize link dimensions
    best_L, phase, rotation, rmse = optimize(human, mode=args.mode)

    # 4. Generate optimized trajectory with recovered phase
    optimized_raw = generate_trajectory(best_L, n=180, phase=phase)

    # 5. Align trajectories for comparison
    R = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])

    optimized_aligned = (optimized_raw - optimized_raw.mean(axis=0)) @ R.T
    human_centered = human - human.mean(axis=0)

    # Scale original BASE trajectory for visual comparison
    scale_factor = np.ptp(human[:, 0]) / np.ptp(original_base[:, 0])
    original_scaled = original_c * scale_factor

    # 6. Save optimized coordinates to CSV
    opt_df = pd.DataFrame(optimized_aligned, columns=["x", "y"])
    opt_df.to_csv("optimized_trajectory.csv", index=False)

    # Save back-calculated link lengths to CSV
    links_data = []
    opt_m = best_L["m"]
    for name in NAMES:
        base_v = BASE[name]
        init_v = base_v * scale_factor
        opt_v = best_L[name]
        pct = ((opt_v - init_v) / init_v) * 100
        ratio = opt_v / opt_m
        links_data.append({
            "link": name,
            "base_jansen_mm": base_v,
            "scaled_jansen_mm": round(init_v, 2),
            "optimized_length_mm": round(opt_v, 2),
            "change_pct": round(pct, 2),
            "ratio_to_crank_m": round(ratio, 3)
        })

    links_df = pd.DataFrame(links_data)
    links_df.to_csv("back_calculated_links.csv", index=False)

    # Compute diagnostic metrics
    metrics = compute_metrics(best_L, human, optimized_aligned)

    # 7. Generate 2-Panel publication-quality comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=250)

    # Panel 1: 2D Sagittal Plane Foot Trajectory Comparison
    ax1.plot(
        human_centered[:, 0],
        human_centered[:, 1],
        "r-",
        linewidth=2.6,
        label="Target Human Gait Trajectory"
    )

    ax1.scatter(
        human_centered[::12, 0],
        human_centered[::12, 1],
        color="crimson",
        s=40,
        zorder=5,
        label="Human Gait Milestones"
    )

    ax1.plot(
        original_scaled[:, 0],
        original_scaled[:, 1],
        color="gray",
        linestyle=":",
        linewidth=1.8,
        label=f"Baseline Jansen (Scaled {scale_factor:.2f}x)"
    )

    ax1.plot(
        optimized_aligned[:, 0],
        optimized_aligned[:, 1],
        "b--",
        linewidth=2.6,
        label=f"Synthesized Jansen (Geom RMSE: {metrics['geometric_rmse']:.2f} mm)"
    )

    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xlabel("X Position (mm) [Anterior (+) / Posterior (-)]", fontsize=11)
    ax1.set_ylabel("Y Position (mm) [Vertical Elevation]", fontsize=11)
    ax1.set_title(
        f"Sagittal Foot Trajectory Comparison\nRelative Error: {metrics['geometric_rmse'] / np.ptp(human[:, 0]) * 100:.2f}% of stride",
        fontsize=12,
        fontweight="bold"
    )
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    # Panel 2: Stance Phase Ground Flatness & Clearance Profile
    # Sort points along stride to plot elevation profile vs horizontal progress
    ax2.plot(
        human_centered[:, 0],
        human_centered[:, 1],
        "r-",
        linewidth=2.2,
        label="Human Foot Clearance Profile"
    )
    ax2.plot(
        optimized_aligned[:, 0],
        optimized_aligned[:, 1],
        "b--",
        linewidth=2.2,
        label="Synthesized Jansen Profile"
    )

    # Highlight stance region
    thresh = np.percentile(optimized_aligned[:, 1], 40)
    ax2.axhline(thresh, color="darkgreen", linestyle="-.", alpha=0.7, label=f"Stance Support Level (σ = {metrics['stance_flatness_std']:.2f} mm)")

    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xlabel("Horizontal Position X (mm)", fontsize=11)
    ax2.set_ylabel("Vertical Position Y (mm)", fontsize=11)
    ax2.set_title(
        f"Stance Flatness & Ground Clearance Profile\nDuty Factor: {metrics['duty_factor_pct']:.1f}% | Flatness σ: {metrics['stance_flatness_std']:.2f} mm",
        fontsize=12,
        fontweight="bold"
    )
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=250)
    plt.close()

    # 8. Generate animation if requested
    if args.animate:
        print("\nGenerating mechanism animation...")
        create_mechanism_animation(best_L, output_path="mechanism_animation.gif", n_frames=50, fps=20)

    print("\nOptimization finished successfully!")
    print("Saved files:")
    print("  -> optimized_trajectory.csv")
    print("  -> back_calculated_links.csv")
    print("  -> trajectory_comparison.png")
    if args.animate:
        print("  -> mechanism_animation.gif")


if __name__ == "__main__":
    main()
