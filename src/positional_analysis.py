#!/usr/bin/env python3
"""
Positional Analysis of the Jansen Mechanism.

Calculates the Center of Mass (CoM) coordinates (X, Y) of each moving link as a
function of the driving crank angle (theta).

The links are named using the standard Jansen letter designations (a, b, c, d, e, f, g, h, i, j, k, l, m)
matching back_calculated_links.csv.

The origin (0, 0) is set at the crank pivot axis (Joint O).

Outputs:
  1. link_centers_of_mass.csv : Full positional table vs crank angle (0° to 360°)
  2. link_com_trajectories.png: 2D trajectory plots and X/Y vs angle profiles
"""

import argparse
import os
import sys

# Ensure matplotlib uses a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add current script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jansen import BASE, forward_kinematics


def load_link_dimensions(csv_path=None):
    """Load back-calculated link lengths from CSV, or fall back to BASE."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "back_calculated_links.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return dict(zip(df["link"], df["optimized_length_mm"]))

    print(f"[INFO] '{csv_path}' not found, using BASE Jansen dimensions.")
    return BASE


def compute_link_centers_of_mass(theta, L):
    """
    Compute (x, y) coordinates of the Center of Mass (CoM) for each moving link
    for a given crank angle theta (in radians).

    The origin (0, 0) is the crank pivot axis (Joint O).

    Link names match the letters in back_calculated_links.csv:
      - link_m       : Driving crank (Bar O-Z, length m)
      - link_j       : Upper crank coupler (Bar Z-X, length j)
      - link_b       : Upper rocker (Bar Y-X, length b)
      - link_k       : Lower crank coupler (Bar Z-W, length k)
      - link_c       : Lower rocker (Bar Y-W, length c)
      - link_e       : Intermediate coupler (Bar X-V, length e)
      - link_d       : Intermediate rocker (Bar Y-V, length d)
      - link_f       : Connecting rod (Bar V-U, length f)
      - link_g       : Lower leg side (Bar W-U, length g)
      - link_h       : Lower leg side (Bar U-T, length h)
      - link_i       : Lower leg side (Bar W-T, length i)
      - triangle_ghi : Lower leg rigid triangle (centroid of Δ W-U-T)
      - foot_T       : Foot contact point (Joint T)

    Returns:
        coms: dict of {link_name: np.array([x_com, y_com])} in mm
        joints: dict of {joint_name: np.array([x, y])} in mm
    """
    joints = forward_kinematics(theta, L)
    if joints is None:
        return None, None

    O = joints["O"]  # Crank Origin (0, 0)
    Y = joints["Y"]  # Fixed chassis pivot (-a, -l)
    Z = joints["Z"]  # Crank pin
    X = joints["X"]  # Upper crank-driven joint
    W = joints["W"]  # Lower crank-driven joint
    V = joints["V"]  # Intermediate rocker joint
    U = joints["U"]  # Lower intermediate joint
    T = joints["T"]  # Foot contact point

    # Centers of Mass named by letter designations matching back_calculated_links.csv
    coms = {
        # Crank and drive loops
        "link_m": (O + Z) / 2.0,           # Crank (length m)
        "link_j": (Z + X) / 2.0,           # Upper crank coupler (length j)
        "link_b": (Y + X) / 2.0,           # Upper rocker (length b)
        "link_k": (Z + W) / 2.0,           # Lower crank coupler (length k)
        "link_c": (Y + W) / 2.0,           # Lower rocker (length c)

        # Intermediate linkages
        "link_e": (X + V) / 2.0,           # Intermediate coupler (length e)
        "link_d": (Y + V) / 2.0,           # Intermediate rocker (length d)
        "link_f": (V + U) / 2.0,           # Lower connecting rod (length f)

        # Lower triangular leg components (links g, h, i)
        "link_g": (W + U) / 2.0,           # Lower leg bar W-U (length g)
        "link_h": (U + T) / 2.0,           # Lower leg bar U-T (length h)
        "link_i": (W + T) / 2.0,           # Lower leg bar W-T (length i)
        "triangle_ghi": (W + U + T) / 3.0, # Lower leg rigid body triangle (centroid)

        # Foot landmark
        "foot_T": T,
    }

    return coms, joints


def run_positional_analysis(L, n_steps=360):
    """
    Run full 360° positional analysis for n_steps angles.
    Returns a pandas DataFrame.
    """
    angles_rad = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    angles_deg = np.rad2deg(angles_rad)

    rows = []
    for rad, deg in zip(angles_rad, angles_deg):
        coms, joints = compute_link_centers_of_mass(rad, L)

        row = {
            "crank_angle_deg": round(deg, 2),
            "crank_angle_rad": round(rad, 5),
        }

        # Add CoM coordinates
        for link_name, pt in coms.items():
            row[f"{link_name}_x"] = round(float(pt[0]), 3)
            row[f"{link_name}_y"] = round(float(pt[1]), 3)

        # Add joint coordinates
        for joint_name, pt in joints.items():
            row[f"joint_{joint_name}_x"] = round(float(pt[0]), 3)
            row[f"joint_{joint_name}_y"] = round(float(pt[1]), 3)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_positional_analysis(df, output_img="link_com_trajectories.png"):
    """
    Generate publication-quality 3-panel figure:
      1. 2D CoM trajectories in the mechanism frame (Origin = Crank Pivot O)
      2. X_CoM vs Crank Angle (Horizontal harmonic motion)
      3. Y_CoM vs Crank Angle (Vertical oscillation)
    """
    fig = plt.figure(figsize=(16, 9), dpi=220)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    ax_2d = fig.add_subplot(gs[:, 0])
    ax_x = fig.add_subplot(gs[0, 1])
    ax_y = fig.add_subplot(gs[1, 1])

    # Distinct colors and descriptive labels for each lettered link
    color_map = {
        "link_m": "#e63946",       # Red
        "link_j": "#f4a261",       # Orange
        "link_b": "#2a9d8f",       # Teal
        "link_k": "#457b9d",       # Steel Blue
        "link_c": "#1d3557",       # Navy Blue
        "link_e": "#9b5de5",       # Purple
        "link_d": "#f15bb5",       # Pink
        "link_f": "#00bbf9",       # Cyan
        "link_g": "#52b788",       # Sage Green
        "link_h": "#38b000",       # Bright Green
        "link_i": "#007200",       # Forest Green
        "triangle_ghi": "#ffd166", # Gold / Yellow
        "foot_T": "#ff006e",       # Hot Pink
    }

    labels = {
        "link_m": "Link m (Crank O-Z)",
        "link_j": "Link j (Coupler Z-X)",
        "link_b": "Link b (Rocker Y-X)",
        "link_k": "Link k (Coupler Z-W)",
        "link_c": "Link c (Rocker Y-W)",
        "link_e": "Link e (Coupler X-V)",
        "link_d": "Link d (Rocker Y-V)",
        "link_f": "Link f (Rod V-U)",
        "link_g": "Link g (Side W-U)",
        "link_h": "Link h (Side U-T)",
        "link_i": "Link i (Side W-T)",
        "triangle_ghi": "Triangle ghi (Leg W-U-T)",
        "foot_T": "Foot Point T",
    }

    # 1. Panel 1: 2D Spatial Trajectories
    for link, color in color_map.items():
        x = df[f"{link}_x"]
        y = df[f"{link}_y"]
        lw = 2.4 if link in ["link_m", "triangle_ghi", "foot_T"] else 1.6
        ls = "-" if link != "foot_T" else ":"
        ax_2d.plot(x, y, color=color, linewidth=lw, linestyle=ls, label=labels[link])

        # Mark initial position at theta = 0
        ax_2d.scatter(x.iloc[0], y.iloc[0], color=color, s=26, zorder=5)

    # Mark Fixed Pivots: Crank Origin O(0,0) and Chassis Pivot Y(-a, -l)
    ax_2d.plot(0, 0, marker="o", color="black", markersize=9, zorder=6)
    ax_2d.text(8, 8, "ORIGIN (0,0)\nCrank Pivot O", fontsize=9, fontweight="bold", color="black")

    y_fixed_x = df["joint_Y_x"].iloc[0]
    y_fixed_y = df["joint_Y_y"].iloc[0]
    ax_2d.plot(y_fixed_x, y_fixed_y, marker="s", color="dimgray", markersize=8, zorder=6)
    ax_2d.text(y_fixed_x - 140, y_fixed_y - 22, f"Chassis Pivot Y (-a, -l)\n({y_fixed_x:.1f}, {y_fixed_y:.1f})", fontsize=8.5, color="dimgray")

    ax_2d.set_aspect("equal")
    ax_2d.grid(True, linestyle="--", alpha=0.55)
    ax_2d.set_xlabel("X Position (mm) [Relative to Crank Pivot (0,0)]", fontsize=11, fontweight="bold")
    ax_2d.set_ylabel("Y Position (mm) [Relative to Crank Pivot (0,0)]", fontsize=11, fontweight="bold")
    ax_2d.set_title("2D Centers of Mass Trajectories\n(Origin at Crank Pivot Axis O)", fontsize=13, fontweight="bold")
    ax_2d.legend(loc="lower left", fontsize=7.5, framealpha=0.92, ncol=2)

    # 2. Panel 2: X_CoM vs Crank Angle
    angles = df["crank_angle_deg"]
    for link in ["link_m", "link_j", "link_b", "link_k", "link_c", "link_e", "link_d", "link_f", "triangle_ghi"]:
        ax_x.plot(angles, df[f"{link}_x"], color=color_map[link], linewidth=1.8, label=labels[link])

    ax_x.set_xlim(0, 360)
    ax_x.grid(True, linestyle="--", alpha=0.55)
    ax_x.set_xlabel("Crank Angle θ (degrees)", fontsize=10, fontweight="bold")
    ax_x.set_ylabel("X Position (mm)", fontsize=10, fontweight="bold")
    ax_x.set_title("Horizontal CoM Displacement vs Crank Angle", fontsize=11, fontweight="bold")
    ax_x.set_xticks(np.arange(0, 361, 45))

    # 3. Panel 3: Y_CoM vs Crank Angle
    for link in ["link_m", "link_j", "link_b", "link_k", "link_c", "link_e", "link_d", "link_f", "triangle_ghi"]:
        ax_y.plot(angles, df[f"{link}_y"], color=color_map[link], linewidth=1.8, label=labels[link])

    ax_y.set_xlim(0, 360)
    ax_y.grid(True, linestyle="--", alpha=0.55)
    ax_y.set_xlabel("Crank Angle θ (degrees)", fontsize=10, fontweight="bold")
    ax_y.set_ylabel("Y Position (mm)", fontsize=10, fontweight="bold")
    ax_y.set_title("Vertical CoM Displacement vs Crank Angle", fontsize=11, fontweight="bold")
    ax_y.set_xticks(np.arange(0, 361, 45))

    plt.suptitle("Jansen Mechanism Kinematic Positional Analysis: Link Centers of Mass (a, b, c, ... m)", fontsize=14, fontweight="bold", y=0.98)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97, hspace=0.32, wspace=0.25)
    plt.savefig(output_img, dpi=250)
    plt.close()
    print(f"Saved plot to '{output_img}'")


def main():
    parser = argparse.ArgumentParser(description="Jansen Mechanism Link Centers of Mass Positional Analysis")
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="Query a single crank angle in degrees (e.g. --angle 45)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=360,
        help="Number of angular steps for 360° sweep (default: 360, i.e., 1° per step)"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="link_centers_of_mass.csv",
        help="Output CSV path (default: link_centers_of_mass.csv)"
    )
    parser.add_argument(
        "--output-img",
        type=str,
        default="link_com_trajectories.png",
        help="Output image path (default: link_com_trajectories.png)"
    )
    parser.add_argument(
        "--links-csv",
        type=str,
        default=None,
        help="Path to back_calculated_links.csv (defaults to back_calculated_links.csv)"
    )
    args = parser.parse_args()

    # 1. Load dimensions
    L = load_link_dimensions(args.links_csv)

    # If single angle query
    if args.angle is not None:
        rad = np.deg2rad(args.angle)
        coms, joints = compute_link_centers_of_mass(rad, L)
        print(f"\n=======================================================")
        print(f" POSITIONAL ANALYSIS AT CRANK ANGLE θ = {args.angle:.1f}°")
        print(f" Origin: Crank Pivot Axis O(0, 0)")
        print(f"=======================================================")
        print(f"{'Link Name':<16} {'Topology / Bars':<20} {'X_CoM (mm)':<13} {'Y_CoM (mm)':<13}")
        print("-" * 65)

        letter_desc = {
            "link_m": "Crank O-Z (length m)",
            "link_j": "Coupler Z-X (length j)",
            "link_b": "Rocker Y-X (length b)",
            "link_k": "Coupler Z-W (length k)",
            "link_c": "Rocker Y-W (length c)",
            "link_e": "Coupler X-V (length e)",
            "link_d": "Rocker Y-V (length d)",
            "link_f": "Connecting Rod V-U (f)",
            "link_g": "Leg Side W-U (length g)",
            "link_h": "Leg Side U-T (length h)",
            "link_i": "Leg Side W-T (length i)",
            "triangle_ghi": "Leg Triangle Δ WUT",
            "foot_T": "Foot Contact Point T",
        }

        for link, pt in coms.items():
            desc = letter_desc.get(link, "")
            print(f"{link:<16} {desc:<20} {pt[0]:<13.2f} {pt[1]:<13.2f}")
        print("=" * 65)
        print(f"Chassis Pivot Y: X = -a = {-L['a']:.2f} mm, Y = -l = {-L['l']:.2f} mm\n")
        return

    # 2. Run full 360° sweep
    print(f"Running full 360° positional analysis ({args.steps} steps)...")
    df = run_positional_analysis(L, n_steps=args.steps)

    # 3. Save CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Saved positional data to '{args.output_csv}' ({len(df)} rows)")

    # 4. Generate plots
    plot_positional_analysis(df, output_img=args.output_img)

    print("\nPositional analysis completed successfully!")


if __name__ == "__main__":
    main()
