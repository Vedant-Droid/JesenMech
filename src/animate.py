import os
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from jansen import forward_kinematics, generate_trajectory


def create_mechanism_animation(
    L,
    output_path="mechanism_animation.gif",
    n_frames=60,
    fps=20,
    dpi=120
):
    """
    Generate an animated GIF of the Jansen mechanism linkage stepping through 360 deg of rotation.
    Shows all moving bars, pivots, the triangular lower leg, and the traced foot locus.
    """
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    foot_trace = generate_trajectory(L, n=120)

    # Compute bounding box
    all_x = foot_trace[:, 0].tolist() + [-L["a"], 0.0]
    all_y = foot_trace[:, 1].tolist() + [-L["l"], 0.0]
    margin = 80
    x_min, x_max = min(all_x) - margin * 1.5, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin * 2

    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
    frames = []

    def plot_bar(p1, p2, color="slategray", lw=2.2, ls="-"):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=lw, linestyle=ls)

    for idx, angle in enumerate(angles):
        ax.clear()
        joints = forward_kinematics(angle, L)

        # 1. Fixed ground link (O to Y)
        plot_bar(joints["O"], joints["Y"], color="black", lw=2.5, ls="--")

        # 2. Driving crank (O to Z)
        plot_bar(joints["O"], joints["Z"], color="crimson", lw=3.2)

        # 3. Upper drive loop (Z-X and Y-X)
        plot_bar(joints["Z"], joints["X"], color="teal", lw=2.4)
        plot_bar(joints["Y"], joints["X"], color="teal", lw=2.4)

        # 4. Lower drive loop (Z-W and Y-W)
        plot_bar(joints["Z"], joints["W"], color="navy", lw=2.4)
        plot_bar(joints["Y"], joints["W"], color="navy", lw=2.4)

        # 5. Intermediate links (X-V, Y-V, V-U)
        plot_bar(joints["X"], joints["V"], color="darkorchid", lw=2.2)
        plot_bar(joints["Y"], joints["V"], color="darkorchid", lw=2.2)
        plot_bar(joints["V"], joints["U"], color="darkorchid", lw=2.2)

        # 6. Lower leg rigid triangle (W - U - T)
        tri_x = [joints["W"][0], joints["U"][0], joints["T"][0], joints["W"][0]]
        tri_y = [joints["W"][1], joints["U"][1], joints["T"][1], joints["W"][1]]
        ax.fill(tri_x, tri_y, color="royalblue", alpha=0.25)
        ax.plot(tri_x, tri_y, color="royalblue", lw=2.6)

        # 7. Traced foot path (point T)
        ax.plot(
            foot_trace[:, 0],
            foot_trace[:, 1],
            color="dodgerblue",
            linestyle=":",
            linewidth=1.8,
            alpha=0.7,
            label="Traced Foot Path (Joint T)"
        )

        # 8. Mark all joints
        for name, pt in joints.items():
            if name in ["O", "Y"]:
                ax.plot(pt[0], pt[1], "ks", markersize=8, zorder=6)
                ax.text(pt[0] + 10, pt[1] + 10, f"{name} (Pivot)", fontsize=9, fontweight="bold")
            elif name == "T":
                ax.plot(pt[0], pt[1], "o", color="red", markersize=9, zorder=6)
                ax.text(pt[0] + 10, pt[1] - 15, "T (Foot)", fontsize=10, fontweight="bold", color="red")
            else:
                ax.plot(pt[0], pt[1], "o", color="dimgray", markersize=6, zorder=5)
                ax.text(pt[0] + 8, pt[1] + 8, name, fontsize=8, color="dimgray")

        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_title(
            f"Jansen Leg Mechanism Kinematics (Crank θ = {np.rad2deg(angle):.0f}°)",
            fontsize=12,
            fontweight="bold"
        )
        ax.set_xlabel("X Position (mm)", fontsize=10)
        ax.set_ylabel("Y Position (mm)", fontsize=10)
        ax.legend(loc="upper right", framealpha=0.85)

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba))

    plt.close(fig)

    duration_ms = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"Saved animation to {output_path} ({n_frames} frames @ {fps} fps)")
