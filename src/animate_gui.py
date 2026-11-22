#!/usr/bin/env python3
"""
Zero-dependency Tkinter Jansen Mechanism Walk Animation.
Runs immediately with standard Python without requiring any external pip packages.
"""

import math
import os
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jansen import BASE, forward_kinematics, generate_trajectory
from trajectory import load_human_trajectory


def load_link_lengths():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "back_calculated_links.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return dict(zip(df["link"], df["optimized_length_mm"]))
    return BASE


class JansenAnimationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jansen Mechanism Kinematics - Interactive Walk Visualizer")
        self.root.geometry("1100x820")
        self.root.configure(bg="#181b22")

        self.L = load_link_lengths()
        self.trace_raw = generate_trajectory(self.L, n=240)

        # Load human trajectory if available
        human_path = os.path.join(os.path.dirname(__file__), "..", "data", "human_trajectory.csv")
        self.human_pts = None
        if os.path.exists(human_path):
            try:
                h = load_human_trajectory(human_path, n=180)
                self.human_pts = h - h.mean(axis=0)
            except Exception:
                self.human_pts = None

        # State
        self.theta = 0.0
        self.speed = tk.DoubleVar(value=90.0)  # deg per sec
        self.is_playing = True
        self.show_trace = tk.BooleanVar(value=True)
        self.show_human = tk.BooleanVar(value=True)
        self.dual_leg = tk.BooleanVar(value=True)

        self._build_ui()
        self.root.after(16, self._animation_loop)

    def _build_ui(self):
        # Top control toolbar
        control_frame = tk.Frame(self.root, bg="#21262d", padx=12, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.play_btn = tk.Button(
            control_frame,
            text="Pause",
            command=self._toggle_play,
            bg="#238636",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief="flat",
            padx=14,
            pady=4
        )
        self.play_btn.pack(side=tk.LEFT, padx=6)

        # Speed slider
        tk.Label(control_frame, text="Speed (°/s):", bg="#21262d", fg="#e6edf3", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(12, 4))
        speed_scale = tk.Scale(
            control_frame,
            from_=10,
            to=360,
            variable=self.speed,
            orient=tk.HORIZONTAL,
            bg="#21262d",
            fg="#e6edf3",
            highlightthickness=0,
            length=140
        )
        speed_scale.pack(side=tk.LEFT, padx=4)

        # Angle scrubber
        tk.Label(control_frame, text="Angle (°):", bg="#21262d", fg="#e6edf3", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(16, 4))
        self.angle_scale = tk.Scale(
            control_frame,
            from_=0,
            to=360,
            orient=tk.HORIZONTAL,
            bg="#21262d",
            fg="#e6edf3",
            highlightthickness=0,
            length=180,
            command=self._on_scrub
        )
        self.angle_scale.pack(side=tk.LEFT, padx=4)

        # Toggles
        tk.Checkbutton(
            control_frame,
            text="Foot Trace",
            variable=self.show_trace,
            bg="#21262d",
            fg="#58a6ff",
            selectcolor="#161b22",
            font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(
            control_frame,
            text="Human Gait Target",
            variable=self.show_human,
            bg="#21262d",
            fg="#f85149",
            selectcolor="#161b22",
            font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(
            control_frame,
            text="Dual Leg (180° pair)",
            variable=self.dual_leg,
            bg="#21262d",
            fg="#d2a8ff",
            selectcolor="#161b22",
            font=("Helvetica", 10)
        ).pack(side=tk.LEFT, padx=8)

        # Status text
        self.status_label = tk.Label(
            control_frame,
            text="θ: 0.0°",
            bg="#21262d",
            fg="#7ee787",
            font=("Monospace", 11, "bold")
        )
        self.status_label.pack(side=tk.RIGHT, padx=12)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="#0d1117", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self._draw_frame())

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play", bg="#238636" if self.is_playing else "#1f6feb")

    def _on_scrub(self, val):
        if not self.is_playing:
            self.theta = math.radians(float(val))
            self._draw_frame()

    def _to_screen(self, pt, width, height):
        origin_x = width // 2 + 100
        origin_y = height // 2 - 120
        scale = 0.52 * min(width / 1000.0, height / 750.0)
        sx = origin_x + pt[0] * scale
        sy = origin_y - pt[1] * scale
        return sx, sy

    def _animation_loop(self):
        if self.is_playing:
            dt = 0.016
            self.theta = (self.theta + math.radians(self.speed.get()) * dt) % (2 * math.pi)
            self.angle_scale.set(int(math.degrees(self.theta)))
            self._draw_frame()
        self.root.after(16, self._animation_loop)

    def _draw_frame(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 100 or h < 100:
            return

        # 1. Ground level line
        foot_min_y = float(np.min(self.trace_raw[:, 1]))
        _, ground_sy = self._to_screen([0, foot_min_y], w, h)
        self.canvas.create_line(0, ground_sy, w, ground_sy, fill="#30363d", width=2, dash=(6, 4))
        self.canvas.create_text(20, ground_sy + 14, text=f"Ground Contact Level ({foot_min_y:.1f} mm)", fill="#8b949e", anchor="w", font=("Helvetica", 9))

        # 2. Target human foot trajectory
        if self.show_human.get() and self.human_pts is not None:
            h_offset = [self.trace_raw[:, 0].mean(), self.trace_raw[:, 1].mean()]
            h_screen = []
            for p in self.human_pts:
                h_screen.extend(self._to_screen([p[0] + h_offset[0], p[1] + h_offset[1]], w, h))
            if len(h_screen) >= 4:
                self.canvas.create_polygon(h_screen, outline="#f85149", fill="", width=2)

        # 3. Foot trajectory trace
        if self.show_trace.get():
            trace_screen = []
            for p in self.trace_raw:
                trace_screen.extend(self._to_screen(p, w, h))
            if len(trace_screen) >= 4:
                self.canvas.create_polygon(trace_screen, outline="#58a6ff", fill="", width=2)

        # 4. Render Leg Mechanism
        def draw_leg(angle, is_paired=False):
            j = forward_kinematics(angle, self.L)
            if j is None:
                return

            bar_color = "#388bfd" if not is_paired else "#1f3b64"
            coupler_color = "#39d353" if not is_paired else "#1e5229"
            tri_fill = "#1f3558" if not is_paired else "#111b2b"
            tri_outline = "#58a6ff" if not is_paired else "#254470"

            # Triangular lower leg (W - U - T)
            w_pt = self._to_screen(j["W"], w, h)
            u_pt = self._to_screen(j["U"], w, h)
            t_pt = self._to_screen(j["T"], w, h)
            self.canvas.create_polygon([*w_pt, *u_pt, *t_pt], fill=tri_fill, outline=tri_outline, width=3)

            def bar(p1, p2, color, width=3):
                pt1 = self._to_screen(j[p1], w, h)
                pt2 = self._to_screen(j[p2], w, h)
                self.canvas.create_line(*pt1, *pt2, fill=color, width=width, capstyle=tk.ROUND)

            # Ground frame
            bar("O", "Y", "#8b949e", 3)
            # Crank
            bar("O", "Z", "#f85149" if not is_paired else "#682020", 4)
            # Upper loops
            bar("Z", "X", coupler_color, 3)
            bar("Y", "X", coupler_color, 3)
            # Lower loops
            bar("Z", "W", bar_color, 3)
            bar("Y", "W", bar_color, 3)
            # Intermediate links
            bar("X", "V", "#bc8cff" if not is_paired else "#4a2d75", 3)
            bar("Y", "V", "#bc8cff" if not is_paired else "#4a2d75", 3)
            bar("V", "U", "#bc8cff" if not is_paired else "#4a2d75", 3)

            # Joint circles
            if not is_paired:
                for name, pt in j.items():
                    sx, sy = self._to_screen(pt, w, h)
                    r = 5 if name not in ["O", "Y", "T"] else 7
                    fill = "#ffffff" if name in ["O", "Y"] else ("#f85149" if name == "T" else "#c9d1d9")
                    self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=fill, outline="#0d1117", width=1.5)
                    self.canvas.create_text(sx + 10, sy - 8, text=name, fill="#c9d1d9", font=("Helvetica", 9, "bold"))

        if self.dual_leg.get():
            draw_leg(self.theta + math.pi, is_paired=True)
        draw_leg(self.theta, is_paired=False)

        # Update status
        deg = math.degrees(self.theta)
        j_curr = forward_kinematics(self.theta, self.L)
        foot_y = j_curr["T"][1] if j_curr else 0.0
        self.status_label.config(text=f"θ: {deg:5.1f}° | Foot Y: {foot_y:.1f} mm")


def main():
    root = tk.Tk()
    app = JansenAnimationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
