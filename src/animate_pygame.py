#!/usr/bin/env python3
"""
Interactive Pygame Jansen Mechanism Walk Animation.

Controls:
  SPACE       : Play / Pause
  UP / DOWN   : Increase / Decrease rotation speed
  LEFT / RIGHT: Step crank angle manually (when paused)
  T           : Toggle foot trajectory trail
  H           : Toggle target human foot trajectory overlay
  L           : Toggle single leg / dual leg (180 deg out-of-phase) mode
  R           : Reset to 0 deg
  ESC / Q     : Quit
"""

import math
import os
import sys

import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jansen import BASE, forward_kinematics, generate_trajectory
from trajectory import load_human_trajectory

try:
    import pygame
except ImportError:
    print("\n[ERROR] pygame is not installed in this environment.")
    print("Please install pygame via:")
    print("    pip install pygame")
    print("Or run the zero-dependency Tkinter desktop animation:")
    print("    python3 src/animate_gui.py\n")
    sys.exit(1)


def load_link_lengths():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "back_calculated_links.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return dict(zip(df["link"], df["optimized_length_mm"]))
    print("[INFO] back_calculated_links.csv not found, using BASE dimensions.")
    return BASE


def main():
    pygame.init()
    pygame.display.set_caption("Jansen Mechanism Kinematics - Human Gait Synthesized")

    WIDTH, HEIGHT = 1100, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    L = load_link_lengths()

    # Load target human trajectory if available
    human_path = os.path.join(os.path.dirname(__file__), "..", "data", "human_trajectory.csv")
    human_pts = None
    if os.path.exists(human_path):
        try:
            h = load_human_trajectory(human_path, n=180)
            human_pts = h - h.mean(axis=0)
        except Exception:
            human_pts = None

    # Precalculate foot trace relative to crank origin O(0,0)
    trace_raw = generate_trajectory(L, n=240)

    # Coordinate transform: map mechanism mm to screen pixels
    # Mechanism X ~ [-550, 300], Y ~ [-900, 250]
    origin_screen = [WIDTH // 2 + 100, HEIGHT // 2 - 120]
    scale = 0.55  # pixels per mm

    def to_screen(pt):
        # pt is [x, y] in mechanism mm
        # Screen Y is inverted (down is positive)
        sx = int(origin_screen[0] + pt[0] * scale)
        sy = int(origin_screen[1] - pt[1] * scale)
        return (sx, sy)

    # State variables
    theta = 0.0
    speed_deg_per_sec = 120.0
    paused = False
    show_trace = True
    show_human = True
    dual_leg = True

    # Color definitions
    BG_COLOR = (24, 27, 34)
    TEXT_COLOR = (220, 225, 235)
    MUTED_COLOR = (140, 150, 165)
    GROUND_FRAME_COLOR = (90, 100, 115)
    CRANK_COLOR = (240, 70, 70)
    UPPER_LOOP_COLOR = (50, 180, 180)
    LOWER_LOOP_COLOR = (70, 120, 220)
    INTERMEDIATE_COLOR = (180, 90, 220)
    TRIANGLE_FILL = (40, 80, 160, 110)
    TRIANGLE_OUTLINE = (65, 130, 245)
    FOOT_TRACE_COLOR = (0, 220, 255)
    HUMAN_COLOR = (255, 90, 90)

    font = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 16)
    title_font = pygame.font.SysFont("DejaVu Sans, Helvetica, Arial", 20, bold=True)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # seconds elapsed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                origin_screen = [WIDTH // 2 + 100, HEIGHT // 2 - 120]
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_deg_per_sec = min(400.0, speed_deg_per_sec + 25.0)
                elif event.key == pygame.K_DOWN:
                    speed_deg_per_sec = max(10.0, speed_deg_per_sec - 25.0)
                elif event.key == pygame.K_LEFT:
                    theta = (theta - math.radians(5.0)) % (2 * math.pi)
                elif event.key == pygame.K_RIGHT:
                    theta = (theta + math.radians(5.0)) % (2 * math.pi)
                elif event.key == pygame.K_t:
                    show_trace = not show_trace
                elif event.key == pygame.K_h:
                    show_human = not show_human
                elif event.key == pygame.K_l:
                    dual_leg = not dual_leg
                elif event.key == pygame.K_r:
                    theta = 0.0

        if not paused:
            theta = (theta + math.radians(speed_deg_per_sec) * dt) % (2 * math.pi)

        screen.fill(BG_COLOR)

        # 1. Draw grid / ground reference
        foot_min_y = np.min(trace_raw[:, 1])
        ground_sy = to_screen([0, foot_min_y])[1]
        pygame.draw.line(screen, (45, 55, 70), (0, ground_sy), (WIDTH, ground_sy), 2)
        ground_text = font.render(f"Ground Contact Level ({foot_min_y:.1f} mm)", True, (110, 125, 145))
        screen.blit(ground_text, (20, ground_sy + 6))

        # 2. Draw Target Human Foot Trajectory if enabled
        if show_human and human_pts is not None:
            # Shift human trajectory so its bottom aligns with ground level
            h_center_offset = [trace_raw[:, 0].mean(), trace_raw[:, 1].mean()]
            h_screen_pts = [to_screen([pt[0] + h_center_offset[0], pt[1] + h_center_offset[1]]) for pt in human_pts]
            if len(h_screen_pts) > 2:
                pygame.draw.lines(screen, HUMAN_COLOR, True, h_screen_pts, 2)

        # 3. Draw Foot Trajectory Trail
        if show_trace:
            trace_screen_pts = [to_screen(pt) for pt in trace_raw]
            pygame.draw.lines(screen, FOOT_TRACE_COLOR, True, trace_screen_pts, 2)

        # Helper function to render one leg mechanism at a given crank angle
        def render_leg(crank_angle, alpha_multiplier=1.0):
            joints = forward_kinematics(crank_angle, L)
            if joints is None:
                return

            def draw_link(p1_name, p2_name, color, thickness=3):
                p1 = to_screen(joints[p1_name])
                p2 = to_screen(joints[p2_name])
                col = tuple(int(c * alpha_multiplier) for c in color[:3])
                pygame.draw.line(screen, col, p1, p2, thickness)

            # Triangular lower leg polygon
            tri_pts = [to_screen(joints["W"]), to_screen(joints["U"]), to_screen(joints["T"])]
            tri_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            poly_color = (*TRIANGLE_FILL[:3], int(TRIANGLE_FILL[3] * alpha_multiplier))
            pygame.draw.polygon(tri_surface, poly_color, tri_pts)
            screen.blit(tri_surface, (0, 0))
            pygame.draw.polygon(screen, tuple(int(c * alpha_multiplier) for c in TRIANGLE_OUTLINE), tri_pts, 3)

            # Link bars
            draw_link("O", "Y", GROUND_FRAME_COLOR, 3)
            draw_link("O", "Z", CRANK_COLOR, 4)
            draw_link("Z", "X", UPPER_LOOP_COLOR, 3)
            draw_link("Y", "X", UPPER_LOOP_COLOR, 3)
            draw_link("Z", "W", LOWER_LOOP_COLOR, 3)
            draw_link("Y", "W", LOWER_LOOP_COLOR, 3)
            draw_link("X", "V", INTERMEDIATE_COLOR, 3)
            draw_link("Y", "V", INTERMEDIATE_COLOR, 3)
            draw_link("V", "U", INTERMEDIATE_COLOR, 3)

            # Joint circles
            for name, pt in joints.items():
                spt = to_screen(pt)
                if name in ["O", "Y"]:
                    pygame.draw.circle(screen, (255, 255, 255), spt, 6)
                    pygame.draw.circle(screen, (30, 30, 30), spt, 3)
                elif name == "T":
                    pygame.draw.circle(screen, (255, 50, 50), spt, 8)
                    pygame.draw.circle(screen, (255, 255, 255), spt, 4)
                else:
                    pygame.draw.circle(screen, (200, 210, 225), spt, 5)

        # Draw paired leg (180 deg out-of-phase) in dimmed color if dual_leg is enabled
        if dual_leg:
            render_leg(theta + math.pi, alpha_multiplier=0.4)

        # Draw primary leg
        render_leg(theta, alpha_multiplier=1.0)

        # 4. On-Screen Display HUD
        deg = math.degrees(theta)
        title_surf = title_font.render("Jansen Leg Kinematics - Synthesized Walking Mechanism", True, TEXT_COLOR)
        screen.blit(title_surf, (20, 20))

        status_text = f"Crank Angle θ: {deg:5.1f}° | Speed: {speed_deg_per_sec:.0f}°/s | Status: {'PAUSED' if paused else 'RUNNING'}"
        screen.blit(font.render(status_text, True, TEXT_COLOR), (20, 50))

        # Instructions HUD
        hud_lines = [
            "Controls:",
            "  SPACE        Play / Pause",
            "  UP / DOWN    Change Speed",
            "  LEFT / RIGHT Step Angle",
            "  T            Toggle Foot Trace",
            "  H            Toggle Human Target Curve",
            "  L            Toggle Dual Leg Mode",
            "  R            Reset to 0°",
        ]
        hud_y = HEIGHT - 200
        for line in hud_lines:
            screen.blit(font.render(line, True, MUTED_COLOR), (20, hud_y))
            hud_y += 22

        # Legend
        leg_x = WIDTH - 270
        leg_y = 25
        pygame.draw.line(screen, CRANK_COLOR, (leg_x, leg_y + 8), (leg_x + 25, leg_y + 8), 3)
        screen.blit(font.render("Crank (O - Z)", True, TEXT_COLOR), (leg_x + 35, leg_y))

        pygame.draw.line(screen, FOOT_TRACE_COLOR, (leg_x, leg_y + 33), (leg_x + 25, leg_y + 33), 2)
        screen.blit(font.render("Synthesized Foot Locus", True, TEXT_COLOR), (leg_x + 35, leg_y + 25))

        if show_human:
            pygame.draw.line(screen, HUMAN_COLOR, (leg_x, leg_y + 58), (leg_x + 25, leg_y + 58), 2)
            screen.blit(font.render("Target Human Gait", True, TEXT_COLOR), (leg_x + 35, leg_y + 50))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
