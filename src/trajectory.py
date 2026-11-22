import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

N = 180


def polygon_signed_area(pts):
    """Calculate the signed area of a 2D polygon using the Shoelace formula."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def load_human_trajectory(path, n=N):
    """
    Load human foot trajectory data from CSV and resample into `n` smooth periodic points.
    Automatically normalizes circulation direction to match the Jansen mechanism.
    """
    df = pd.read_csv(path)

    points = df[["x", "y"]].to_numpy(dtype=float)

    if len(points) < 4 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid trajectory data: need at least 4 finite 2D points")

    # If first and last points are duplicates, drop the last for periodic spline
    if np.allclose(points[0], points[-1], atol=1e-3):
        pts = points[:-1]
    else:
        pts = points

    m = len(pts)
    t_orig = np.linspace(0, 1, m, endpoint=False)
    t_extended = np.concatenate([t_orig, [1.0]])
    pts_extended = np.vstack([pts, pts[0]])

    cs_x = CubicSpline(t_extended, pts_extended[:, 0], bc_type="periodic")
    cs_y = CubicSpline(t_extended, pts_extended[:, 1], bc_type="periodic")

    t_new = np.linspace(0, 1, n, endpoint=False)
    trajectory = np.column_stack((cs_x(t_new), cs_y(t_new)))

    # Ensure consistent circulation direction (counter-clockwise / positive area)
    if polygon_signed_area(trajectory) < 0:
        trajectory = trajectory[::-1]

    return trajectory
