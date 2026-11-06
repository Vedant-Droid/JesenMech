
import numpy as np
import pandas as pd

N = 180

def load_human_trajectory(path):
    df = pd.read_csv(path)

    points = df[["x", "y"]].to_numpy(dtype=float)

    if len(points) < 5 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid trajectory data")

    old_t = np.linspace(0, 1, len(points))
    new_t = np.linspace(0, 1, N, endpoint=False)

    x = np.interp(new_t, old_t, points[:, 0])
    y = np.interp(new_t, old_t, points[:, 1])

    return np.column_stack((x, y))
