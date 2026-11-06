
import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution, least_squares

from jansen import (
    BASE,
    NAMES,
    generate_trajectory
)


N = 180


def load_human_trajectory(path):
    df = pd.read_csv(path)

    points = df[["x", "y"]].to_numpy(dtype=float)

    if len(points) < 5 or not np.all(np.isfinite(points)):
        raise ValueError("Invalid trajectory data")

    # Resample uniformly in time/sample index.
    old_t = np.linspace(0, 1, len(points))
    new_t = np.linspace(0, 1, N, endpoint=False)

    x = np.interp(new_t, old_t, points[:, 0])
    y = np.interp(new_t, old_t, points[:, 1])

    return np.column_stack((x, y))


def center(points):
    return points - np.mean(points, axis=0)


def rotate(points, angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return points @ R.T


def get_dimensions(values):
    return dict(zip(NAMES, values[:len(NAMES)]))


def residual(values, human):
    L = get_dimensions(values)

    phase = values[-2]
    rotation = values[-1]

    try:
        simulated = generate_trajectory(
            L, n=N, phase=phase
        )

        simulated = rotate(
            center(simulated), rotation
        )

        target = center(human)

        return (simulated - target).flatten()

    except (ValueError, FloatingPointError):
        return np.full(N * 2, 1e5)


def objective(values, human):
    errors = residual(values, human)
    return np.mean(errors**2)


def optimize(human):
    # Scale the standard mechanism to the measured stride width.
    baseline = generate_trajectory(BASE, n=N)

    target_width = np.ptp(human[:, 0])
    baseline_width = np.ptp(baseline[:, 0])

    scale = target_width / baseline_width

    initial = np.array([
        BASE[name] * scale for name in NAMES
    ])

    print("Initial mechanism scale:", scale)

    # Allow each dimension to vary by +/- 25%.
    bounds = [
        (value * 0.75, value * 1.25)
        for value in initial
    ]

    # Phase and orientation are also optimized.
    bounds += [
        (0, 2 * np.pi),
        (-np.pi, np.pi)
    ]

    print("Starting Differential Evolution...")

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(human,),
        maxiter=150,
        popsize=12,
        tol=1e-5,
        polish=False,
        workers=1,
        seed=42
    )

    print("\nDifferential Evolution completed.")
    print("Initial best error:", result.fun)

    print("\nStarting least-squares refinement...")

    refined = least_squares(
        residual,
        result.x,
        bounds=(
            [b[0] for b in bounds],
            [b[1] for b in bounds]
        ),
        args=(human,),
        max_nfev=1000,
        verbose=1
    )

    print("\nOptimized dimensions:")

    L = get_dimensions(refined.x)

    for name, value in L.items():
        print(f"{name}: {value:.3f} mm")

    print("\nFinal RMSE:")

    errors = residual(refined.x, human)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"{rmse:.3f} mm")

    return L, refined.x[-2], refined.x[-1], rmse
