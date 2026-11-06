
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from trajectory import load_human_trajectory
from jansen import BASE, generate_trajectory
from optimizer import optimize


# 1. Load human trajectory
human = load_human_trajectory(
    "data/human_trajectory.csv"
)

# 2. Generate original Jansen trajectory
original = generate_trajectory(BASE, n=180)

# 3. Optimize link dimensions
best_L, phase, rotation, rmse = optimize(human)

# 4. Generate optimized trajectory
optimized = generate_trajectory(
    best_L,
    n=180,
    phase=phase
)

# 5. Align optimized output for plotting
R = np.array([
    [np.cos(rotation), -np.sin(rotation)],
    [np.sin(rotation), np.cos(rotation)]
])

optimized = (optimized - optimized.mean(axis=0)) @ R.T
human_centered = human - human.mean(axis=0)

# 6. Save optimized coordinates
pd.DataFrame(
    optimized,
    columns=["x", "y"]
).to_csv(
    "optimized_trajectory.csv",
    index=False
)

# 7. Plot comparison
plt.figure(figsize=(10, 7))

plt.plot(
    human_centered[:, 0],
    human_centered[:, 1],
    "r-",
    linewidth=2,
    label="Human trajectory"
)

plt.plot(
    optimized[:, 0],
    optimized[:, 1],
    "b--",
    linewidth=2,
    label="Optimized Jansen trajectory"
)

plt.scatter(
    human_centered[::10, 0],
    human_centered[::10, 1],
    color="red",
    s=20
)

plt.axis("equal")
plt.grid(True)
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.title(f"Human vs Jansen Foot Trajectory | RMSE: {rmse:.2f} mm")
plt.legend()

plt.savefig("trajectory_comparison.png", dpi=200)
plt.show()

print("\nOptimization finished.")
print("Saved optimized_trajectory.csv")
print("Saved trajectory_comparison.png")
