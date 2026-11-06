
import numpy as np

# Standard Jansen linkage dimensions (mm)
BASE = {
    "m": 15.0,
    "j": 50.0,
    "b": 41.5,
    "k": 61.9,
    "c": 39.3,
    "d": 40.1,
    "e": 55.8,
    "f": 39.4,
    "g": 36.7,
    "h": 65.7,
    "i": 49.0,
    "a": 38.0,
    "l": 7.8,
}

NAMES = list(BASE.keys())


def circle_intersections(p1, r1, p2, r2):
    """Return the two intersections of two circles."""

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    delta = p2 - p1
    distance = np.linalg.norm(delta)

    if distance < 1e-10:
        raise ValueError("Coincident circle centres")

    if distance > r1 + r2 or distance < abs(r1 - r2):
        raise ValueError("Impossible linkage geometry")

    x = (r1**2 - r2**2 + distance**2) / (2 * distance)

    height_sq = r1**2 - x**2

    if height_sq < -1e-7:
        raise ValueError("No real circle intersection")

    height = np.sqrt(max(0.0, height_sq))

    unit = delta / distance
    perpendicular = np.array([-unit[1], unit[0]])

    midpoint = p1 + x * unit

    return [
        midpoint + height * perpendicular,
        midpoint - height * perpendicular
    ]


def choose(p1, r1, p2, r2, mode, axis):
    points = circle_intersections(p1, r1, p2, r2)

    if mode == "max":
        return max(points, key=lambda p: p[axis])

    return min(points, key=lambda p: p[axis])


def forward_kinematics(theta, L):
    """
    Calculate all joint positions for a crank angle.
    Coordinates are in mm.
    """

    O = np.array([0.0, 0.0])
    Y = np.array([-L["a"], -L["l"]])

    Z = L["m"] * np.array([
        np.cos(theta),
        np.sin(theta)
    ])

    # Upper and lower crank-driven joints
    X = choose(Z, L["j"], Y, L["b"], "max", 1)
    W = choose(Z, L["k"], Y, L["c"], "min", 1)

    # Intermediate joints
    V = choose(Y, L["d"], X, L["e"], "min", 0)
    U = choose(V, L["f"], W, L["g"], "min", 0)

    # Foot joint
    T = choose(W, L["i"], U, L["h"], "min", 1)

    return {
        "O": O,
        "Y": Y,
        "Z": Z,
        "X": X,
        "W": W,
        "V": V,
        "U": U,
        "T": T
    }


def generate_trajectory(L, n=180, phase=0.0):
    angles = np.linspace(
        0, 2 * np.pi, n, endpoint=False
    ) + phase

    points = []

    for angle in angles:
        joints = forward_kinematics(angle, L)
        points.append(joints["T"])

    return np.array(points)
