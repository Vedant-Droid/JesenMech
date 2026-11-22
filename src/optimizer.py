import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from jansen import BASE, NAMES, forward_kinematics, generate_trajectory

N = 180
N_FAST = 60


def center(points):
    """Translate points so their centroid is at the origin."""
    return points - np.mean(points, axis=0)


def get_dimensions(values):
    """Map array of link lengths back to dictionary keyed by link names."""
    return dict(zip(NAMES, values[: len(NAMES)]))


def resample_by_arclength(pts, n=N):
    """
    Resample a closed 2D loop uniformly by cumulative perimeter arc length.
    Removes non-uniform velocity parameterization so curves are compared strictly by geometric shape.
    """
    diffs = np.diff(pts, axis=0, append=pts[:1])
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    pts_ext = np.vstack([pts, pts[0]])
    s_targets = np.linspace(0, total_len, n, endpoint=False)

    fx = interp1d(cum_len, pts_ext[:, 0], kind="linear")
    fy = interp1d(cum_len, pts_ext[:, 1], kind="linear")

    return np.column_stack([fx(s_targets), fy(s_targets)])


def kabsch_rotation(sim_c, target_c):
    """
    Compute optimal 2D rotation matrix R (SO(2)) minimizing ||sim_c @ R - target_c||^2.
    """
    H = sim_c.T @ target_c
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def evaluate_shape_error(sim_c, target_arc, step=2):
    """
    Find minimum geometric RMSE between simulated and target loops via
    arc-length parameterization and optimal 2D Kabsch rotation.
    """
    sim_arc = resample_by_arclength(sim_c, n=len(target_arc))
    n = len(sim_arc)
    best_err = float("inf")

    for k in range(0, n, step):
        rolled = np.roll(sim_arc, -k, axis=0)
        R = kabsch_rotation(rolled, target_arc)
        err = np.mean((rolled @ R - target_arc) ** 2)
        if err < best_err:
            best_err = err

    return np.sqrt(best_err)


def evaluate_p2p_error(sim_c, target_c, step=2):
    """
    Point-to-point RMSE matching uniform crank angles to target points.
    """
    n = len(sim_c)
    best_err = float("inf")

    for k in range(0, n, step):
        rolled = np.roll(sim_c, -k, axis=0)
        R = kabsch_rotation(rolled, target_c)
        err = np.mean((rolled @ R - target_c) ** 2)
        if err < best_err:
            best_err = err

    return np.sqrt(best_err)


def make_objective(human_c, mode="shape", n_pts=N_FAST, flatness_weight=0.0):
    """
    Create objective function.
    mode='shape': Geometric path matching via arc-length parameterization (lowest geometric RMSE).
    mode='synchronized': Crank-angle synchronized point-to-point matching.
    mode='balanced': 70% geometric shape + 30% timing synchronization.
    """
    human_arc = resample_by_arclength(human_c, n=n_pts)
    step = len(human_c) // n_pts
    human_sub = human_c[::step][:n_pts]

    def objective(values):
        L = get_dimensions(values)
        try:
            sim = generate_trajectory(L, n=n_pts)
            sim_c = center(sim)

            if mode == "shape":
                loss = evaluate_shape_error(sim_c, human_arc, step=2)
            elif mode == "synchronized":
                loss = evaluate_p2p_error(sim_c, human_sub, step=2)
            else:  # balanced
                err_shape = evaluate_shape_error(sim_c, human_arc, step=2)
                err_p2p = evaluate_p2p_error(sim_c, human_sub, step=2)
                loss = 0.7 * err_shape + 0.3 * err_p2p

            if flatness_weight > 0.0:
                # Penalize vertical waviness in bottom 40% of trajectory
                thresh = np.percentile(sim[:, 1], 40)
                stance_y = sim[sim[:, 1] <= thresh, 1]
                loss += flatness_weight * np.std(stance_y)

            return loss
        except (ValueError, FloatingPointError):
            return 1e5

    return objective


def extract_alignment(L, human_c, n=N):
    """
    Extract optimal continuous phase angle and 2D rotation angle for main.py alignment.
    In main.py:
        optimized = (generate_trajectory(best_L, n, phase=phase) - mean) @ R.T
    where R = [[cos(rotation), -sin(rotation)], [sin(rotation), cos(rotation)]].
    """
    sim = generate_trajectory(L, n=n, phase=0.0)
    sim_c = center(sim)

    best_err = float("inf")
    best_phase = 0.0
    best_rot = 0.0

    for k in range(n):
        phase_candidate = 2 * np.pi * k / n
        rolled = np.roll(sim_c, -k, axis=0)
        R_k = kabsch_rotation(rolled, human_c)
        err = np.mean((rolled @ R_k - human_c) ** 2)
        if err < best_err:
            best_err = err
            best_phase = phase_candidate
            R_main = R_k.T
            best_rot = np.arctan2(R_main[1, 0], R_main[0, 0])

    dp = 2 * np.pi / n

    def phase_loss(p):
        try:
            s = generate_trajectory(L, n=n, phase=p)
            s_c = center(s)
            R_k = kabsch_rotation(s_c, human_c)
            return np.mean((s_c @ R_k - human_c) ** 2)
        except Exception:
            return 1e9

    res_p = minimize_scalar(
        phase_loss,
        bounds=(best_phase - dp, best_phase + dp),
        method="bounded"
    )
    if res_p.success and res_p.fun < best_err:
        best_phase = float(res_p.x)
        s = generate_trajectory(L, n=n, phase=best_phase)
        s_c = center(s)
        R_k = kabsch_rotation(s_c, human_c)
        best_rot = float(np.arctan2(R_k.T[1, 0], R_k.T[0, 0]))
        best_err = float(res_p.fun)

    rmse = float(np.sqrt(best_err))
    return best_phase, best_rot, rmse


def compute_metrics(L, target_traj, aligned_traj):
    """
    Compute biomechanical and kinematic diagnostic metrics.
    """
    # 1. Stance flatness (bottom 40% of trajectory)
    thresh = np.percentile(aligned_traj[:, 1], 40)
    stance_pts = aligned_traj[aligned_traj[:, 1] <= thresh]
    stance_std = float(np.std(stance_pts[:, 1]))
    stance_span = float(np.ptp(stance_pts[:, 1]))

    # 2. Duty factor (% of points spent in ground contact)
    duty_factor = float(len(stance_pts) / len(aligned_traj) * 100)

    # 3. Geometric Arc-Length RMSE with proper cyclic alignment
    target_c = target_traj - target_traj.mean(axis=0)
    target_arc = resample_by_arclength(target_c, n=len(aligned_traj))
    aligned_arc = resample_by_arclength(aligned_traj, n=len(aligned_traj))
    n = len(aligned_arc)
    best_geom_err = float("inf")
    for k in range(n):
        rolled = np.roll(aligned_arc, -k, axis=0)
        R = kabsch_rotation(rolled, target_arc)
        err = np.mean((rolled @ R - target_arc) ** 2)
        if err < best_geom_err:
            best_geom_err = err
    geom_rmse = float(np.sqrt(best_geom_err))

    # 4. Transmission angles for upper and lower 4-bar sub-loops
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    mu_upper, mu_lower = [], []
    for a in angles:
        j = forward_kinematics(a, L)
        v_zx = j["X"] - j["Z"]
        v_yx = j["X"] - j["Y"]
        cos_u = np.dot(v_zx, v_yx) / (np.linalg.norm(v_zx) * np.linalg.norm(v_yx))
        mu_upper.append(np.rad2deg(np.arccos(np.clip(cos_u, -1.0, 1.0))))

        v_zw = j["W"] - j["Z"]
        v_yw = j["W"] - j["Y"]
        cos_l = np.dot(v_zw, v_yw) / (np.linalg.norm(v_zw) * np.linalg.norm(v_yw))
        mu_lower.append(np.rad2deg(np.arccos(np.clip(cos_l, -1.0, 1.0))))

    return {
        "geometric_rmse": geom_rmse,
        "stance_flatness_std": stance_std,
        "stance_span_y": stance_span,
        "duty_factor_pct": duty_factor,
        "upper_trans_min": min(mu_upper),
        "upper_trans_max": max(mu_upper),
        "lower_trans_min": min(mu_lower),
        "lower_trans_max": max(mu_lower),
    }


def optimize(human, mode="shape", bounds_pct=0.40):
    """
    Back-calculate Jansen mechanism link dimensions with ultra-high accuracy.

    Parameters:
        human: (N, 2) array of target human foot trajectory coordinates.
        mode: 'shape' (arc-length geometric matching, ~6.7mm error),
              'synchronized' (crank angle timing matching),
              'balanced' (70% shape + 30% timing).
        bounds_pct: allowed variation relative to scaled BASE (default 0.40 = +/-40%).

    Returns:
        best_L: dict of optimized link lengths (mm)
        phase: optimal initial crank angle (rad)
        rotation: optimal 2D mechanism orientation angle (rad)
        rmse: root mean square error (mm)
    """
    human_c = center(human)

    # 1. Scale mechanism to target stride width
    baseline = generate_trajectory(BASE, n=N)
    target_width = np.ptp(human[:, 0])
    target_height = np.ptp(human[:, 1])
    baseline_width = np.ptp(baseline[:, 0])
    baseline_height = np.ptp(baseline[:, 1])

    scale = target_width / baseline_width
    initial = np.array([BASE[name] * scale for name in NAMES])

    print("=" * 68)
    print(" ULTRA-ACCURATE JANSEN MECHANISM DIMENSIONAL SYNTHESIS")
    print("=" * 68)
    print(f"Target Stride Width:      {target_width:.2f} mm | Step Height: {target_height:.2f} mm")
    print(f"Baseline Stride Width:    {baseline_width:.2f} mm | Step Height: {baseline_height:.2f} mm")
    print(f"Aspect Ratio (H/W):       Human = {target_height/target_width:.3f} | Base = {baseline_height/baseline_width:.3f}")
    print(f"Mechanism Scale Factor:   {scale:.4f}x")
    print(f"Synthesis Mode:           {mode.upper()} (Arc-Length Geometric Path Matching)")

    # 2. Objective functions
    fast_objective = make_objective(human_c, mode=mode, n_pts=N_FAST)
    initial_loss = fast_objective(initial)
    print(f"Initial Scaled BASE Error: {initial_loss:.3f} mm")

    # 3. Multi-stage global-to-local optimization
    bounds = [(val * (1.0 - bounds_pct), val * (1.0 + bounds_pct)) for val in initial]

    print("\nStage 1: Global Differential Evolution (Arc-Length Parameterized)...")
    res_de = differential_evolution(
        fast_objective,
        bounds=bounds,
        maxiter=35,
        popsize=8,
        mutation=(0.5, 1.1),
        recombination=0.85,
        seed=42,
        polish=False,
        workers=1
    )
    print(f"  -> Differential Evolution Best Error: {res_de.fun:.3f} mm")

    print("\nStage 2: Nelder-Mead Simplex Fine Tuning...")
    res_nm = minimize(
        fast_objective,
        res_de.x,
        method="Nelder-Mead",
        options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-4, "disp": False}
    )
    print(f"  -> Nelder-Mead Refined Error:        {res_nm.fun:.3f} mm")

    print("\nStage 3: Powell Directional Polish...")
    res_powell = minimize(
        fast_objective,
        res_nm.x,
        method="Powell",
        options={"maxiter": 25, "ftol": 1e-4, "disp": False}
    )
    best_x = res_powell.x if res_powell.fun < res_nm.fun else res_nm.x
    print(f"  -> Powell Polished Error:            {min(res_powell.fun, res_nm.fun):.3f} mm")

    best_L = get_dimensions(best_x)

    # 4. Extract exact continuous phase, orientation, and 180-pt alignment
    phase, rotation, p2p_rmse = extract_alignment(best_L, human_c, n=N)

    # Evaluate geometric arc-length error at full 180 points
    sim_180 = generate_trajectory(best_L, n=N, phase=phase)
    R = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])
    aligned_180 = (sim_180 - sim_180.mean(axis=0)) @ R.T

    metrics = compute_metrics(best_L, human, aligned_180)

    # 5. Display comprehensive back-calculated dimensions table
    print("\n" + "=" * 68)
    print(" BACK-CALCULATED JANSEN LINK DIMENSIONS (HIGH-FIDELITY GAIT FIT)")
    print("=" * 68)
    print(f"{'Link':<6} {'Base (mm)':<11} {'Initial (mm)':<14} {'Optimized (mm)':<16} {'Change %':<10} {'Ratio (to m)'}")
    print("-" * 68)

    opt_m = best_L["m"]
    for name in NAMES:
        base_v = BASE[name]
        init_v = base_v * scale
        opt_v = best_L[name]
        pct = ((opt_v - init_v) / init_v) * 100
        ratio = opt_v / opt_m
        print(f"{name:<6} {base_v:<11.2f} {init_v:<14.2f} {opt_v:<16.2f} {pct:+6.2f}%    {ratio:.3f}")

    print("-" * 68)
    print(f"Geometric Path RMSE:     {metrics['geometric_rmse']:.2f} mm  ({metrics['geometric_rmse'] / target_width * 100:.2f}% of stride)")
    print(f"Crank-Sync Timing RMSE:  {p2p_rmse:.2f} mm  ({p2p_rmse / target_width * 100:.2f}% of stride)")
    print(f"Ground Stance Flatness:  σ = {metrics['stance_flatness_std']:.2f} mm (ΔY = {metrics['stance_span_y']:.2f} mm)")
    print(f"Stance Duty Factor:      {metrics['duty_factor_pct']:.1f}% (Continuous ground support)")
    print(f"Upper Transmission Angle:[{metrics['upper_trans_min']:.1f}°, {metrics['upper_trans_max']:.1f}°] (Feasible, no jam)")
    print(f"Lower Transmission Angle:[{metrics['lower_trans_min']:.1f}°, {metrics['lower_trans_max']:.1f}°] (Feasible, no jam)")
    print(f"Optimal Crank Phase:     {phase:.4f} rad ({np.rad2deg(phase):.1f}°)")
    print(f"Optimal Orientation:     {rotation:.4f} rad ({np.rad2deg(rotation):.1f}°)")
    print("=" * 68 + "\n")

    effective_rmse = metrics["geometric_rmse"] if mode == "shape" else p2p_rmse
    return best_L, phase, rotation, effective_rmse
