"""Deterministic generator for grouped-emission-source-inversion task."""
import json
import os
import numpy as np
import random

SEEDS = {
    "example": [101, 102, 103],
    "hidden": [201, 202, 203, 204, 205, 206],
}

STABILITY_LABELS = ["A", "B", "C", "D", "E", "F"]


def make_case(rng, seed):
    # Plant geometry
    S = rng.randint(4, 7)
    K = 6
    # cohorts: assign sources to cohorts (some cohorts may have multiple sources)
    cohort_of_source = [rng.randrange(K) for _ in range(S)]
    # ensure all cohorts used? not necessary
    R = rng.randint(12, 20)
    # source coords in a grid-ish region
    source_xy = np.array([[rng.uniform(0, 500), rng.uniform(0, 500)] for _ in range(S)])
    receptor_xy = np.array([[rng.uniform(-200, 800), rng.uniform(-200, 800)] for _ in range(R)])
    stack_height = np.array([rng.uniform(0, 2) for _ in range(S)])
    exit_velocity = np.array([rng.uniform(5, 20) for _ in range(S)])

    # true theta (log emissions) per cohort
    theta_true = np.array([rng.uniform(15.0, 17.0) for _ in range(K)])
    # prior
    prior_mean = theta_true + np.array([rng.uniform(-0.2, 0.2) for _ in range(K)])
    prior_sigma = np.array([rng.uniform(0.3, 0.6) for _ in range(K)])

    # monitors: subset of receptors
    M = rng.randint(5, 9)
    monitor_idx = sorted(rng.sample(range(R), M))

    # calibration meteorologies
    T = rng.randint(4, 6)
    # stability classes for calibration (exclude one class to be withheld)
    calib_stabilities = [rng.choice(["A", "B", "C", "D"]) for _ in range(T)]
    # mislabel one snapshot
    mislabel_idx = rng.randrange(T)
    calib_stability_labels = list(calib_stabilities)
    # store actual
    calib_stability_actual = list(calib_stabilities)
    # apply mislabel: change label at mislabel_idx to a different class
    wrong = rng.choice([c for c in STABILITY_LABELS if c != calib_stabilities[mislabel_idx]])
    calib_stability_labels[mislabel_idx] = wrong

    wind_speeds = [rng.uniform(2, 8) for _ in range(T)]
    wind_dirs = [rng.uniform(0, 360) for _ in range(T)]

    # withheld meteorology: stability class not in calibration, different wind regime
    withheld_stability = rng.choice(["E", "F"])
    withheld_u = rng.uniform(0.5, 1.5)  # low wind
    withheld_dir = rng.uniform(0, 360)

    # background offsets per snapshot (large, near-rank-one)
    bg_offsets = [rng.uniform(-5, 5) for _ in range(T)]

    # noise covariance L (T*M, T*M): base iid + rank-one background direction
    N = T * M
    L = np.eye(N) * rng.uniform(0.5, 1.5)
    # rank-one background component (per-snapshot block)
    v = np.zeros(N)
    for t in range(T):
        v[t * M:(t + 1) * M] = 1.0 / np.sqrt(M)
    L = L + np.outer(v, v) * rng.uniform(3, 6)
    # cholesky-ish: ensure lower triangular positive
    A = L @ L.T
    L = np.linalg.cholesky(A)

    # forward model import
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from _model import concentration_field

    # compute predicted monitor concentrations per snapshot
    pred_flat = np.zeros(N)
    for t in range(T):
        stab = calib_stability_actual[t]
        u = wind_speeds[t]
        wd = wind_dirs[t]
        C = concentration_field(source_xy, receptor_xy, np.array(cohort_of_source),
                                theta_true, stack_height, exit_velocity, u, wd, stab)
        mon_vals = C[monitor_idx]
        pred_flat[t * M:(t + 1) * M] = mon_vals + bg_offsets[t]

    # observations = pred + noise from L
    noise = L @ np.random.standard_normal(N)
    y_flat = pred_flat + noise
    y = y_flat.reshape(T, M)

    # withheld concentrations (true)
    C_withheld = concentration_field(source_xy, receptor_xy, np.array(cohort_of_source),
                                    theta_true, stack_height, exit_velocity,
                                    withheld_u, withheld_dir, withheld_stability)
    # withheld monitor values (for verifier) = C_withheld[monitor_idx] + small noise?
    # We'll store true concentrations at all receptors and monitor subset.
    withheld_monitors = C_withheld[monitor_idx]

    # limit
    limit = 50.0

    case = {
        "network": {
            "source_xy": source_xy,
            "receptor_xy": receptor_xy,
            "cohort_of_source": np.array(cohort_of_source),
            "K": K,
            "stack_height": stack_height,
            "exit_velocity": exit_velocity,
            "monitor_idx": np.array(monitor_idx),
        },
        "meteorology": {
            "calib_wind_speed": np.array(wind_speeds),
            "calib_wind_dir": np.array(wind_dirs),
            "calib_stability": np.array(calib_stability_labels),  # (possibly mislabelled)
            "calib_stability_actual": np.array(calib_stability_actual),
            "mislabel_idx": mislabel_idx,
            "withheld_wind_speed": withheld_u,
            "withheld_wind_dir": withheld_dir,
            "withheld_stability": withheld_stability,
            "bg_offsets": np.array(bg_offsets),
        },
        "observations": {
            "y": y,
            "monitor_idx": np.array(monitor_idx),
        },
        "noise_factor": {
            "L": L,
        },
        "prior": {
            "theta_mean": prior_mean.tolist(),
            "theta_sigma": prior_sigma.tolist(),
            "admissible_sigma": 3.0,
        },
        "limit": limit,
        "truth": {
            "theta": theta_true.tolist(),
            "predicted_concentrations": C_withheld.tolist(),
            "withheld_monitors": withheld_monitors.tolist(),
        },
    }
    return case, seed


def save_case(case, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "network.npz"),
             source_xy=case["network"]["source_xy"],
             receptor_xy=case["network"]["receptor_xy"],
             cohort_of_source=case["network"]["cohort_of_source"],
             K=case["network"]["K"],
             stack_height=case["network"]["stack_height"],
             exit_velocity=case["network"]["exit_velocity"],
             monitor_idx=case["network"]["monitor_idx"])
    np.savez(os.path.join(out_dir, "meteorology.npz"),
             calib_wind_speed=case["meteorology"]["calib_wind_speed"],
             calib_wind_dir=case["meteorology"]["calib_wind_dir"],
             calib_stability=case["meteorology"]["calib_stability"],
             withheld_wind_speed=case["meteorology"]["withheld_wind_speed"],
             withheld_wind_dir=case["meteorology"]["withheld_wind_dir"],
             withheld_stability=case["meteorology"]["withheld_stability"])
    np.savez(os.path.join(out_dir, "observations.npz"),
             y=case["observations"]["y"],
             monitor_idx=case["observations"]["monitor_idx"])
    np.savez(os.path.join(out_dir, "noise_factor.npz"),
             L=case["noise_factor"]["L"])
    with open(os.path.join(out_dir, "prior.json"), "w") as f:
        json.dump(case["prior"], f, indent=2)


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # task root
    # examples
    for i, seed in enumerate(SEEDS["example"]):
        rng = random.Random(seed)
        np.random.seed(seed)
        case, _ = make_case(rng, seed)
        case_dir = os.path.join(base, "environment", "data", "examples", f"case_{i+1}")
        save_case(case, case_dir)
        # expected.json
        with open(os.path.join(case_dir, "expected.json"), "w") as f:
            json.dump(case["truth"], f, indent=2)
    # hidden
    for i, seed in enumerate(SEEDS["hidden"]):
        rng = random.Random(seed)
        np.random.seed(seed)
        case, _ = make_case(rng, seed)
        case_dir = os.path.join(base, "tests", "hidden", f"case_{i+4}")
        save_case(case, case_dir)
        # hidden expected
        exp_dir = os.path.join(base, "tests", "hidden_expected")
        os.makedirs(exp_dir, exist_ok=True)
        with open(os.path.join(exp_dir, f"case_{i+4}.json"), "w") as f:
            json.dump(case["truth"], f, indent=2)
        # also save limit etc
        with open(os.path.join(exp_dir, f"case_{i+4}_meta.json"), "w") as f:
            json.dump({"limit": case["limit"], "prior": case["prior"]}, f, indent=2)


if __name__ == "__main__":
    main()
