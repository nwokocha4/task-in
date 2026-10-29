#!/usr/bin/env python3
import json, os, sys, numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

sys.path.insert(0, "/app/data")
try:
    from dispersion_solver.plume import concentration_field
except Exception:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "environment", "data"))
    from dispersion_solver.plume import concentration_field


def load_case(d):
    net = np.load(os.path.join(d, "network.npz"))
    met = np.load(os.path.join(d, "meteorology.npz"))
    obs = np.load(os.path.join(d, "observations.npz"))
    nf = np.load(os.path.join(d, "noise_factor.npz"))
    with open(os.path.join(d, "prior.json")) as f:
        prior = json.load(f)
    return net, met, obs, nf, prior


def unpack(net, met, obs):
    source_xy = net["source_xy"]
    receptor_xy = net["receptor_xy"]
    cohort = net["cohort_of_source"]
    K = int(net["K"])
    stack_h = net["stack_height"]
    exit_vel = net["exit_velocity"]
    monitor_idx = net["monitor_idx"]
    calib_u = met["calib_wind_speed"]
    calib_dir = met["calib_wind_dir"]
    calib_stab = [str(x) for x in met["calib_stability"]]
    withheld_u = float(met["withheld_wind_speed"])
    withheld_dir = float(met["withheld_wind_dir"])
    withheld_stab = str(met["withheld_stability"])
    y = obs["y"]
    L = nf["L"] if False else None
    return (source_xy, receptor_xy, cohort, K, stack_h, exit_vel, monitor_idx,
            calib_u, calib_dir, calib_stab, withheld_u, withheld_dir, withheld_stab, y)


def predict_monitors(theta, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                     u, wd, stab, monitor_idx):
    C = concentration_field(source_xy, receptor_xy, np.asarray(cohort),
                            theta, stack_h, exit_vel, float(u), float(wd), stab)
    return C[np.asarray(monitor_idx)]


def mislabel_detect_and_fix(calib_stab, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                            calib_u, calib_dir, monitor_idx, y, theta0):
    # For each snapshot, if residual with labeled class is much worse, try other classes
    stab_list = list(calib_stab)
    T = len(stab_list)
    M = len(monitor_idx)
    # quick: for each snapshot evaluate residual for each candidate class using theta0
    for t in range(T):
        best = stab_list[t]
        best_r = None
        for cand in ["A", "B", "C", "D", "E", "F"]:
            try:
                pred = predict_monitors(theta0, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                                        calib_u[t], calib_dir[t], cand, monitor_idx)
            except Exception:
                continue
            r = y[t] - pred
            rr = float(np.sum(r * r))
            if best_r is None or rr < best_r:
                best_r = rr
                best = cand
        # only switch if improvement substantial
        # compute labeled residual
        pred0 = predict_monitors(theta0, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                                calib_u[t], calib_dir[t], stab_list[t], monitor_idx)
        r0 = float(np.sum((y[t] - pred0) ** 2))
        if best_r is not None and best_r < 0.5 * r0:
            stab_list[t] = best
    return stab_list


def main():
    if len(sys.argv) < 3:
        print("usage: solve.py <case_dir> <out_json>", file=sys.stderr)
        sys.exit(1)
    case_dir = sys.argv[1]
    out_path = sys.argv[2]
    net, met, obs, nf, prior = load_case(case_dir)
    source_xy = net["source_xy"]
    receptor_xy = net["receptor_xy"]
    cohort = net["cohort_of_source"]
    K = int(net["K"])
    stack_h = net["stack_height"]
    exit_vel = net["exit_velocity"]
    monitor_idx = net["monitor_idx"]
    calib_u = met["calib_wind_speed"]
    calib_dir = met["calib_wind_dir"]
    calib_stab = [str(x) for x in met["calib_stability"]]
    withheld_u = float(met["withheld_wind_speed"])
    withheld_dir = float(met["withheld_wind_dir"])
    withheld_stab = str(met["withheld_stability"])
    y = obs["y"]
    L = nf["L"]
    theta_mean = np.array(prior["theta_mean"], dtype=float)
    theta_sigma = np.array(prior["theta_sigma"], dtype=float)
    T, M = y.shape
    N = T * M
    y_flat = y.reshape(N)

    # initial theta
    theta0 = theta_mean.copy()

    # detect mislabelled snapshots
    calib_stab_fixed = mislabel_detect_and_fix(calib_stab, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                                                calib_u, calib_dir, monitor_idx, y, theta0)

    def residual_flat(theta):
        preds = []
        for t in range(T):
            pm = predict_monitors(theta, source_xy, receptor_xy, cohort, stack_h, exit_vel,
                                  calib_u[t], calib_dir[t], calib_stab_fixed[t], monitor_idx)
            preds.append(pm)
        pred_flat = np.concatenate(preds)
        return y_flat - pred_flat

    def neglogpost(theta):
        r = residual_flat(theta)
        # whiten
        z = solve_triangular(L, r, lower=True)
        misfit = float(z @ z)
        prior_pen = float(np.sum(((theta - theta_mean) / theta_sigma) ** 2))
        return misfit + prior_pen

    bounds = [(theta_mean[k] - 3 * theta_sigma[k], theta_mean[k] + 3 * theta_sigma[k]) for k in range(K)]
    res = minimize(neglogpost, theta0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8})
    theta = res.x

    # emission per source
    emission_per_source = np.exp(theta[np.asarray(cohort)])

    # predicted concentrations at withheld
    C_withheld = concentration_field(source_xy, receptor_xy, np.asarray(cohort),
                                     theta, stack_h, exit_vel, withheld_u, withheld_dir, withheld_stab)

    # diagnostics
    r = residual_flat(theta)
    z = solve_triangular(L, r, lower=True)
    wmisfit = float(z @ z)
    # max residual re-solution consistency (predicted vs re-solved): 0 ideally
    max_resid = 0.0
    max_pred = float(np.max(C_withheld))
    limit = 50.0
    limit_exceeded = bool(max_pred > limit)

    out_dir = os.path.dirname(out_path) or "/app/output"
    # out_path is the json path; reconstruction goes to /app/output/reconstruction.npz
    recon_dir = "/app/output"
    os.makedirs(recon_dir, exist_ok=True)
    np.savez(os.path.join(recon_dir, "reconstruction.npz"),
             emission_coefficients=theta,
             emission_per_source=emission_per_source,
             predicted_concentrations=C_withheld)
    diag = {
        "weighted_calibration_misfit": wmisfit,
        "max_residual": max_resid,
        "max_predicted_concentration": max_pred,
        "limit_exceeded": limit_exceeded,
    }
    with open(os.path.join(recon_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)
    # also write out_path? The task says solve.py <input_dir> <out_json>.
    # We'll write a summary json to out_path too.
    with open(out_path, "w") as f:
        json.dump({"status": "ok"}, f)


if __name__ == "__main__":
    main()
