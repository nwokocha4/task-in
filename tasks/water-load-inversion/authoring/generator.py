"""Deterministic generator for water-load-inversion."""
import json, os, math, random
import numpy as np
from _model import build_H

SEEDS = {"example": [101,102,103], "hidden": [201,202,203,204,205,206]}
REMARKS_CENSORED = ["hold-exceeded", "preservation-failed"]

def _round_half_away(x):
    import math as _m
    return int(_m.floor(x+0.5)) if x >= 0 else int(_m.ceil(x-0.5))

def make_case(rng, seed):
    S = rng.randint(4, 7); K = 3 + (rng.randint(0,3))
    branch = [rng.randrange(K) for _ in range(S)]
    R = rng.randint(10, 18); M = rng.randint(5, 9)
    source_xy = np.array([[rng.uniform(0,500), rng.uniform(0,500)] for _ in range(S)], dtype=float)
    receptor_xy = np.array([[rng.uniform(-200,800), rng.uniform(-200,800)] for _ in range(R)], dtype=float)
    monitor_idx = sorted(rng.sample(range(R), M))
    atten = rng.uniform(0.1, 0.6); dil_exp = rng.uniform(0.4, 0.9)
    m_true = np.array([rng.uniform(13.0, 15.5) for _ in range(K)])
    m0 = np.array([rng.uniform(12.0, 14.0) for _ in range(K)])
    sigma = np.array([rng.uniform(0.3, 0.6) for _ in range(K)])
    T = rng.randint(5, 8)
    calib_flow = [rng.uniform(2, 8) for _ in range(T)]
    calib_regime = [rng.choice(["low","mid"]) for _ in range(T)]
    withheld_flow = rng.uniform(0.5, 1.2); withheld_regime = "high"
    blank_factor = np.array([rng.uniform(0.5, 2.0) for _ in range(T)])
    N = T*M
    L = np.eye(N) * rng.uniform(0.4, 1.0)
    v = np.zeros(N)
    for t in range(T):
        v[t*M:(t+1)*M] = 1.0 / (M**0.5)
    L = L + np.outer(v, v) * rng.uniform(2, 5)
    L = np.linalg.cholesky(L @ L.T)
    y_raw = np.zeros((T, M))
    remarks = [["" for _ in range(M)] for _ in range(T)]
    batch_id = np.zeros((T, M), dtype=int)
    # Build observations with QA wording that exercises the policy
    for t in range(T):
        for j in range(M):
            if t == 0 and j == 0:
                remarks[t][j] = "batch blank"
            elif t == 0 and j == 1:
                remarks[t][j] = "blank-corrected"
            elif t == 1 and j == 0:
                remarks[t][j] = rng.choice(REMARKS_CENSORED)
            else:
                remarks[t][j] = rng.choice(["dilution: 1:10","hold-flag","batch note"])
            batch_id[t,j] = t
        H = build_H(source_xy, receptor_xy, np.array(branch), atten, dil_exp, calib_flow[t], K=K)
        m_exp = np.exp(m_true)
        conc = H @ m_exp
        pred = conc[monitor_idx]
        y_raw[t,:] = pred + blank_factor[t]
    y_flat = y_raw.reshape(-1) + L @ np.random.standard_normal(N)
    y_raw = y_flat.reshape(T, M)
    # Self-consistent truth: define the sealed expected as the MAP fixed-point for this y,
    # so the reference solver lands on the same integers (deterministic grading).
    # This preserves the intended difficulty (whitening/blank/QA vs. not) while making the
    # exact-integer grading sound: the identifiable subspace + prior + convergence criterion pin the digit.
    def _apply_qa_local(y_raw_l, remarks_l, batch_id_l, blank_factor_l):
        import numpy as _np
        T_l, M_l = y_raw_l.shape
        y_l = _np.full((T_l, M_l), _np.nan)
        mask_l = _np.zeros((T_l, M_l), dtype=bool)
        for tt in range(T_l):
            for jj in range(M_l):
                rmk = str(remarks_l[tt,jj])
                if "blank" in rmk and "blank-corrected" not in rmk:
                    continue
                if "hold-exceeded" in rmk or "preservation-failed" in rmk:
                    continue
                if "blank-corrected" in rmk:
                    y_l[tt,jj] = float(y_raw_l[tt,jj])
                else:
                    y_l[tt,jj] = float(y_raw_l[tt,jj]) - float(blank_factor_l[int(batch_id_l[tt,jj])])
                mask_l[tt,jj] = True
        flat = _np.where(mask_l.reshape(-1))[0]
        return y_l, mask_l, flat

    try:
        from scipy.optimize import minimize as _min
        from scipy.linalg import solve_triangular as _tri
        _y, _mask, _flat = _apply_qa_local(y_raw, remarks, batch_id, blank_factor)
        _rows = []
        for tt in range(len(calib_flow)):
            _H = build_H(source_xy, receptor_xy, np.array(branch), atten, dil_exp, float(calib_flow[tt]), K=K)
            for jj in range(int(np.array(monitor_idx).size)):
                if _mask[tt, jj]:
                    _rows.append(_H[int(monitor_idx[jj]), :])
        _Hvalid = np.array(_rows) if _rows else np.zeros((0, K))
        _yvalid = _y[_mask]
        _Nfull = int(L.shape[0]); _Lsub = L[np.ix_(_flat, _flat)] if len(_flat) < _Nfull else L
        _m0 = np.array(m0, dtype=float); _sig = np.array(sigma, dtype=float)
        def _res(m_log):
            return _yvalid - _Hvalid @ np.exp(m_log)
        def _obj(m_log):
            _r = _res(m_log)
            _z = _tri(_Lsub, _r, lower=True) if len(_flat) else _r
            return float(_z @ _z) + float(np.sum(( (m_log-_m0)/_sig)**2))
        _bounds = [(_m0[k]-3*_sig[k], _m0[k]+3*_sig[k]) for k in range(K)]
        _res_opt = _min(_obj, _m0.copy(), method="L-BFGS-B", bounds=_bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10})
        _m_star = _res_opt.x if _res_opt.success else m_true
        # Withheld prediction from the MAP loads is the graded quantity
        _H_with = build_H(source_xy, receptor_xy, np.array(branch), atten, dil_exp, withheld_flow, K=K)
        _conc_with = _H_with @ np.exp(_m_star)
        loads_int = [_round_half_away(float(np.exp(_m_star[k]))) for k in range(K)]
        conc_int = [_round_half_away(float(_conc_with[r]*100.0)) for r in range(len(_conc_with))]
        limit_exceeded = bool(np.max(_conc_with) > 50.0)
    except Exception as _e:
        # Fallback to m_true if optimizer unavailable
        m_exp_true = np.exp(m_true)
        H_with = build_H(source_xy, receptor_xy, np.array(branch), atten, dil_exp, withheld_flow, K=K)
        conc_with = H_with @ np.exp(m_true)
        loads_int = [_round_half_away(float(np.exp(m_true[k]))) for k in range(K)]
        conc_int = [_round_half_away(float(conc_with[r]*100.0)) for r in range(R)]
        limit_exceeded = bool(np.max(conc_with) > 50.0)
    case = dict(network=dict(source_xy=source_xy, receptor_xy=receptor_xy, branch_of_source=np.array(branch), K=K, attenuation_rate=atten, dilution_exponent=dil_exp, monitor_idx=np.array(monitor_idx)), hydrology=dict(calib_flow=np.array(calib_flow), calib_regime=np.array(calib_regime), withheld_flow=withheld_flow, withheld_regime=withheld_regime, blank_factor=blank_factor), obs=dict(y_raw=y_raw, remarks=np.array(remarks, dtype=object), batch_id=batch_id), L=L, prior=dict(m0=m0.tolist(), sigma=sigma.tolist()), truth=dict(loads=loads_int, conc=conc_int, limit_exceeded=limit_exceeded))
    return case, seed

def _save_case(case, out_dir):
    import os as _os
    _os.makedirs(out_dir, exist_ok=True)
    np.savez(_os.path.join(out_dir, "network.npz"), source_xy=case["network"]["source_xy"], receptor_xy=case["network"]["receptor_xy"], branch_of_source=case["network"]["branch_of_source"], K=case["network"]["K"], attenuation_rate=case["network"]["attenuation_rate"], dilution_exponent=case["network"]["dilution_exponent"], monitor_idx=case["network"]["monitor_idx"])
    np.savez(_os.path.join(out_dir, "hydrology.npz"), calib_flow=case["hydrology"]["calib_flow"], calib_regime=case["hydrology"]["calib_regime"], withheld_flow=case["hydrology"]["withheld_flow"], withheld_regime=case["hydrology"]["withheld_regime"], blank_factor=case["hydrology"]["blank_factor"])
    remarks_arr = np.array(case["obs"]["remarks"], dtype="<U40")
    np.savez(_os.path.join(out_dir, "observations.npz"), y_raw=case["obs"]["y_raw"], remarks=remarks_arr, batch_id=case["obs"]["batch_id"])
    np.savez(_os.path.join(out_dir, "noise_factor.npz"), L=case["L"])
    np.savez(_os.path.join(out_dir, "qa_factor.npz"), blank_factor=case["hydrology"]["blank_factor"])
    import json as _js
    with open(_os.path.join(out_dir, "prior.json"), "w") as f:
        _js.dump(case["prior"], f, indent=2)
    with open(_os.path.join(out_dir, "expected.json"), "w") as f:
        _js.dump({"grouped_loads": case["truth"]["loads"], "receptor_concentrations": case["truth"]["conc"], "compliance": {"limit_exceeded": case["truth"]["limit_exceeded"]}}, f, indent=2)

def main():
    import os as _os
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for i, seed in enumerate(SEEDS["example"]):
        rng = random.Random(seed); np.random.seed(seed)
        case,_ = make_case(rng, seed)
        _save_case(case, _os.path.join(base, "environment","data","examples", f"case_{i+1}"))
    for i, seed in enumerate(SEEDS["hidden"]):
        rng = random.Random(seed); np.random.seed(seed)
        case,_ = make_case(rng, seed)
        hd = _os.path.join(base, "tests","hidden", f"case_{i+4}")
        _os.makedirs(hd, exist_ok=True)
        np.savez(_os.path.join(hd, "network.npz"), source_xy=case["network"]["source_xy"], receptor_xy=case["network"]["receptor_xy"], branch_of_source=case["network"]["branch_of_source"], K=case["network"]["K"], attenuation_rate=case["network"]["attenuation_rate"], dilution_exponent=case["network"]["dilution_exponent"], monitor_idx=case["network"]["monitor_idx"])
        np.savez(_os.path.join(hd, "hydrology.npz"), calib_flow=case["hydrology"]["calib_flow"], calib_regime=case["hydrology"]["calib_regime"], withheld_flow=case["hydrology"]["withheld_flow"], withheld_regime=case["hydrology"]["withheld_regime"], blank_factor=case["hydrology"]["blank_factor"])
        remarks_arr = np.array(case["obs"]["remarks"], dtype="<U40")
        np.savez(_os.path.join(hd, "observations.npz"), y_raw=case["obs"]["y_raw"], remarks=remarks_arr, batch_id=case["obs"]["batch_id"])
        np.savez(_os.path.join(hd, "noise_factor.npz"), L=case["L"])
        np.savez(_os.path.join(hd, "qa_factor.npz"), blank_factor=case["hydrology"]["blank_factor"])
        with open(_os.path.join(hd, "prior.json"), "w") as f:
            _js.dump(case["prior"], f, indent=2) if ( _js:=__import__("json")) else None
        expd = _os.path.join(base, "tests","hidden_expected")
        _os.makedirs(expd, exist_ok=True)
        with open(_os.path.join(expd, f"case_{i+4}.json"), "w") as f:
            import json as _js2; _js2.dump({"grouped_loads": case["truth"]["loads"], "receptor_concentrations": case["truth"]["conc"], "compliance": {"limit_exceeded": case["truth"]["limit_exceeded"]}}, f, indent=2)
        with open(_os.path.join(expd, f"case_{i+4}_meta.json"), "w") as f:
            import json as _js3; _js3.dump({"prior": case["prior"], "attenuation_rate": case["network"]["attenuation_rate"], "dilution_exponent": case["network"]["dilution_exponent"]}, f, indent=2)

if __name__ == "__main__":
    main()
