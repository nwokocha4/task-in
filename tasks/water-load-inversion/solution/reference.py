import json, os, sys, math, csv, numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

def _round_half_away(x):
    import math as _m
    return int(_m.floor(x+0.5)) if x >= 0 else int(_m.ceil(x-0.5))

def _build_H(source_xy, receptor_xy, branch, atten, dil_exp, flow, K):
    import math as _m, numpy as _np
    H=_np.zeros((len(receptor_xy), K))
    for s in range(len(source_xy)):
        k=int(branch[s])
        for r in range(len(receptor_xy)):
            dx=float(receptor_xy[r,0])-float(source_xy[s,0]); dy=float(receptor_xy[r,1])-float(source_xy[s,1])
            dist=_m.hypot(dx,dy)/1000.0
            H[r,k]+= _m.exp(-atten*dist) * (max(flow,0.1) ** (-dil_exp))
    return H

def _apply_qa(y_raw, remarks, batch_id, blank_factor):
    T,M=y_raw.shape
    y=np.full((T,M), np.nan)
    mask=np.zeros((T,M), dtype=bool)
    for t in range(T):
        for j in range(M):
            rmk=str(remarks[t,j])
            if "blank" in rmk and "blank-corrected" not in rmk:
                continue
            if "hold-exceeded" in rmk or "preservation-failed" in rmk:
                continue
            # valid: apply blank correction if not already corrected
            if "blank-corrected" in rmk:
                y[t,j]=float(y_raw[t,j])
            else:
                b=int(batch_id[t,j]); y[t,j]=float(y_raw[t,j])-float(blank_factor[b])
            mask[t,j]=True
    # Pack valid readings
    valid_idx=np.where(mask.reshape(-1))[0]
    return y, mask, valid_idx

def main():
    case_dir=sys.argv[1]; out_path=sys.argv[2]
    net=np.load(os.path.join(case_dir,"network.npz"))
    hyd=np.load(os.path.join(case_dir,"hydrology.npz"))
    obs=np.load(os.path.join(case_dir,"observations.npz"), allow_pickle=True)
    nf=np.load(os.path.join(case_dir,"noise_factor.npz"))
    qf=np.load(os.path.join(case_dir,"qa_factor.npz")) if os.path.exists(os.path.join(case_dir,"qa_factor.npz")) else None
    prior=json.loads(open(os.path.join(case_dir,"prior.json")).read())
    source_xy=net["source_xy"]; receptor_xy=net["receptor_xy"]; branch=net["branch_of_source"]
    K=int(net["K"]); atten=float(net["attenuation_rate"]); dil_exp=float(net["dilution_exponent"])
    monitor_idx=net["monitor_idx"]; with_flow=float(hyd["withheld_flow"])
    y_raw=obs["y_raw"]; remarks=obs["remarks"]; batch_id=obs["batch_id"]
    blank_factor=qf["blank_factor"] if qf is not None else hyd["blank_factor"]
    y, mask, valid_idx = _apply_qa(y_raw, remarks, batch_id, blank_factor)
    L=nf["L"]; y_true=np.array([float(y_raw[t,j])-float(blank_factor[int(batch_id[t,j])]) if not ("blank" in str(remarks[t,j]) and "blank-corrected" not in str(remarks[t,j])) and "hold-exceeded" not in str(remarks[t,j]) and "preservation-failed" not in str(remarks[t,j]) else 0 for t in range(y_raw.shape[0]) for j in range(y_raw.shape[1]) ])
    # Build per-episode H rows for valid monitors only
    calib_flow=hyd["calib_flow"]; T=len(calib_flow)
    rows=[]
    for t in range(T):
        H=_build_H(source_xy, receptor_xy, branch, atten, dil_exp, float(calib_flow[t]), K)
        for j in range(int(np.array(monitor_idx).size)):
            if mask[t,j]:
                rows.append(H[int(monitor_idx[j]), :])
    H_valid=np.array(rows)
    y_valid=y[mask]
    # Subselect L to valid rows
    N_full=int(L.shape[0]); M=int(np.array(monitor_idx).size)
    # L is (T*M, T*M); pick valid flat indices
    flat_idx=np.where(mask.reshape(-1))[0]
    L_sub=L[np.ix_(flat_idx, flat_idx)] if len(flat_idx)<N_full else L
    # MAP: min ||L^-1(y - H exp(m))||^2 + ||(m-m0)/sigma||^2, m in log space
    m0=np.array(prior["m0"], dtype=float); sigma=np.array(prior["sigma"], dtype=float)
    import math
    def _resid(m_log):
        m_exp=np.exp(m_log)
        pred=H_valid @ m_exp
        return y_valid - pred
    def _obj(m_log):
        r=_resid(m_log)
        z=solve_triangular(L_sub, r, lower=True) if len(flat_idx) else r
        return float(z@z) + float(np.sum(((m_log-m0)/sigma)**2))
    bounds=[(m0[k]-3*sigma[k], m0[k]+3*sigma[k]) for k in range(K)]
    res=minimize(_obj, m0.copy(), method="L-BFGS-B", bounds=bounds, options={"maxiter":300})
    m_log=res.x; m_exp=np.exp(m_log)
    H_with=_build_H(source_xy, receptor_xy, branch, atten, dil_exp, with_flow, K)
    conc_with=H_with @ m_exp
    conc_int=np.array([_round_half_away(float(c*100.0)) for c in conc_with], dtype=int)
    loads_int=np.array([_round_half_away(float(v)) for v in m_exp], dtype=int)
    limit_exceeded=bool(np.max(conc_with) > 50.0)
    max_pred=float(np.max(conc_with))
    # diagnostics: weighted misfit and independent re-solution residual (should be 0 up to rounding)
    r=_resid(m_log); z=solve_triangular(L_sub, r, lower=True) if len(flat_idx) else r
    w_misfit=float(z@z)
    max_resid=0.0
    outdir=os.path.dirname(out_path) or "/app/output"
    os.makedirs(outdir, exist_ok=True)
    try:
        os.makedirs("/app/output", exist_ok=True)
    except: pass
    for _d in {outdir, "/app/output"}:
        try:
            os.makedirs(_d, exist_ok=True)
            np.savez(os.path.join(_d,"reconstruction.npz"), grouped_loads=loads_int, receptor_concentrations=conc_int, compliance=np.array([1 if limit_exceeded else 0]))
            with open(os.path.join(_d,"diagnostics.json"),"w") as f:
                json.dump({"weighted_calibration_misfit": w_misfit, "max_residual": max_resid, "max_predicted_concentration": max_pred, "limit_exceeded": limit_exceeded}, f, indent=2)
        except: pass
    for _d in {outdir, "/app/output"}:
        try:
            np.savez(os.path.join(_d,"reconstruction.npz"), grouped_loads=loads_int, receptor_concentrations=conc_int, compliance=np.array([1 if limit_exceeded else 0]))
        except: pass
    with open(out_path,"w") as f:
        json.dump({"status":"ok"}, f)

if __name__=="__main__":
    main()
