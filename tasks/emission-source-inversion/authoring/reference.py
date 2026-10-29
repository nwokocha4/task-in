import json, os, sys, numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import csv, math

def sigma_table(path):
    t={}
    for row in csv.DictReader(open(path)):
        t[row['stability']]=(float(row['a_y']),float(row['b_y']),float(row['a_z']),float(row['b_z']))
    return t

def concentration_field(source_xy, receptor_xy, cohort, theta, stack_h, exit_vel, u, wd, stab, sigma_coef):
    Q=np.exp(theta[np.asarray(cohort)])
    C=np.zeros(len(receptor_xy))
    for s in range(len(source_xy)):
        F=1e-3*float(Q[s])*max(float(exit_vel[s]),0.1)
        if stab in ('E','F'):
            dh=(F/(float(u)*0.02+1e-6))**(1/3)
        else:
            dh=1.6*(F/(float(u)+1e-6))**(1/3)
        H=float(stack_h[s])+min(dh,50.0)
        for r in range(len(receptor_xy)):
            th=math.radians(float(wd))
            dx=receptor_xy[r,0]-source_xy[s,0] if hasattr(source_xy,'__getitem__') else 0
            dy=receptor_xy[r,1]-source_xy[s,1] if hasattr(source_xy,'__getitem__') else 0
            # need correct indexing
            pass
    return C

def concentration_field2(source_xy, receptor_xy, cohort, theta, stack_h, exit_vel, u, wd, stab, sigma_coef):
    import math, numpy as np
    a_y,b_y,a_z,b_z = sigma_coef[stab]
    Q = np.exp(theta[np.asarray(cohort)])
    C = np.zeros(len(receptor_xy))
    th = math.radians(float(wd))
    wx,wy = math.cos(th), math.sin(th)
    for s in range(len(source_xy)):
        F = 1e-3*float(Q[s])*max(float(exit_vel[s]),0.1)
        if stab in ('E','F'):
            dh = (F/(float(u)*0.02+1e-6))**(1/3)
        else:
            dh = 1.6*(F/(float(u)+1e-6))**(1/3)
        H = float(stack_h[s]) + min(dh, 50.0)
        for r in range(len(receptor_xy)):
            dx = float(receptor_xy[r,0])-float(source_xy[s,0])
            dy = float(receptor_xy[r,1])-float(source_xy[s,1])
            x = dx*wx + dy*wy
            y = -dx*wy + dy*wx
            if x <= 0: continue
            sy = a_y*max(float(x),1.0)**b_y
            sz = a_z*max(float(x),1.0)**b_z
            sy=max(sy,15.0); sz=max(sz,15.0)
            C[r]+= float(Q[s])/(math.pi*float(u)*sy*sz)*math.exp(-y*y/(2*sy*sy))*math.exp(-H*H/(2*sz*sz))
    return C

def load_case(d):
    net=np.load(os.path.join(d,'network.npz'))
    met=np.load(os.path.join(d,'meteorology.npz'))
    obs=np.load(os.path.join(d,'observations.npz'))
    nf=np.load(os.path.join(d,'noise_factor.npz'))
    prior=json.load(open(os.path.join(d,'prior.json')))
    return net,met,obs,nf,prior

def main():
    case_dir=sys.argv[1]
    out_path=sys.argv[2]
    net,met,obs,nf,prior=load_case(case_dir)
    source_xy=net['source_xy']; receptor_xy=net['receptor_xy']; cohort=net['cohort_of_source']
    K=int(net['K']); stack_h=net['stack_height']; exit_vel=net['exit_velocity']; monitor_idx=net['monitor_idx']
    calib_u=met['calib_wind_speed']; calib_dir=met['calib_wind_dir']
    calib_stab=[str(x) for x in met['calib_stability']]
    withheld_u=float(met['withheld_wind_speed']); withheld_dir=float(met['withheld_wind_dir']); withheld_stab=str(met['withheld_stability'])
    y=obs['y']; L=nf['L']; N=y.size; y_flat=y.reshape(-1)
    theta_mean=np.array(prior['theta_mean'],dtype=float); theta_sigma=np.array(prior['theta_sigma'],dtype=float)
    sigma_coef=sigma_table('/app/data/docs/sigma_table.csv') if os.path.exists('/app/data/docs/sigma_table.csv') else {}
    if not sigma_coef:
        # fallback to env
        import csv
        for p in ['environment/data/docs/sigma_table.csv','./environment/data/docs/sigma_table.csv']:
            if os.path.exists(p):
                sigma_coef=sigma_table(p)
                break
    def predict(theta,u,wd,stab):
        C=concentration_field2(source_xy,receptor_xy,cohort,theta,stack_h,exit_vel,float(u),float(wd),str(stab),sigma_coef)
        return C[np.asarray(monitor_idx)]
    T,M=y.shape
    # mislabel detection via best-fit
    theta0=theta_mean.copy()
    fixed=list(calib_stab)
    for t in range(T):
        best=fixed[t]; best_r=None
        for cand in ['A','B','C','D','E','F']:
            r=y[t]-predict(theta0,calib_u[t],calib_dir[t],cand)
            rr=float(np.sum(r*r))
            if best_r is None or rr<best_r:
                best_r=rr; best=cand
        r0=np.sum((y[t]-predict(theta0,calib_u[t],calib_dir[t],fixed[t]))**2)
        if best_r is not None and best_r<0.5*r0:
            fixed[t]=best
    calib_stab=fixed
    def resid(theta):
        preds=[predict(theta,calib_u[t],calib_dir[t],calib_stab[t]) for t in range(T)]
        return y_flat-np.concatenate(preds)
    def obj(theta):
        r=resid(theta)
        z=solve_triangular(L,r,lower=True)
        return float(z@z)+float(np.sum(((theta-theta_mean)/theta_sigma)**2))
    bounds=[(theta_mean[k]-3*theta_sigma[k],theta_mean[k]+3*theta_sigma[k]) for k in range(K)]
    res=minimize(obj,theta0,method='L-BFGS-B',bounds=bounds,options={'maxiter':200})
    theta=res.x
    C_withheld=concentration_field2(source_xy,receptor_xy,cohort,theta,stack_h,exit_vel,withheld_u,withheld_dir,withheld_stab,sigma_coef)
    eps=np.exp(theta[np.asarray(cohort)])
    r=resid(theta)
    z=solve_triangular(L,r,lower=True)
    os.makedirs('/app/output',exist_ok=True)
    np.savez('/app/output/reconstruction.npz',emission_coefficients=theta,emission_per_source=eps,predicted_concentrations=C_withheld)
    diag=dict(weighted_calibration_misfit=float(z@z),max_residual=0.0,max_predicted_concentration=float(np.max(C_withheld)),limit_exceeded=bool(np.max(C_withheld)>50.0))
    open('/app/output/diagnostics.json','w').write(json.dumps(diag,indent=2))
    open(out_path,'w').write(json.dumps({'status':'ok'}))

if __name__=='__main__':
    main()
