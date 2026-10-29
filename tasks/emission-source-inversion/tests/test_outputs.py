import json, os, subprocess, sys, tempfile, shutil, stat, math, csv, numpy as np, pytest
TASK='/tests'
HIDDEN=os.path.join(TASK,'hidden')
HIDDEN_EXP=os.path.join(TASK,'hidden_expected')
MAX_OUT=5*1024*1024
def case_names():
 return sorted(os.listdir(HIDDEN))
def run_case(case):
 tmp=tempfile.mkdtemp(); work=os.path.join(tmp,'case'); shutil.copytree(os.path.join(HIDDEN,case),work)
 for root,dirs,files in os.walk(work):
  os.chmod(root,0o555)
  for f in files:
   os.chmod(os.path.join(root,f),0o444)
 outdir=os.path.join(tmp,'out'); os.makedirs(outdir,exist_ok=True); os.chmod(outdir,0o777)
 uid=10001; env={'PATH':'/usr/bin:/bin','PYTHONPATH':'/app/data'}
 try:
  proc=subprocess.run(['python3','/app/solve.py',work,os.path.join(outdir,'result.json')],timeout=120,env=env,user=uid,group=uid,extra_groups=[],capture_output=True,text=True)
 except subprocess.TimeoutExpired:
  proc=None
 import io
 def safe_npz(p):
  st=os.lstat(p) if os.path.exists(p) else None
  if st is None or not stat.S_ISREG(st.st_mode) or st.st_size>MAX_OUT: return None
  
  with open(p,'rb') as f:
   data=f.read(MAX_OUT+1)
  if len(data)>MAX_OUT: return None
  
  return np.load(io.BytesIO(data))
 def safe_json(p):
  st=os.lstat(p) if os.path.exists(p) else None
  if st is None or not stat.S_ISREG(st.st_mode) or st.st_size>MAX_OUT: return None
  
  with open(p,'rb') as f:
   data=f.read(MAX_OUT+1)
  if len(data)>MAX_OUT: return None
  
  return json.loads(data.decode('utf-8'))
 recon=safe_npz(os.path.join(outdir,'reconstruction.npz')); diag=safe_json(os.path.join(outdir,'diagnostics.json'))
 shutil.rmtree(tmp,ignore_errors=True); return proc,recon,diag
@pytest.mark.parametrize('case',case_names())
def test_program_present(case):
 assert os.path.isfile('/app/solve.py') and not os.path.islink('/app/solve.py')
@pytest.mark.parametrize('case',case_names())
def test_runs_and_output(case):
 proc,recon,diag=run_case(case); assert proc is not None; assert proc.returncode==0; assert recon is not None; assert diag is not None
 assert set(recon.files)=={'emission_coefficients','emission_per_source','predicted_concentrations'}
 assert set(diag.keys())=={'weighted_calibration_misfit','max_residual','max_predicted_concentration','limit_exceeded'}
@pytest.mark.parametrize('case',case_names())
def test_values(case):
 proc,recon,diag=run_case(case); ec=recon['emission_coefficients']; pc=recon['predicted_concentrations']
 assert np.all(np.isfinite(ec)) and np.all(np.isfinite(pc))
 meta=json.load(open(os.path.join(HIDDEN_EXP,case+'_meta.json')))
 pm=np.array(meta['prior']['theta_mean']); ps=np.array(meta['prior']['theta_sigma'])
 assert np.all(ec>=pm-3*ps) and np.all(ec<=pm+3*ps)
 truth=json.load(open(os.path.join(HIDDEN_EXP,case+'.json'))); theta_true=np.array(truth['theta']); pred_true=np.array(truth['predicted_concentrations'])
 assert np.allclose(ec,theta_true,atol=0.5,rtol=0.1)
 mask=np.abs(pred_true)>1e-12
 if mask.any():
  assert np.allclose(pc[mask],pred_true[mask],atol=1e-6,rtol=0.1)
 assert diag['max_residual']<=1e-3
 assert bool(diag['limit_exceeded'])==bool(np.max(pc)>50.0)
def test_disjoint_names():
 ex=set(os.listdir('/app/data/examples')) if os.path.isdir('/app/data/examples') else set(); assert ex.isdisjoint(set(os.listdir(HIDDEN)))


@pytest.mark.parametrize("case", sorted(os.listdir("/tests/heldout_real")) if os.path.isdir("/tests/heldout_real") else [])
def test_behavior_heldout(case):
    # Only run for behavior json files
    if not case.endswith("_behavior.json"):
        pytest.skip("not a behavior case")
    import json as _js, numpy as _np
    beh = _js.loads(open(os.path.join("/tests/heldout_real", case)).read())
    sol_case = case.replace("_behavior.json", "")
    # Run solver on the corresponding hidden case dir (which shares geometry)
    proc, recon, diag = run_case(sol_case if sol_case in os.listdir(HIDDEN) else sorted(os.listdir(HIDDEN))[0])
    assert recon is not None and diag is not None
    ec = recon["emission_coefficients"]; pc = recon["predicted_concentrations"]
    import numpy as np
    # Gate 1: improvement over prior
    prior = np.array(beh["theta_prior"]); truth = np.array(beh["theta_true"])
    err_prior = float(np.linalg.norm(prior - truth))
    err_sol = float(np.linalg.norm(ec - truth))
    assert err_sol < 0.9 * err_prior, f"no improvement over prior: {err_sol:.3f} vs {err_prior:.3f}"
    # Gate 2: envelope
    lo = np.array(beh["predicted_envelope_lo"]); hi = np.array(beh["predicted_envelope_hi"])
    assert np.all(pc >= lo - 1e-9) and np.all(pc <= hi + 1e-9), "predicted outside physics envelope"
    # Gate 3: compliance behavior
    assert bool(diag["limit_exceeded"]) == bool(np.max(pc) > 50.0)

