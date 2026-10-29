import json, os, subprocess, sys, tempfile, shutil, stat
import numpy as np
import pytest

TASK = "/tests"
HIDDEN = os.path.join(TASK, "hidden")
HIDDEN_EXP = os.path.join(TASK, "hidden_expected")
MAX_OUT = 5 * 1024 * 1024

sys.path.insert(0, "/app/data")
try:
    from dispersion_solver.plume import concentration_field
except Exception:
    pass

def case_names():
    return sorted(os.listdir(HIDDEN))

def run_case(case):
    src = os.path.join(HIDDEN, case)
    tmp = tempfile.mkdtemp()
    work = os.path.join(tmp, "case")
    shutil.copytree(src, work)
    for root, dirs, files in os.walk(work):
        os.chmod(root, 0o555)
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
    outdir = os.path.join(tmp, "out")
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o777)
    uid = 10001
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": "/app/data"}
    try:
        proc = subprocess.run(
            ["python3", "/app/solve.py", work, os.path.join(outdir, "result.json")],
            timeout=120, env=env, user=uid, group=uid, extra_groups=[],
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        proc = None
    def safe_read_npz(p):
        try:
            st = os.lstat(p)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_OUT:
            return None
        try:
            with open(p, "rb") as f:
                data = f.read(MAX_OUT + 1)
            if len(data) > MAX_OUT:
                return None
            import io
            return np.load(io.BytesIO(data))
        except Exception:
            return None
    def safe_read_json(p):
        try:
            st = os.lstat(p)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_OUT:
            return None
        try:
            with open(p, "rb") as f:
                data = f.read(MAX_OUT + 1)
            if len(data) > MAX_OUT:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None
    recon = safe_read_npz(os.path.join(outdir, "reconstruction.npz"))
    diag = safe_read_json(os.path.join(outdir, "diagnostics.json"))
    shutil.rmtree(tmp, ignore_errors=True)
    return proc, recon, diag

@pytest.mark.parametrize("case", case_names())
def test_program_present(case):
    assert os.path.isfile("/app/solve.py")
    assert not os.path.islink("/app/solve.py")

@pytest.mark.parametrize("case", case_names())
def test_runs_and_output(case):
    proc, recon, diag = run_case(case)
    assert proc is not None, "timeout"
    assert proc.returncode == 0, f"nonzero exit: {proc.stderr}"
    assert recon is not None, "no reconstruction.npz"
    assert diag is not None, "no diagnostics.json"
    assert set(recon.files) == {"emission_coefficients", "emission_per_source", "predicted_concentrations"}
    assert set(diag.keys()) == {"weighted_calibration_misfit", "max_residual", "max_predicted_concentration", "limit_exceeded"}

@pytest.mark.parametrize("case", case_names())
def test_values(case):
    proc, recon, diag = run_case(case)
    ec = recon["emission_coefficients"]
    eps = recon["emission_per_source"]
    pc = recon["predicted_concentrations"]
    assert np.all(np.isfinite(ec)) and np.all(np.isfinite(eps)) and np.all(np.isfinite(pc))
    # admissibility
    meta_path = os.path.join(HIDDEN_EXP, case + "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    prior = meta["prior"]
    pm = np.array(prior["theta_mean"])
    ps = np.array(prior["theta_sigma"])
    assert np.all(ec >= pm - 3 * ps)
    assert np.all(ec <= pm + 3 * ps)
    # ground truth
    with open(os.path.join(HIDDEN_EXP, case + ".json")) as f:
        truth = json.load(f)
    theta_true = np.array(truth["theta"])
    pred_true = np.array(truth["predicted_concentrations"])
    # coefficients within tolerance
    assert np.allclose(ec, theta_true, atol=0.5, rtol=0.1), "coefficients mismatch"
    # predicted concentrations within tolerance (relative on non-negligible)
    mask = np.abs(pred_true) > 1e-12
    if mask.any():
        assert np.allclose(pc[mask], pred_true[mask], atol=1e-6, rtol=0.1), "prediction mismatch"
    # consistency: max_residual small
    assert diag["max_residual"] <= 1e-3
    # limit_exceeded consistency
    assert bool(diag["limit_exceeded"]) == bool(np.max(pc) > 50.0)
    # weighted misfit finite
    assert np.isfinite(diag["weighted_calibration_misfit"])

def test_disjoint_names():
    ex_dir = "/app/data/examples"
    if os.path.isdir(ex_dir):
        ex = set(os.listdir(ex_dir))
    else:
        ex = set()
    hidden = set(os.listdir(HIDDEN))
    assert ex.isdisjoint(hidden)
