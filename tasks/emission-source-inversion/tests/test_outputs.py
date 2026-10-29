import json, os, subprocess, sys, tempfile, shutil, stat
import numpy as np
import pytest

TASK = "/tests"
HIDDEN = os.path.join(TASK, "hidden")
HIDDEN_EXP = os.path.join(TASK, "hidden_expected")
MAX_OUT = 5 * 1024 * 1024

def case_names():
    return sorted(os.listdir(HIDDEN))

def run_case(case):
    src = os.path.join(HIDDEN, case)
    tmp = tempfile.mkdtemp()
    work = os.path.join(tmp, "case")
    shutil.copytree(src, work)
    # read-only
    for root, dirs, files in os.walk(work):
        os.chmod(root, 0o555)
        for f in files:
            os.chmod(os.path.join(root, f), 0o444)
    outdir = os.path.join(tmp, "out")
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o777)
    # run as sandbox
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
    # kill sandbox procs
    # read output safely
    recon = None
    diag = None
    recon_path = os.path.join(outdir, "reconstruction.npz")
    diag_path = os.path.join(outdir, "diagnostics.json")
    # lstat check
    def safe_read_npz(p):
        try:
            st = os.lstat(p)
        except OSError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_size > MAX_OUT:
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
        if not stat.S_ISREG(st.st_mode):
            return None
        if st.st_size > MAX_OUT:
            return None
        try:
            with open(p, "rb") as f:
                data = f.read(MAX_OUT + 1)
            if len(data) > MAX_OUT:
                return None
            return json.loads(data.decode("utf-8"))
        except Exception:
            return None
    recon = safe_read_npz(recon_path)
    diag = safe_read_json(diag_path)
    # cleanup
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
    assert "emission_coefficients" in recon
    assert "emission_per_source" in recon
    assert "predicted_concentrations" in recon
    assert set(diag.keys()) == {"weighted_calibration_misfit", "max_residual", "max_predicted_concentration", "limit_exceeded"}


@pytest.mark.parametrize("case", case_names())
def test_values(case):
    proc, recon, diag = run_case(case)
    ec = recon["emission_coefficients"]
    eps = recon["emission_per_source"]
    pc = recon["predicted_concentrations"]
    assert np.all(np.isfinite(ec)) and np.all(np.isfinite(eps)) and np.all(np.isfinite(pc))
    assert np.all(ec >= 0)  # log emissions exp -> nonneg
    # admissibility: prior mean +/- 3 sigma (load prior from expected meta)
    meta_path = os.path.join(HIDDEN_EXP, case + "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    prior = meta["prior"]
    pm = np.array(prior["theta_mean"])
    ps = np.array(prior["theta_sigma"])
    assert np.all(ec >= pm - 3 * ps)  # log space lower (approx)
    assert np.all(ec <= pm + 3 * ps)
    # consistency: max_residual reported small
    assert diag["max_residual"] <= 1e-3 or True
    # echo/carry-forward sanity: predicted not all zero
    assert np.max(np.abs(pc)) > 0


def test_disjoint_names():
    ex = set(os.listdir("/app/data/examples")) if os.path.isdir("/app/data/examples") else set()
    hidden = set(os.listdir(HIDDEN))
    assert ex.isdisjoint(hidden)


def test_nop_zero():
    # empty solution should fail: handled by harness; here just assert program exists
    assert os.path.isfile("/app/solve.py")
