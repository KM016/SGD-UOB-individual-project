# =============================================================================
# Imports
# =============================================================================
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Safety Constants
# ============================================================
OBJ_CAP = 1e50
STAT_CAP = 1e50
XNORM_CAP = 1e25
VAR_CAP = 1e50

GAP_FLOOR = 1e-16
VAR_FLOOR = 1e-16
STAT_FLOOR = 1e-14

FIG_FACECOLOR = "lightsteelblue"
AX_FACECOLOR  = "lemonchiffon"

# ============================================================
# Problem Setup
# ============================================================

def make_ill_conditioned_least_squares(n=2000, d=50, noise_std=0.1, mu_ridge=1e-6, seed=42, kappa_target=1e4):
    """
    Generates ill-conditioned ridge least-squares data.

    Constructs A via SVD with a prescribed singular value spectrum, then rescales
    so that lambda_max(A^T A / n) = 1 - mu_ridge, giving L = lambda_max(H) = 1
    exactly. Setting mu_ridge = 1/kappa_target then gives kappa = L/lambda_min ≈
    kappa_target (the factor ~2 difference is because lambda_min ≈ 2*mu_ridge).

    Returns:
      - A: (n, d) ill-conditioned feature matrix with L=1
      - b: (n,) target vector
      - x_true: (d,) truth parameter vector
      - mu: ridge regularisation parameter (mu_ridge), returned as float
    """
    rng = np.random.default_rng(seed)

    # Build A with prescribed singular values via SVD: A = U * S * Vt
    U, _ = np.linalg.qr(rng.normal(size=(n, d)))
    Vt, _ = np.linalg.qr(rng.normal(size=(d, d)))

    # Singular values log-spaced so sigma_max/sigma_min = sqrt(kappa_target),
    # making lambda_max(A^TA/n) / lambda_min(A^TA/n) = kappa_target before ridge.
    svals = np.exp(np.linspace(0, 0.5 * np.log(kappa_target), d)).astype(np.float64)
    svals = svals / svals.max()
    S = np.diag(svals)
    A = (U[:, :d] @ S @ Vt).astype(np.float64)

    # Rescale so lambda_max(A^TA/n) = 1 - mu_ridge → L = lambda_max(H) = 1 exactly.
    ATA_max = float(np.linalg.eigvalsh(A.T @ A / n).max())
    target_ATA_max = max(1.0 - float(mu_ridge), 0.5)
    A = A * np.sqrt(target_ATA_max / ATA_max)

    x_true = rng.normal(size=(d,)).astype(np.float64)
    b = (A @ x_true + noise_std * rng.normal(size=(n,))).astype(np.float64)
    return A, b, x_true, float(mu_ridge)

def standardise_problem(A, b, eps=1e-12):
    A = (A - A.mean(axis=0)) / (A.std(axis=0) + eps)
    b = b - b.mean()
    return A.astype(np.float64), b.astype(np.float64)

def fi_grad(A, b, i, x, mu=0.0):
    return (A[i] @ x - b[i]) * A[i] + mu * x

def F_grad(A, b, x, mu=0.0):
    return (A.T @ (A @ x - b)) / A.shape[0] + mu * x

def F_value(A, b, x, mu=0.0):
    r = A @ x - b
    return 0.5 * np.mean(r * r) + 0.5 * mu * np.sum(x * x)

def P_value(A, b, x, mu=0.0, lam=0.0):
    return F_value(A, b, x, mu) + lam * np.sum(np.abs(x))

def prox_l1(x, alpha, lam):
    return np.sign(x) * np.maximum(np.abs(x) - alpha * lam, 0.0)

def gradient_mapping(A, b, x, alpha, mu, lam):
    grad = F_grad(A, b, x, mu)
    return (x - prox_l1(x - alpha * grad, alpha, lam)) / alpha

def estimate_L_and_m(A, mu=0.0):
    H = (A.T @ A) / A.shape[0] + mu * np.eye(A.shape[1], dtype=np.float64)
    evals = np.linalg.eigvalsh(H)
    return float(evals.max()), float(evals.min())

def solve_smooth_star(A, b, mu):
    n, d = A.shape
    H = (A.T @ A) / n + mu * np.eye(d, dtype=np.float64)
    return np.linalg.solve(H, (A.T @ b) / n)

def solve_composite_star(A, b, mu, lam, max_iter=50000, tol=1e-12, verbose=True):
    x = np.zeros(A.shape[1], dtype=np.float64)
    L, _ = estimate_L_and_m(A, mu)
    alpha = 0.99 / L

    for it in range(max_iter):
        x_old = x.copy()
        x = prox_l1(x - alpha * F_grad(A, b, x, mu), alpha, lam)
        if np.linalg.norm(x - x_old) < tol:
            residual = np.linalg.norm(
                x - prox_l1(x - alpha * F_grad(A, b, x, mu), alpha, lam)
            ) / alpha
            if verbose:
                print(f"  Reference ISTA converged: {it+1} iters, residual={residual:.2e}")
            return x, float(residual)

    residual = np.linalg.norm(
        x - prox_l1(x - alpha * F_grad(A, b, x, mu), alpha, lam)
    ) / alpha
    if verbose:
        print(f"  Reference ISTA max_iter, residual={residual:.2e}")
    return x, float(residual)

# ============================================================
# Variance Diagnostics
# ============================================================

def estimate_variance(A, b, x, mu, indices, estimator="sgd", x_snap=None, mu_grad_snap=None):
    full_grad = F_grad(A, b, x, mu)
    if not np.all(np.isfinite(full_grad)):
        return np.inf

    sq = 0.0
    for i in indices:
        if estimator == "sgd":
            gi = fi_grad(A, b, i, x, mu)

        elif estimator == "svrg":
            if x_snap is None or mu_grad_snap is None:
                return np.nan
            
            gi = fi_grad(A, b, i, x, mu) - fi_grad(A, b, i, x_snap, mu) + mu_grad_snap

        else:
            return np.nan

        diff = gi - full_grad
        if not np.all(np.isfinite(diff)):
            return np.inf

        dd = float(np.dot(diff, diff))
        if (not np.isfinite(dd)) or (dd > VAR_CAP):
            return np.inf
        
        sq += dd

        if sq > VAR_CAP:
            return np.inf

    return sq / max(len(indices), 1)


def compute_metrics(A, b, x, mu=0.0, lam=0.0, alpha=None,
    x_star=None, obj_star=None, is_composite=False,
    variance_indices=None, estimator="sgd", x_snap=None, mu_grad_snap=None):
    out = {}

    if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > XNORM_CAP):
        out.update({"objective": np.inf, "obj_gap": np.inf, "dist2": np.inf,
                    "stationarity": np.inf, "variance": np.nan, "diverged": True})
        return out

    obj = P_value(A, b, x, mu, lam) if is_composite else F_value(A, b, x, mu)

    if (not np.isfinite(obj)) or (obj > OBJ_CAP):
        out.update({"objective": np.inf, "obj_gap": np.inf, "dist2": np.inf,
                    "stationarity": np.inf, "variance": np.nan, "diverged": True})
        return out

    out["objective"] = float(obj)
    out["obj_gap"] = float(max(obj - obj_star, 0.0)) if obj_star is not None else np.nan
    out["dist2"] = float(np.sum((x - x_star) ** 2)) if x_star is not None else np.nan

    if is_composite and (alpha is not None):
        st = float(np.linalg.norm(gradient_mapping(A, b, x, alpha, mu, lam)))
    else:
        st = float(np.linalg.norm(F_grad(A, b, x, mu)))

    if (not np.isfinite(st)) or (st > STAT_CAP):
        out.update({"objective": np.inf, "obj_gap": np.inf, "dist2": np.inf,
                    "stationarity": np.inf, "variance": np.nan, "diverged": True})
        return out

    out["stationarity"] = st

    if variance_indices is not None and len(variance_indices) > 0:
        v = estimate_variance(A, b, x, mu, variance_indices, estimator, x_snap, mu_grad_snap)
        out["variance"] = float(v) if np.isfinite(v) else np.inf
    else:
        out["variance"] = np.nan

    out["diverged"] = False
    return out

# ============================================================
# Algorithms
# ============================================================

# rm stepsize schedule 
def rm_stepsize_schedule(alpha0, b=1000):
    b = int(max(b, 1))
    a = float(alpha0) * b
    def alpha_k(k):
        return a / (k + b)
    return alpha_k

# SGD
def run_sgd(A, b, x0, steps, alpha, mu=0.0, x_star=None, obj_star=None,
            seed=42, log_every=200, variance_sample_size=100, alpha_fn=None):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    x = x0.copy()
    grad_calls = 0
    logs = []

    alpha0 = float(alpha)
    var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
    a_now = alpha_fn(0) if alpha_fn is not None else alpha0
    m = compute_metrics(A, b, x, mu, 0.0, a_now, x_star, obj_star, False, var_idx, "sgd")
    m.update({"k": 0, "grad_calls": 0, "alpha": float(a_now)})
    logs.append(m)

    for k in range(steps):
        a_now = alpha_fn(k) if alpha_fn is not None else alpha0
        i = rng.integers(0, n)
        x = x - a_now * fi_grad(A, b, i, x, mu)
        grad_calls += 1

        if (k % log_every == 0) or (k == steps - 1):
            var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
            m = compute_metrics(A, b, x, mu, 0.0, a_now, x_star, obj_star, False, var_idx, "sgd")
            m.update({"k": k + 1, "grad_calls": grad_calls, "alpha": float(a_now)})
            logs.append(m)
            if m.get("diverged", False):
                break

        if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > XNORM_CAP):
            break

    return x, logs

# PROX-SGD
def run_prox_sgd(A, b, x0, steps, alpha, mu=0.0, lam=0.0, x_star=None, obj_star=None,
                 seed=42, log_every=200, variance_sample_size=100, alpha_fn=None):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    x = x0.copy()
    grad_calls = 0
    logs = []

    alpha0 = float(alpha)
    var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
    a_now = alpha_fn(0) if alpha_fn is not None else alpha0
    m = compute_metrics(A, b, x, mu, lam, a_now, x_star, obj_star, True, var_idx, "sgd")
    m.update({"k": 0, "grad_calls": 0, "alpha": float(a_now)})
    logs.append(m)

    for k in range(steps):
        a_now = alpha_fn(k) if alpha_fn is not None else alpha0
        i = rng.integers(0, n)
        x = prox_l1(x - a_now * fi_grad(A, b, i, x, mu), a_now, lam)
        grad_calls += 1

        if (k % log_every == 0) or (k == steps - 1):
            var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
            m = compute_metrics(A, b, x, mu, lam, a_now, x_star, obj_star, True, var_idx, "sgd")
            m.update({"k": k + 1, "grad_calls": grad_calls, "alpha": float(a_now)})
            logs.append(m)
            if m.get("diverged", False):
                break

        if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > XNORM_CAP):
            break

    return x, logs

# SVRGD
def run_svrg(A, b, x0, epochs, m_inner, alpha0, mu=0.0, x_star=None, obj_star=None,
             seed=42, log_every=1, variance_sample_size=100,
             backoff_factor=0.5, max_backoffs=8, divergence_mult=50.0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    x_tilde = x0.copy()
    grad_calls = 0
    logs = []
    alpha = float(alpha0)

    var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
    m = compute_metrics(A, b, x_tilde, mu, 0.0, alpha, x_star, obj_star, False,
                        var_idx, "svrg", x_tilde, F_grad(A, b, x_tilde, mu))
    m.update({"epoch": 0, "grad_calls": 0, "alpha": float(alpha)})
    logs.append(m)

    for s in range(epochs):
        x_snap = x_tilde.copy()
        mu_grad_snap = F_grad(A, b, x_snap, mu)
        obj_start = F_value(A, b, x_snap, mu)
        grad_calls += n

        success = False
        x_new = None

        for _attempt in range(max_backoffs + 1):
            x = x_snap.copy()
            diverged_inner = False

            for _ in range(m_inner):
                i = rng.integers(0, n)
                g = fi_grad(A, b, i, x, mu) - fi_grad(A, b, i, x_snap, mu) + mu_grad_snap
                grad_calls += 2
                x = x - alpha * g

                if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > XNORM_CAP):
                    diverged_inner = True
                    break

            if not diverged_inner:
                obj_end = F_value(A, b, x, mu)
                if np.isfinite(obj_end) and (obj_end <= divergence_mult * max(obj_start, 1e-16)) and (obj_end <= OBJ_CAP):
                    success = True
                    x_new = x
                    break

            alpha *= backoff_factor

        if not success:
            logs.append({"epoch": s + 1, "grad_calls": grad_calls, "alpha": float(alpha),
                         "diverged": True, "objective": np.inf, "obj_gap": np.inf})
            return x_tilde, logs

        x_tilde = x_new

        if (s % log_every == 0) or (s == epochs - 1):
            var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
            mm = compute_metrics(A, b, x_tilde, mu, 0.0, alpha, x_star, obj_star, False,
                                 var_idx, "svrg", x_snap, mu_grad_snap)
            mm.update({"epoch": s + 1, "grad_calls": grad_calls, "alpha": float(alpha)})
            logs.append(mm)
            if mm.get("diverged", False):
                return x_tilde, logs

    return x_tilde, logs

# PROX-SVRGD
def run_prox_svrg(A, b, x0, epochs, m_inner, alpha0, mu=0.0, lam=0.0, x_star=None, obj_star=None,
                  seed=42, log_every=1, variance_sample_size=100,
                  backoff_factor=0.5, max_backoffs=8, divergence_mult=50.0):
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    x_tilde = x0.copy()
    grad_calls = 0
    logs = []
    alpha = float(alpha0)

    var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
    m = compute_metrics(A, b, x_tilde, mu, lam, alpha, x_star, obj_star, True,
                        var_idx, "svrg", x_tilde, F_grad(A, b, x_tilde, mu))
    m.update({"epoch": 0, "grad_calls": 0, "alpha": float(alpha)})
    logs.append(m)

    for s in range(epochs):
        x_snap = x_tilde.copy()
        mu_grad_snap = F_grad(A, b, x_snap, mu)
        obj_start = P_value(A, b, x_snap, mu, lam)
        grad_calls += n

        success = False
        x_new = None

        for _attempt in range(max_backoffs + 1):
            x = x_snap.copy()
            diverged_inner = False

            for _ in range(m_inner):
                i = rng.integers(0, n)
                g = fi_grad(A, b, i, x, mu) - fi_grad(A, b, i, x_snap, mu) + mu_grad_snap
                grad_calls += 2
                x = prox_l1(x - alpha * g, alpha, lam)

                if (not np.all(np.isfinite(x))) or (np.linalg.norm(x) > XNORM_CAP):
                    diverged_inner = True
                    break

            if not diverged_inner:
                obj_end = P_value(A, b, x, mu, lam)
                if np.isfinite(obj_end) and (obj_end <= divergence_mult * max(obj_start, 1e-16)) and (obj_end <= OBJ_CAP):
                    success = True
                    x_new = x
                    break

            alpha *= backoff_factor

        if not success:
            logs.append({"epoch": s + 1, "grad_calls": grad_calls, "alpha": float(alpha),
                         "diverged": True, "objective": np.inf, "obj_gap": np.inf})
            return x_tilde, logs

        x_tilde = x_new

        if (s % log_every == 0) or (s == epochs - 1):
            var_idx = rng.integers(0, n, size=min(variance_sample_size, n))
            mm = compute_metrics(A, b, x_tilde, mu, lam, alpha, x_star, obj_star, True,
                                 var_idx, "svrg", x_snap, mu_grad_snap)
            mm.update({"epoch": s + 1, "grad_calls": grad_calls, "alpha": float(alpha)})
            logs.append(mm)
            if mm.get("diverged", False):
                return x_tilde, logs

    return x_tilde, logs

# ============================================================
# Analysis helpers
# ============================================================

# calculates floor 
def floor_for(ykey):
    if ykey == "obj_gap":   return GAP_FLOOR
    if ykey == "variance":  return VAR_FLOOR
    if ykey == "stationarity": return STAT_FLOOR
    return 1e-16

# extracts and cleans (x,y)
def to_xy(logs, xkey="grad_calls", ykey="obj_gap", y_floor=1e-16):
    pairs = []
    for r in logs:
        if r.get("diverged", False): continue
        if xkey not in r or ykey not in r: continue
        x = float(r[xkey]); y = float(r[ykey])
        if not np.isfinite(x) or not np.isfinite(y): continue
        y = max(y, 0.0); y = max(y, y_floor)
        pairs.append((x, y))
    if not pairs: return np.array([]), np.array([])
    pairs.sort()
    xs, ys = zip(*pairs)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

# aggregates log curves 
def aggregate_log_curves(all_logs, xkey="grad_calls", ykey="obj_gap", y_floor=1e-16):
    xs_all = []
    for logs in all_logs:
        xs, _ = to_xy(logs, xkey, ykey, y_floor)
        if len(xs) > 0: xs_all.append(xs)
    if not xs_all: return np.array([]), np.array([]), np.array([]), np.array([])

    x_grid = np.unique(np.sort(np.concatenate(xs_all)))
    logY = []
    for logs in all_logs:
        xs, ys = to_xy(logs, xkey, ykey, y_floor)
        if len(xs) == 0: continue
        ly = np.log(np.clip(ys, y_floor, np.inf))
        ly_i = np.interp(x_grid, xs, ly)
        ly_i[(x_grid < xs.min()) | (x_grid > xs.max())] = np.nan
        logY.append(ly_i)

    if not logY: return np.array([]), np.array([]), np.array([]), np.array([])

    LogY = np.array(logY, dtype=np.float64)
    m = np.nanmean(LogY, axis=0)
    s = np.nanstd(LogY, axis=0)
    valid = np.isfinite(m)
    xg = x_grid[valid]
    ym = np.exp(m[valid]); ylo = np.exp(m[valid] - s[valid]); yhi = np.exp(m[valid] + s[valid])
    return xg, ym, ylo, yhi

# fits linear rate 
def fit_linear_rate(logs, ykey="obj_gap", y_hi=1e-2, y_lo=1e-10):
    xs, ys = to_xy(logs, ykey=ykey, y_floor=floor_for(ykey))
    if len(xs) < 8: return np.nan, np.nan, 0
    mask = (ys <= y_hi) & (ys >= y_lo) & np.isfinite(ys)
    xs_w, ys_w = xs[mask], ys[mask]
    if len(xs_w) < 8: return np.nan, np.nan, 0
    ly = np.log(np.clip(ys_w, floor_for(ykey), np.inf))
    res = stats.linregress(xs_w, ly)
    return float(res.slope), float(res.rvalue ** 2), int(len(xs_w))

# fits power rate 
def fit_power_rate(logs, ykey="obj_gap", y_hi=1e-2, y_lo=1e-10):
    xs, ys = to_xy(logs, ykey=ykey, y_floor=floor_for(ykey))
    if len(xs) < 8: return np.nan, np.nan, 0
    mask = (xs > 0) & (ys <= y_hi) & (ys >= y_lo) & np.isfinite(ys)
    xs_w, ys_w = xs[mask], ys[mask]
    if len(xs_w) < 8: return np.nan, np.nan, 0
    lx = np.log(xs_w); ly = np.log(np.clip(ys_w, floor_for(ykey), np.inf))
    res = stats.linregress(lx, ly)
    return float(res.slope), float(res.rvalue ** 2), int(len(xs_w))

# looks at ys over the tail of the x range
def tail_stats(logs, ykey="obj_gap", tail_frac=0.25):
    xs, ys = to_xy(logs, ykey=ykey, y_floor=floor_for(ykey))
    if len(xs) < 8: return np.nan, np.nan, 0
    cut = xs.min() + (1.0 - tail_frac) * (xs.max() - xs.min())
    yt = ys[xs >= cut]
    if len(yt) < 3: return np.nan, np.nan, 0
    return float(np.mean(yt)), float(np.std(yt)), int(len(yt))

# dumps logs 
def dump_logs_json(filename, all_logs):
    with open(filename, "w") as f:
        json.dump(all_logs, f, indent=2)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ILL-CONDITIONED Experiment Configurations
    #
    # Key design choices:
    #   mu = 1e-4 = 1/kappa_target -> after A rescaling L=1, so κ = L/λ_min ≈ 5000
    #   lam = 5e-4 -> x* is partially sparse (~11/50 zeros), composite case non-trivial
    #   svrg_epochs=120, svrg_m=4000 -> enough budget for SVRG geometric decay to show
    #   seeds=[0,1,2,3,4], alpha_sgd=0.05/L, alpha_vr=0.02/L

    cfg = {
        "n": 2000, "d": 50,
        "noise_std": 0.1,
        "mu": 1e-4,         # 1/kappa_target κ ≈ 5000 after A rescaling (L=1)
        "lam": 5e-4,        # small enough that x* is NOT all-zeros
        "kappa_target": 1e4,

        "seeds": [0, 1, 2, 3, 4],

        "svrg_epochs": 120,
        "svrg_m": 4000,

        "log_every_sgd": 200,
        "log_every_svrg": 1,
        "variance_sample_size": 120,

        "tail_frac": 0.25,
        "rate_y_hi_smooth": 1e-2,
        "rate_y_lo_smooth": 1e-10,
        "rate_y_hi_comp": 1e-2,
        "rate_y_lo_comp": 1e-10,
    }

    # Setup – ill-conditioned problem
    A, b, _, mu = make_ill_conditioned_least_squares(
        cfg["n"], cfg["d"], cfg["noise_std"], cfg["mu"],
        seed=42, kappa_target=cfg["kappa_target"]
    )
    # NO standardisation as this would destroy ill-conditioning 
    # (We centre b to remove intercept bias, but leave A's scale intact)
    b = b - b.mean()

    L, m_sc = estimate_L_and_m(A, mu)
    kappa = L / max(m_sc, 1e-16)

    total_calls = cfg["svrg_epochs"] * (cfg["n"] + 2 * cfg["svrg_m"])
    sgd_steps = total_calls

    alpha_sgd = 0.05 / L   
    alpha_vr  = 0.02 / L   

    rm_b = 1000
    alpha_fn_sgd = rm_stepsize_schedule(alpha_sgd, b=rm_b)

    print("=" * 80)
    print(f"ILL-CONDITIONED SETUP: n={cfg['n']}, d={cfg['d']}, μ={mu:.2e}, L={L:.6f}, κ={kappa:.3e}")
    print(f"Steps: α_SGD={alpha_sgd:.6e} (αL={alpha_sgd*L:.4f}), α_VR={alpha_vr:.6e} (αL={alpha_vr*L:.4f})")
    print(f"Budget: {sgd_steps} component-gradient calls")
    print("=" * 80)

    x_smooth = solve_smooth_star(A, b, mu)
    x_comp, resid = solve_composite_star(A, b, mu, cfg["lam"])
    F_star = F_value(A, b, x_smooth, mu)
    P_star = P_value(A, b, x_comp, mu, cfg["lam"])

    print(f"F*={F_star:.8e}, P*={P_star:.8e}")
    print(f"Sparsity (<1e-6): {np.sum(np.abs(x_comp) < 1e-6)}/{cfg['d']}, residual={resid:.2e}")

    rng0 = np.random.default_rng(123)
    var_idx_star = rng0.integers(0, A.shape[0], size=min(500, A.shape[0]))
    var_sgd_at_star = estimate_variance(A, b, x_smooth, mu, var_idx_star, estimator="sgd")
    print(f"Estimated Var[∇f_i(x*)] (SGD oracle) ≈ {var_sgd_at_star:.3e}")

    x0 = np.zeros(cfg["d"], dtype=np.float64)

    print("\nRunning experiments...")
    logs_sgd_const = [
        run_sgd(A, b, x0, sgd_steps, alpha_sgd, mu, x_smooth, F_star,
                seed=s, log_every=cfg["log_every_sgd"],
                variance_sample_size=cfg["variance_sample_size"],
                alpha_fn=None)[1]
        for s in cfg["seeds"]
    ]
    logs_sgd_rm = [
        run_sgd(A, b, x0, sgd_steps, alpha_sgd, mu, x_smooth, F_star,
                seed=s, log_every=cfg["log_every_sgd"],
                variance_sample_size=cfg["variance_sample_size"],
                alpha_fn=alpha_fn_sgd)[1]
        for s in cfg["seeds"]
    ]
    logs_svrg = [
        run_svrg(A, b, x0, cfg["svrg_epochs"], cfg["svrg_m"], alpha_vr, mu, x_smooth, F_star,
                 seed=s, log_every=cfg["log_every_svrg"],
                 variance_sample_size=cfg["variance_sample_size"])[1]
        for s in cfg["seeds"]
    ]
    logs_psgd_const = [
        run_prox_sgd(A, b, x0, sgd_steps, alpha_sgd, mu, cfg["lam"], x_comp, P_star,
                     seed=s, log_every=cfg["log_every_sgd"],
                     variance_sample_size=cfg["variance_sample_size"],
                     alpha_fn=None)[1]
        for s in cfg["seeds"]
    ]
    logs_psgd_rm = [
        run_prox_sgd(A, b, x0, sgd_steps, alpha_sgd, mu, cfg["lam"], x_comp, P_star,
                    seed=s, log_every=cfg["log_every_sgd"],
                    variance_sample_size=cfg["variance_sample_size"],
                    alpha_fn=alpha_fn_sgd)[1]
        for s in cfg["seeds"]
    ]
    logs_psvrg = [
        run_prox_svrg(A, b, x0, cfg["svrg_epochs"], cfg["svrg_m"], alpha_vr, mu, cfg["lam"], x_comp, P_star,
                      seed=s, log_every=cfg["log_every_svrg"],
                      variance_sample_size=cfg["variance_sample_size"])[1]
        for s in cfg["seeds"]
    ]

    # Save logs
    dump_logs_json("logs/ill-conditioned/logs_sgd_const_illcond.json", logs_sgd_const)
    dump_logs_json("logs/ill-conditioned/logs_sgd_rm_illcond.json", logs_sgd_rm)
    dump_logs_json("logs/ill-conditioned/logs_svrg_illcond.json", logs_svrg)
    dump_logs_json("logs/ill-conditioned/logs_psgd_const_illcond.json", logs_psgd_const)
    dump_logs_json("logs/ill-conditioned/logs_psgd_rm_illcond.json", logs_psgd_rm)
    dump_logs_json("logs/ill-conditioned/logs_psvrg_illcond.json", logs_psvrg)

    # =========================================================
    # MECHANISM PLOTS
    # =========================================================
    print("\nGenerating mechanism plots (split smooth/composite)...")

    def style_axes(ax):
        ax.set_facecolor(AX_FACECOLOR)
        ax.grid(True, which="both", alpha=0.3)

    def add_floor_line(ax, y_floor):
        ax.axhline(y_floor, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(0.02, 0.05, f"plot floor = {y_floor:.0e}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=9, alpha=0.8)

    # ----------------------
    # Smooth
    # ----------------------
    fig_mech_s, axes_s = plt.subplots(1, 3, figsize=(18, 7))
    fig_mech_s.patch.set_facecolor(FIG_FACECOLOR)
    fig_mech_s.suptitle(f"ILL-CONDITIONED SETUP  (κ ≈ {kappa:.1e},  μ={mu:.0e})",
                        fontsize=13, fontweight="bold", y=1.01)

    smooth_panels = [
        (axes_s[0], "obj_gap", r"$F(x)-F^*$", "Smooth: objective gap"),
        (axes_s[1], "variance", r"$\mathbb{E}\|g_i-\nabla F\|^2$", "Smooth: estimator variance"),
        (axes_s[2], "stationarity", r"$\|\nabla F(x)\|$", "Smooth: stationarity"),
    ]

    for ax, ykey, ylabel, title in smooth_panels:
        y_floor = floor_for(ykey)

        for name, logs in [("SGD-const", logs_sgd_const), ("SGD-RM", logs_sgd_rm), ("SVRG", logs_svrg)]:
            xs, ym, ylo, yhi = aggregate_log_curves(logs, ykey=ykey, y_floor=y_floor)

            if len(xs) > 0:
                ax.plot(xs, ym, label=name, linewidth=2.5)
                ax.fill_between(xs, ylo, yhi, alpha=0.2)

        add_floor_line(ax, y_floor)
        ax.set_yscale("log"); ax.set_xlabel("Component gradient calls")
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", pad=10)
        style_axes(ax); ax.legend()


    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("figures/ill-conditioned/mechanism_plots_smooth_illcond.png", dpi=300,
                bbox_inches="tight", facecolor=fig_mech_s.get_facecolor())
    print("Saved mechanism_plots_smooth_illcond.png")

    # ----------------------
    # Composite
    # ----------------------
    fig_mech_c, axes_c = plt.subplots(1, 3, figsize=(18, 7))
    fig_mech_c.patch.set_facecolor(FIG_FACECOLOR)
    fig_mech_c.suptitle(f"ILL-CONDITIONED SETUP  (κ ≈ {kappa:.1e},  μ={mu:.0e})",
                        fontsize=13, fontweight="bold", y=1.01)

    comp_panels = [
        (axes_c[0], "obj_gap", r"$P(x)-P^*$", "Composite: objective gap"),
        (axes_c[1], "variance", r"$\mathbb{E}\|g_i-\nabla F\|^2$", "Composite: smooth-estimator variance"),
        (axes_c[2], "stationarity", r"$\|G_\alpha(x)\|$", "Composite: prox stationarity"),
    ]

    for ax, ykey, ylabel, title in comp_panels:
        y_floor = floor_for(ykey)
        for name, logs in [("Prox-SGD-const", logs_psgd_const), ("Prox-SGD-RM", logs_psgd_rm), ("Prox-SVRG", logs_psvrg)]:
            xs, ym, ylo, yhi = aggregate_log_curves(logs, ykey=ykey, y_floor=y_floor)
            if len(xs) > 0:
                ax.plot(xs, ym, label=name, linewidth=2.5)
                ax.fill_between(xs, ylo, yhi, alpha=0.2)
        add_floor_line(ax, y_floor)
        ax.set_yscale("log"); ax.set_xlabel("Component gradient calls")
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold", pad=10)
        style_axes(ax); ax.legend()

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("figures/ill-conditioned/mechanism_plots_composite_illcond.png", dpi=300,
                bbox_inches="tight", facecolor=fig_mech_c.get_facecolor())
    print("Saved mechanism_plots_composite_illcond.png")

    # =========================================================
    # SUMMARY PLOTS
    # =========================================================
    print("\nGenerating summary plots...")

    def summarise_method(all_logs, label, is_smooth=True, fit_kind="linear"):
        slopes, r2s, nfits = [], [], []
        for logs in all_logs:

            if fit_kind == "linear":
                slope, r2, nfit = fit_linear_rate(logs, ykey="obj_gap",
                    y_hi=cfg["rate_y_hi_smooth"] if is_smooth else cfg["rate_y_hi_comp"],
                    y_lo=cfg["rate_y_lo_smooth"] if is_smooth else cfg["rate_y_lo_comp"])
                
            elif fit_kind == "power":
                slope, r2, nfit = fit_power_rate(logs, ykey="obj_gap",
                    y_hi=cfg["rate_y_hi_smooth"] if is_smooth else cfg["rate_y_hi_comp"],
                    y_lo=cfg["rate_y_lo_smooth"] if is_smooth else cfg["rate_y_lo_comp"])
                
            else:
                slope, r2, nfit = np.nan, np.nan, 0

            slopes.append(slope); r2s.append(r2); nfits.append(nfit)

        gaps, dists, vars_end = [], [], []
        for logs in all_logs:
            gm, _, _ = tail_stats(logs, ykey="obj_gap", tail_frac=cfg["tail_frac"])
            dm, _, _ = tail_stats(logs, ykey="dist2", tail_frac=cfg["tail_frac"])
            gaps.append(gm); dists.append(dm)
            v = np.nan

            for r in reversed(logs):
                vv = r.get("variance", np.nan)

                if (not r.get("diverged", False)) and np.isfinite(vv):
                    v = float(vv); break
                
            vars_end.append(v)

        valid_fit = np.array(nfits) >= 8

        return {
            "label": label, "fit_kind": fit_kind,
            "fit_frac": float(np.mean(valid_fit)) if len(valid_fit) else 0.0,
            "nfit_mean": float(np.mean([n for n in nfits if n >= 8])) if np.any(valid_fit) else 0.0,
            "slope_mean": float(np.nanmean(slopes)),
            "slope_std":  float(np.nanstd(slopes)),
            "r2_mean":    float(np.nanmean(r2s)),
            "gap_tail_mean": float(np.nanmean(gaps)),
            "gap_tail_std":  float(np.nanstd(gaps)),
            "dist_tail_mean": float(np.nanmean(dists)),
            "dist_tail_std":  float(np.nanstd(dists)),
            "var_end_mean": float(np.nanmean(vars_end)),
            "var_end_std":  float(np.nanstd(vars_end)),
        }

    smooth_summaries = [
        summarise_method(logs_sgd_const, "SGD-const", is_smooth=True,  fit_kind="linear"),
        summarise_method(logs_sgd_rm, "SGD-RM", is_smooth=True, fit_kind="power"),
        summarise_method(logs_svrg, "SVRG", is_smooth=True, fit_kind="linear"),
    ]
    comp_summaries = [
        summarise_method(logs_psgd_const, "Prox-SGD-const", is_smooth=False, fit_kind="linear"),
        summarise_method(logs_psgd_rm, "Prox-SGD-RM", is_smooth=False, fit_kind="power"),
        summarise_method(logs_psvrg, "Prox-SVRG", is_smooth=False, fit_kind="linear"),
    ]

    # =========================================================
    # SUMMARY STATISTICS TEXT FILE
    # =========================================================
    print("\nWriting summary statistics...")

    def fmt(val, std=None, sci=True):
        """Format a value (and optional std) for display."""
        if not np.isfinite(val):
            return "N/A"
        if sci:
            s = f"{val:.3e}"
            if std is not None and np.isfinite(std):
                s += f" +/- {std:.2e}"
        else:
            s = f"{val:.4f}"
            if std is not None and np.isfinite(std):
                s += f" +/- {std:.4f}"
        return s

    def fmt_latex(val, std=None, sci=True):
        """Format a value for a LaTeX table cell."""
        if not np.isfinite(val):
            return r"\text{N/A}"
        if sci:
            exp = int(np.floor(np.log10(abs(val)))) if val != 0 else 0
            mantissa = val / 10**exp
            s = f"{mantissa:.2f} \\times 10^{{{exp}}}"
            if std is not None and np.isfinite(std):
                exp_s = int(np.floor(np.log10(abs(std)))) if std != 0 else 0
                m_s = std / 10**exp_s
                s += f" \\pm {m_s:.1f} \\times 10^{{{exp_s}}}"
        else:
            s = f"{val:.4f}"
            if std is not None and np.isfinite(std):
                s += f" \\pm {std:.4f}"
        return s

    with open("summary_statistics_illcond.txt", "w") as f:

        def w(line=""):
            f.write(line + "\n")

        # ── Header ────────────────────────────────────────────────────────
        w("=" * 80)
        w("ILL-CONDITIONED EXPERIMENT: SUMMARY STATISTICS")
        w("=" * 80)
        w(f"  Problem:  n={cfg['n']}, d={cfg['d']}, kappa_target={cfg['kappa_target']:.0e}")
        w(f"  Computed: L={L:.6f}, mu_min={m_sc:.4e}, kappa={kappa:.3e}")
        w(f"  Ridge:    mu={mu:.2e}")
        w(f"  L1 reg:   lam={cfg['lam']:.2e}  (composite case)")
        w(f"  Seeds:    {cfg['seeds']}  ({len(cfg['seeds'])} runs per method)")
        w(f"  Budget:   {sgd_steps:,} component-gradient calls per run")
        w(f"  SVRG:     {cfg['svrg_epochs']} epochs x {cfg['svrg_m']} inner steps")
        w(f"  Tail fraction for final-stage stats: last {int(cfg['tail_frac']*100)}% of iterates")
        w(f"  SGD oracle variance at x*: {var_sgd_at_star:.4e}  (theoretical noise floor)")
        w()

        for section_label, summaries, gap_label, stat_label in [
            ("SMOOTH CASE  (F(x) = (1/2n)||Ax-b||^2 + (mu/2)||x||^2)",
             smooth_summaries, "F(x) - F*", "||grad F(x)||"),
            ("COMPOSITE CASE  (P(x) = F(x) + lam*||x||_1)",
             comp_summaries,   "P(x) - P*", "||G_alpha(x)||  [prox-gradient mapping]"),
        ]:
            w("=" * 80)
            w(section_label)
            w("=" * 80)

            # ── 1. Convergence rate fit ────────────────────────────────────
            w()
            w("1. CONVERGENCE RATE FIT")
            w("-" * 60)
            w("   For SGD-const / Prox-SGD-const: linear fit of log(gap) vs calls")
            w("   For SGD-RM    / Prox-SGD-RM:    power-law fit  log(gap) ~ p*log(calls)")
            w("   For SVRG      / Prox-SVRG:       linear fit of log(gap) vs calls")
            w()
            w(f"   {'Method':<18} {'Fit type':<10} {'Slope / exponent p':<28} {'R^2':<8}")
            w(f"   {'-'*18} {'-'*10} {'-'*28} {'-'*8}")
            for s in summaries:
                kind_str = "linear" if s["fit_kind"] == "linear" else "power-law"
                slope_str = fmt(s["slope_mean"], s["slope_std"], sci=False)
                r2_str = f"{s['r2_mean']:.4f}"
                w(f"   {s['label']:<18} {kind_str:<10} {slope_str:<28} {r2_str}")
            w()
            w("   Interpretation:")
            w("     linear slope < 0: geometric (linear) convergence in log scale.")
            w("     power-law p < 0:  sublinear convergence; gap ~ k^p.")
            w("     slope ~ 0:        no convergence (plateaued at noise floor).")

            # ── 2. Final-stage objective gap ───────────────────────────────
            w()
            w(f"2. FINAL-STAGE OBJECTIVE GAP  ({gap_label})  [last {int(cfg['tail_frac']*100)}% of iterates]")
            w("-" * 60)
            w(f"   {'Method':<18} {'Mean':<18} {'Std':<18}")
            w(f"   {'-'*18} {'-'*18} {'-'*18}")
            for s in summaries:
                w(f"   {s['label']:<18} {s['gap_tail_mean']:<18.4e} {s['gap_tail_std']:<18.4e}")
            w()
            w("   Interpretation:")
            w("     SGD-const plateau = irreducible noise floor O(alpha).")
            w("     SGD-RM decays but slowly (power-law); still >>0 at end.")
            w("     SVRG drives gap to near-zero (variance-reduced estimator).")

            # ── 3. Final-stage distance to x* ─────────────────────────────
            w()
            w("3. FINAL-STAGE DISTANCE TO x*  (||x_k - x*||^2)  [last 25% of iterates]")
            w("-" * 60)
            w(f"   {'Method':<18} {'Mean':<18} {'Std':<18}")
            w(f"   {'-'*18} {'-'*18} {'-'*18}")
            for s in summaries:
                w(f"   {s['label']:<18} {s['dist_tail_mean']:<18.4e} {s['dist_tail_std']:<18.4e}")

            # ── 4. Variance at end ─────────────────────────────────────────
            w()
            w("4. GRADIENT ESTIMATOR VARIANCE AT END  (E||g_i - grad F||^2)")
            w("-" * 60)
            w(f"   SGD oracle variance at x* (theoretical floor): {var_sgd_at_star:.4e}")
            w()
            w(f"   {'Method':<18} {'Var[g_i](x_end) mean':<24} {'Std':<18}")
            w(f"   {'-'*18} {'-'*24} {'-'*18}")
            for s in summaries:
                w(f"   {s['label']:<18} {s['var_end_mean']:<24.4e} {s['var_end_std']:<18.4e}")
            w()
            w("   Interpretation:")
            w("     SGD/SGD-RM: variance stays at oracle level (noise does not vanish).")
            w("     SVRG: variance -> 0 as x_k -> x* (control-variate eliminates noise).")

            # ── 5. Stationarity ────────────────────────────────────────────
            w()
            w(f"5. STATIONARITY AT END  ({stat_label})  [last iterate]")
            w("-" * 60)
            for s in summaries:
                # pull last finite stationarity value from logs
                log_list = {
                    "SGD-const": logs_sgd_const, "SGD-RM": logs_sgd_rm,
                    "SVRG": logs_svrg, "Prox-SGD-const": logs_psgd_const,
                    "Prox-SGD-RM": logs_psgd_rm, "Prox-SVRG": logs_psvrg,
                }.get(s["label"], [])
                vals = []
                for seed_logs in log_list:
                    for r in reversed(seed_logs):
                        v = r.get("stationarity", np.nan)
                        if not r.get("diverged", False) and np.isfinite(v):
                            vals.append(v); break
                mean_stat = float(np.nanmean(vals)) if vals else np.nan
                std_stat  = float(np.nanstd(vals))  if vals else np.nan
                w(f"   {s['label']:<18} mean={mean_stat:.4e}   std={std_stat:.4e}")

            w()

        w("=" * 80)
        w("END OF SUMMARY")
        w("=" * 80)

    print("Saved summary_statistics_illcond.txt")
    print("Done!")