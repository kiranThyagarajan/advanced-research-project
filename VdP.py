import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Van der Pol dynamics (CT)
# =========================
def f_vdp(x, mu=1.0):
    """
    Autonomous Van der Pol oscillator (mu > 0).
    state x = [position, velocity] = [x, v]
    x' = v
    v' = mu*(1 - x**2)*v - x
    """
    pos, vel = x
    dpos = vel
    dvel = mu*(1.0 - pos**2)*vel - pos
    return np.array([dpos, dvel], dtype=float)

# =========================
# 2) Integrator & simulation
# =========================
def rk4_step(x, dt, dyn=f_vdp, **dyn_params):
    k1 = dyn(x, **dyn_params)
    k2 = dyn(x + 0.5*dt*k1, **dyn_params)
    k3 = dyn(x + 0.5*dt*k2, **dyn_params)
    k4 = dyn(x + dt*k3, **dyn_params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_traj(x0, N, dt, dyn=f_vdp, step=rk4_step, **dyn_params):
    X = np.zeros((N+1, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(N):
        X[k+1] = step(X[k], dt, dyn=dyn, **dyn_params)
    return X

# =========================
# 3) Build Koopman pairs
# =========================
def build_one_step_pairs(traj):
    X = traj[:-1].T   # (2, N)
    Xn = traj[1:].T   # (2, N)
    return X, Xn

def stack_pairs(pair_list):
    X_all     = np.hstack([p[0] for p in pair_list])
    Xnext_all = np.hstack([p[1] for p in pair_list])
    return X_all, Xnext_all

# =========================
# 4) Observables (monomials)
# =========================
def phi_from_X_vdp(X):
    """
    X: (2, N) rows [x; v]
    Phi rows:
      0: 1
      1: x
      2: x^2
      3: x^3
      4: v
      5: v^2
      6: x*v
      7: x^2*v   <-- key VdP nonlinearity term
    """
    x = X[0, :]
    v = X[1, :]
    ones = np.ones_like(x)
    Phi = np.vstack([ones, x, x**2, x**3, v, v**2, x*v, (x**2)*v])
    return Phi

# =========================
# 5) Standardization utils
# =========================
def standardize_fit(Phi):
    mu  = Phi.mean(axis=1, keepdims=True)
    std = Phi.std(axis=1, keepdims=True) + 1e-12
    return (Phi - mu)/std, mu, std

def standardize_apply(Phi, mu, std):
    return (Phi - mu)/std

def unstandardize(Phi_std, mu, std):
    return Phi_std*std + mu

# =========================
# 6) Fit Koopman (SVD–ridge)
# =========================
def fit_koopman_svd_ridge(Phi_std, Phi_next_std, lam=1e-5):
    """
    Tikhonov-regularized pseudoinverse via SVD:
      Phi^†_ridge = V diag(S/(S^2 + lam)) U^T
      K = Phi_next * Phi^†_ridge
    """
    U, S, Vt = np.linalg.svd(Phi_std, full_matrices=False)
    filt = S / (S**2 + lam)
    Phi_pinv = Vt.T @ np.diag(filt) @ U.T
    K = Phi_next_std @ Phi_pinv
    return K

# =========================
# 7) Rollout & decode
# =========================
def rollout_and_decode_vdp(K, Phi0_std, steps, mu_stats, std_stats):
    """
    Advance in standardized feature space, then unstandardize to read x,v rows.
    Returns arrays x_hat, v_hat of length steps+1.
    """
    m = Phi0_std.shape[0]
    Phi_std = Phi0_std.reshape(m, 1)
    xs, vs = [], []
    for k in range(steps+1):
        Phi = unstandardize(Phi_std, mu_stats, std_stats)
        x = Phi[1, 0]   # x row index = 1
        v = Phi[4, 0]   # v row index = 4
        xs.append(x); vs.append(v)
        if k < steps:
            Phi_std = K @ Phi_std
    return np.array(xs), np.array(vs)

# =========================
# 8) Train/test, eval
# =========================
def train_and_eval_vdp(X, Xnext, lam=1e-5, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)

    # Train
    X_tr, Xp_tr = X[:, :idx_split], Xnext[:, :idx_split]
    Phi_tr      = phi_from_X_vdp(X_tr)
    Phi_tr_p    = phi_from_X_vdp(Xp_tr)
    Phi_tr_std, mu_stats, std_stats = standardize_fit(Phi_tr)
    Phi_tr_p_std = standardize_apply(Phi_tr_p, mu_stats, std_stats)
    K = fit_koopman_svd_ridge(Phi_tr_std, Phi_tr_p_std, lam=lam)

    # Test
    X_te, Xp_te = X[:, idx_split:], Xnext[:, idx_split:]
    Phi_te      = phi_from_X_vdp(X_te)
    Phi_te_p    = phi_from_X_vdp(Xp_te)
    Phi_te_std  = standardize_apply(Phi_te,   mu_stats, std_stats)
    Phi_te_p_std= standardize_apply(Phi_te_p, mu_stats, std_stats)

    # One-step
    Phi_pred_std = K @ Phi_te_std
    Phi_pred     = unstandardize(Phi_pred_std, mu_stats, std_stats)
    x_true  = Phi_te_p[1, :];  v_true  = Phi_te_p[4, :]
    x_pred1 = Phi_pred[1, :];  v_pred1 = Phi_pred[4, :]

    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        "one_step_RMSE_x": rmse(x_pred1, x_true),
        "one_step_RMSE_v": rmse(v_pred1, v_true),
    }

    # Multi-step rollout from first test column
    Phi0_std = Phi_te_std[:, 0]
    x_hat, v_hat = rollout_and_decode_vdp(K, Phi0_std, steps=horizon_steps,
                                          mu_stats=mu_stats, std_stats=std_stats)
    x_true_ms = Phi_te[1, :horizon_steps+1]
    v_true_ms = Phi_te[4, :horizon_steps+1]
    metrics.update({
        f"multistep_RMSE_x(h{horizon_steps})": rmse(x_hat, x_true_ms),
        f"multistep_RMSE_v(h{horizon_steps})": rmse(v_hat, v_true_ms),
    })

    return K, metrics, (mu_stats, std_stats, idx_split)

# =========================
# 9) Plots (show)
# =========================
def show_figures_vdp(X, K, mu_stats, std_stats, dt=0.01, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)
    X_te  = X[:, idx_split:]
    Phi_te     = phi_from_X_vdp(X_te)
    Phi_te_std = standardize_apply(Phi_te, mu_stats, std_stats)
    Phi0_std   = Phi_te_std[:, 0]

    # Truth vs Koopman rollout
    x_hat, v_hat = rollout_and_decode_vdp(K, Phi0_std, steps=horizon_steps,
                                          mu_stats=mu_stats, std_stats=std_stats)
    x_true_ms = Phi_te[1, :horizon_steps+1]
    v_true_ms = Phi_te[4, :horizon_steps+1]
    t = np.arange(horizon_steps+1) * dt

    plt.figure()
    plt.plot(t, x_true_ms, label="x true")
    plt.plot(t, x_hat, "--", label="x Koopman")
    plt.xlabel("time (s)"); plt.ylabel("x"); plt.legend()
    plt.title("VdP position: true vs. Koopman (multi-step)")

    plt.figure()
    plt.plot(t, v_true_ms, label="v true")
    plt.plot(t, v_hat, "--", label="v Koopman")
    plt.xlabel("time (s)"); plt.ylabel("v"); plt.legend()
    plt.title("VdP velocity: true vs. Koopman (multi-step)")

    # Error growth
    H = min(horizon_steps, Phi_te.shape[1]-1)
    rmse_x, rmse_v = [], []
    for h in range(1, H+1):
        xh, vh = rollout_and_decode_vdp(K, Phi0_std, steps=h,
                                        mu_stats=mu_stats, std_stats=std_stats)
        x_true_h = Phi_te[1, :h+1]
        v_true_h = Phi_te[4, :h+1]
        rmse_x.append(np.sqrt(np.mean((xh - x_true_h)**2)))
        rmse_v.append(np.sqrt(np.mean((vh - v_true_h)**2)))

    horizons = np.arange(1, H+1)*dt
    plt.figure()
    plt.plot(horizons, rmse_x, label="RMSE x")
    plt.plot(horizons, rmse_v, label="RMSE v")
    plt.xlabel("prediction horizon (s)"); plt.ylabel("RMSE")
    plt.title("VdP: error growth over horizon")
    plt.legend()

    plt.show()

# =========================
# 10) Example usage
# =========================
if __name__ == "__main__":
    # Parameters & sim horizon
    mu = 1.0          # try 0.5, 1.0, 2.0
    dt, T = 0.01, 120.0
    N = int(T/dt)

    # Multiple trajectories (cover basin of the limit cycle)
    ics = [[-2.0, 0.0], [-1.0, 1.0], [0.5, 0.0], [2.0, -1.0], [0.0, 2.0]]
    pairs = []
    for x0 in ics:
        tr = simulate_traj(x0, N, dt, dyn=f_vdp, mu=mu)
        pairs.append(build_one_step_pairs(tr))
    X, Xnext = stack_pairs(pairs)

    print("Shapes:", X.shape, Xnext.shape)

    # Fit & evaluate
    K, metrics, (mu_stats, std_stats, idx_split) = train_and_eval_vdp(
        X, Xnext, lam=1e-5, horizon_steps=200
    )
    print("Koopman metrics (VdP):", metrics)

    # Plots
    show_figures_vdp(X, K, mu_stats, std_stats, dt=dt, horizon_steps=200)
