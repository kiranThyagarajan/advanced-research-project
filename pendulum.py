import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Pendulum dynamics (CT)
# =========================
def f_pendulum(x, g=9.81, L=1.0, c=0.1):
    """
    Damped, unforced pendulum:
      state x = [theta, omega]
      theta' = omega
      omega' = -(g/L) * sin(theta) - c * omega
    """
    theta, omega = x
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - c*omega
    return np.array([dtheta, domega], dtype=float)


# RK4 step
def rk4_step(x, dt, dyn=f_pendulum, **dyn_params):
    k1 = dyn(x, **dyn_params)
    k2 = dyn(x + 0.5*dt*k1, **dyn_params)
    k3 = dyn(x + 0.5*dt*k2, **dyn_params)
    k4 = dyn(x + dt*k3, **dyn_params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# simulate one trajectory
def simulate_traj(x0, N, dt, dyn=f_pendulum, step=rk4_step, **dyn_params):
    X = np.zeros((N+1, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(N):
        X[k+1] = step(X[k], dt, dyn=dyn, **dyn_params)
    return X


# =========================
# 2) Build Koopman pairs
# =========================
def build_one_step_pairs(traj):
    X  = traj[:-1].T   # (2, N)
    Xn = traj[1:].T    # (2, N)
    return X, Xn

def stack_pairs(pair_list):
    """Pair list is [(X1,X1n), (X2,X2n), ...]; returns big (2, Ntot)."""
    X_all     = np.hstack([p[0] for p in pair_list])
    Xnext_all = np.hstack([p[1] for p in pair_list])
    return X_all, Xnext_all


# =========================
# 3) Observables (same dictionary as your original pendulum code)
# =========================
def phi_from_X_pendulum(X):
    """
    X: (2, N) with rows [theta; omega]

    Dictionary (17 features), in this exact order:

    0: 1
    1: sin(theta)
    2: cos(theta)
    3: omega
    4: omega**2
    5: sin(theta) * omega
    6: sin(2*theta)
    7: cos(2*theta)
    8: sin(3*theta)
    9: cos(3*theta)
    10: omega**3
    11: omega**4
    12: sin(theta) * omega**2
    13: cos(theta) * omega
    14: cos(theta) * omega**2
    15: sin(2*theta) * omega
    16: cos(2*theta) * omega
    """
    theta = X[0, :]
    omega = X[1, :]

    ones   = np.ones_like(theta)
    sinth  = np.sin(theta)
    costh  = np.cos(theta)
    sin2th = np.sin(2.0 * theta)
    cos2th = np.cos(2.0 * theta)
    sin3th = np.sin(3.0 * theta)
    cos3th = np.cos(3.0 * theta)

    features = [
        ones,                 # 0: 1
        sinth,                # 1: sin(theta)
        costh,                # 2: cos(theta)
        omega,                # 3: omega
        omega**2,             # 4: omega**2
        sinth * omega,        # 5: sin(theta)*omega
        sin2th,               # 6: sin(2*theta)
        cos2th,               # 7: cos(2*theta)
        sin3th,               # 8: sin(3*theta)
        cos3th,               # 9: cos(3*theta)
        omega**3,             # 10: omega**3
        omega**4,             # 11: omega**4
        sinth * omega**2,     # 12: sin(theta)*omega**2
        costh * omega,        # 13: cos(theta)*omega
        costh * omega**2,     # 14: cos(theta)*omega**2
        sin2th * omega,       # 15: sin(2*theta)*omega
        cos2th * omega,       # 16: cos(2*theta)*omega
    ]

    return np.vstack(features)
"""
def phi_from_X_pendulum(X):
    
    theta = X[0, :]
    omega = X[1, :]
    ones  = np.ones_like(theta)
    sinth = np.sin(theta)
    costh = np.cos(theta)
    Phi = np.vstack([ones, sinth, costh, omega, omega**2, sinth*omega])
    return Phi
"""

# =========================
# 4) Standardization utils
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
# 5) Fit Koopman (SVD–ridge)
# =========================
def fit_koopman_svd_ridge(Phi_std, Phi_next_std, lam=1e-5):
    """
    Compute Koopman via SVD-based Tikhonov:
      Phi^†_ridge = V diag(S/(S^2 + lam)) U^T
      K = Phi_next * Phi^†_ridge
    """
    U, S, Vt = np.linalg.svd(Phi_std, full_matrices=False)
    filt = S / (S**2 + lam)
    Phi_pinv = Vt.T @ np.diag(filt) @ U.T
    K = Phi_next_std @ Phi_pinv
    return K


# =========================
# 6) Rollout & decode (pendulum)
# =========================
def rollout_and_decode_pendulum(K, Phi0_std, steps, mu, std):
    """
    Advance in standardized feature space, then unstandardize to read theta, omega.
    Dictionary rows: [1, sin(theta), cos(theta), omega, omega^2, sin(theta)*omega].
    Returns arrays theta_hat, omega_hat of length steps+1.
    """
    m = Phi0_std.shape[0]
    Phi_std = Phi0_std.reshape(m, 1)
    thetas, omegas = [], []
    for k in range(steps+1):
        Phi = unstandardize(Phi_std, mu, std)  # back to raw features
        sinth = Phi[1, 0]   # sin(theta)
        costh = Phi[2, 0]   # cos(theta)
        omega = Phi[3, 0]   # omega
        theta = np.arctan2(sinth, costh)
        thetas.append(theta)
        omegas.append(omega)
        if k < steps:
            Phi_std = K @ Phi_std
    return np.array(thetas), np.array(omegas)


# =========================
# 7) Train/test, eval (pendulum)
# =========================
def train_and_eval_pendulum(X, Xnext, lam=1e-4, horizon_steps=200):
    """
    Train Koopman operator for damped pendulum with your 6D dictionary.
    """
    N = X.shape[1]
    idx_split = int(0.7 * N)

    # Train
    X_tr, Xp_tr = X[:, :idx_split], Xnext[:, :idx_split]
    Phi_tr   = phi_from_X_pendulum(X_tr)
    Phi_tr_p = phi_from_X_pendulum(Xp_tr)
    Phi_tr_std, mu, std = standardize_fit(Phi_tr)
    Phi_tr_p_std = standardize_apply(Phi_tr_p, mu, std)
    K = fit_koopman_svd_ridge(Phi_tr_std, Phi_tr_p_std, lam=lam)

    # Test
    X_te, Xp_te = X[:, idx_split:], Xnext[:, idx_split:]
    Phi_te      = phi_from_X_pendulum(X_te)
    Phi_te_p    = phi_from_X_pendulum(Xp_te)
    Phi_te_std  = standardize_apply(Phi_te,   mu, std)
    Phi_te_p_std= standardize_apply(Phi_te_p, mu, std)

    # One-step predictions in feature space
    Phi_pred_std = K @ Phi_te_std
    Phi_pred     = unstandardize(Phi_pred_std, mu, std)

    # Decode theta, omega from features (rows 1–3)
    theta_true_1 = np.arctan2(Phi_te_p[1, :], Phi_te_p[2, :])
    omega_true_1 = Phi_te_p[3, :]

    theta_pred_1 = np.arctan2(Phi_pred[1, :], Phi_pred[2, :])
    omega_pred_1 = Phi_pred[3, :]

    rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        "one_step_RMSE_theta": rmse(theta_pred_1, theta_true_1),
        "one_step_RMSE_omega": rmse(omega_pred_1, omega_true_1),
    }

    # Multi-step rollout from first test column
    Phi0_std = Phi_te_std[:, 0]
    theta_hat, omega_hat = rollout_and_decode_pendulum(
        K, Phi0_std, steps=horizon_steps, mu=mu, std=std
    )
    theta_true_ms = np.arctan2(Phi_te[1, :horizon_steps+1], Phi_te[2, :horizon_steps+1])
    omega_true_ms = Phi_te[3, :horizon_steps+1]

    metrics.update({
        f"multistep_RMSE_theta(h{horizon_steps})": rmse(theta_hat, theta_true_ms),
        f"multistep_RMSE_omega(h{horizon_steps})": rmse(omega_hat, omega_true_ms),
    })

    return K, metrics, (mu, std, idx_split)


# =========================
# 8) Plots (pendulum)
# =========================
def show_figures_pendulum(
    X, K, mu, std,
    dt=0.01,
    horizon_steps=200,
    multi_traj=False,
    traj_list=None
):
    """
    Visualize pendulum Koopman predictions.

    Parameters
    ----------
    X : array, shape (2, N)
        Stacked state data.
    K : array
        Learned Koopman operator.
    mu, std : arrays
        Standardization statistics.
    dt : float
        Time step.
    horizon_steps : int
        Prediction horizon for rollouts.
    multi_traj : bool
        If True, overlay results for all trajectories (needs traj_list).
        If False, shows single test segment.
    traj_list : list of arrays, optional
        Each trajectory array of shape (N_i+1, 2).
    """

    # ========================
    # SINGLE-TRAJECTORY MODE
    # ========================
    if not multi_traj or traj_list is None:
        N = X.shape[1]
        idx_split = int(0.7 * N)
        X_te = X[:, idx_split:]
        Phi_te = phi_from_X_pendulum(X_te)
        Phi_te_std = standardize_apply(Phi_te, mu, std)
        Phi0_std = Phi_te_std[:, 0]

        # Truth vs Koopman rollout
        theta_hat, omega_hat = rollout_and_decode_pendulum(
            K, Phi0_std, steps=horizon_steps, mu=mu, std=std
        )
        theta_true_ms = np.arctan2(Phi_te[1, :horizon_steps+1], Phi_te[2, :horizon_steps+1])
        omega_true_ms = Phi_te[3, :horizon_steps+1]
        t = np.arange(horizon_steps+1) * dt

        # theta(t)
        plt.figure()
        plt.plot(t, theta_true_ms, label=r"$\theta$ true")
        plt.plot(t, theta_hat, "--", label=r"$\theta$ Koopman")
        plt.xlabel("time (s)"); plt.ylabel(r"$\theta$ (rad)"); plt.legend()
        plt.title("Pendulum angle: true vs Koopman (multi-step)")

        # omega(t)
        plt.figure()
        plt.plot(t, omega_true_ms, label=r"$\omega$ true")
        plt.plot(t, omega_hat, "--", label=r"$\omega$ Koopman")
        plt.xlabel("time (s)"); plt.ylabel(r"$\omega$ (rad/s)"); plt.legend()
        plt.title("Pendulum angular velocity: true vs Koopman (multi-step)")

        # Phase portrait (theta–omega)
        plt.figure()
        plt.plot(theta_true_ms, omega_true_ms, label="True trajectory", linewidth=2)
        plt.plot(theta_hat, omega_hat, "--", label="Koopman trajectory", linewidth=2)
        plt.xlabel(r"$\theta$ (rad)")
        plt.ylabel(r"$\omega$ (rad/s)")
        plt.title("Pendulum: phase portrait (theta–omega)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Error growth
        H = min(horizon_steps, Phi_te.shape[1]-1)
        rmse_theta, rmse_omega = [], []
        for h in range(1, H+1):
            th_h, om_h = rollout_and_decode_pendulum(
                K, Phi0_std, steps=h, mu=mu, std=std
            )
            th_true = np.arctan2(Phi_te[1, :h+1], Phi_te[2, :h+1])
            om_true = Phi_te[3, :h+1]
            rmse_theta.append(np.sqrt(np.mean((th_h - th_true)**2)))
            rmse_omega.append(np.sqrt(np.mean((om_h - om_true)**2)))

        horizons = np.arange(1, H+1)*dt
        plt.figure()
        plt.plot(horizons, rmse_theta, label=r"RMSE $\theta$")
        plt.plot(horizons, rmse_omega, label=r"RMSE $\omega$")
        plt.xlabel("prediction horizon (s)"); plt.ylabel("RMSE")
        plt.title("Pendulum: error growth over horizon")
        plt.legend()

        plt.show()
        return  # early exit

    # ========================
    # MULTI-TRAJECTORY MODE
    # ========================
    Phi_list = []
    for tr in traj_list:
        Xi = tr.T  # (2, Ni)
        Phi_i = phi_from_X_pendulum(Xi)
        Phi_i_std = standardize_apply(Phi_i, mu, std)
        Phi_list.append((Phi_i, Phi_i_std))

    # --- 1) theta(t) and omega(t) for all trajectories
    plt.figure()
    plt.title("Pendulum angle: true vs Koopman (multi-step, all trajectories)")
    plt.xlabel("time (s)"); plt.ylabel(r"$\theta$ (rad)")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        theta_hat, _ = rollout_and_decode_pendulum(
            K, Phi0_std, steps=H_i, mu=mu, std=std
        )
        theta_true = np.arctan2(Phi_i[1, :H_i+1], Phi_i[2, :H_i+1])
        t = np.arange(H_i+1)*dt
        plt.plot(t, theta_true, label=f"True traj {i+1}" if i < 5 else None)
        plt.plot(t, theta_hat, "--", label=f"Koopman traj {i+1}" if i < 5 else None)
    plt.legend(ncol=2, fontsize=9); plt.grid(True, linestyle='--', alpha=0.5)

    plt.figure()
    plt.title("Pendulum angular velocity: true vs Koopman (multi-step, all trajectories)")
    plt.xlabel("time (s)"); plt.ylabel(r"$\omega$ (rad/s)")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        _, omega_hat = rollout_and_decode_pendulum(
            K, Phi0_std, steps=H_i, mu=mu, std=std
        )
        omega_true = Phi_i[3, :H_i+1]
        t = np.arange(H_i+1)*dt
        plt.plot(t, omega_true, label=f"True traj {i+1}" if i < 5 else None)
        plt.plot(t, omega_hat, "--", label=f"Koopman traj {i+1}" if i < 5 else None)
    plt.legend(ncol=2, fontsize=5); plt.grid(True, linestyle='--', alpha=0.5)

    # --- 2) Phase portrait
    plt.figure()
    plt.title("Pendulum phase portrait (theta–omega): true vs Koopman (all trajectories)")
    plt.xlabel(r"$\theta$ (rad)"); plt.ylabel(r"$\omega$ (rad/s)")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        theta_hat, omega_hat = rollout_and_decode_pendulum(
            K, Phi0_std, steps=H_i, mu=mu, std=std
        )
        theta_true = np.arctan2(Phi_i[1, :H_i+1], Phi_i[2, :H_i+1])
        omega_true = Phi_i[3, :H_i+1]
        plt.plot(theta_true, omega_true, label=f"True traj {i+1}" if i < 5 else None)
        plt.plot(theta_hat, omega_hat, "--", label=f"Koopman traj {i+1}" if i < 5 else None)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncol=2, fontsize=9)

    # --- 3) Error growth (averaged)
    rmse_theta_curves, rmse_omega_curves = [], []
    for (Phi_i, Phi_i_std) in Phi_list:
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        rx, rv = [], []
        for h in range(1, H_i+1):
            theta_h, omega_h = rollout_and_decode_pendulum(
                K, Phi0_std, steps=h, mu=mu, std=std
            )
            theta_true_h = np.arctan2(Phi_i[1, :h+1], Phi_i[2, :h+1])
            omega_true_h = Phi_i[3, :h+1]
            rx.append(np.sqrt(np.mean((theta_h - theta_true_h)**2)))
            rv.append(np.sqrt(np.mean((omega_h - omega_true_h)**2)))
        rmse_theta_curves.append(np.array(rx))
        rmse_omega_curves.append(np.array(rv))

    H_min = min(len(r) for r in rmse_theta_curves)
    horizons = np.arange(1, H_min+1) * dt
    plt.figure()
    for r in rmse_theta_curves:
        plt.plot(horizons, r[:H_min], alpha=0.4)
    for r in rmse_omega_curves:
        plt.plot(horizons, r[:H_min], alpha=0.25)
    plt.plot(horizons, np.mean([r[:H_min] for r in rmse_theta_curves], axis=0),
             linewidth=2, label=r"Mean RMSE $\theta$")
    plt.plot(horizons, np.mean([r[:H_min] for r in rmse_omega_curves], axis=0),
             linewidth=2, label=r"Mean RMSE $\omega$")
    plt.xlabel("prediction horizon (s)"); plt.ylabel("RMSE")
    plt.title("Pendulum: error growth over horizon (mean across trajectories)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


# =========================
# 9) Example usage
# =========================
if __name__ == "__main__":
    # Pendulum physical parameters
    g, L, c = 9.81, 0.7, 0.08   # gravity, length, damping
    dt, T = 0.01, 300.0
    N = int(T / dt)
    lam = 1e-5          # ridge regularization

    # ---- Option A: single trajectory (quick test)
    """
    x0 = [0.9, 0.0]  # theta, omega
    traj = simulate_traj(x0, N, dt, dyn=f_pendulum, g=g, L=L, c=c)
    X, Xnext = build_one_step_pairs(traj)
    traj_list = [traj]
    """

    # ---- Option B: multiple trajectories (recommended)
    ics = [
        [0.9, 0.0],
        [-0.5, 0.2],
        [-1.0, 0.1],
        [0.5, 0.3],
        [1.0, -0.5],
        [1.5, -0.2],
        [2.0, -0.5],
        [-1.5, 0.4],
        [-2.0, 0.5],
        [-1.9, -0.3]
    ]

    traj_list = []
    pairs = []
    for x0 in ics:
        tr = simulate_traj(x0, N, dt, dyn=f_pendulum, g=g, L=L, c=c)
        traj_list.append(tr)
        pairs.append(build_one_step_pairs(tr))

    # Stack all (current,next) pairs across trajectories
    X, Xnext = stack_pairs(pairs)
    print("Shapes:", X.shape, Xnext.shape)

    # Fit & evaluate Koopman operator
    K, metrics, (mu, std, idx_split) = train_and_eval_pendulum(
        X, Xnext, lam=lam, horizon_steps=200
    )
    print("Koopman metrics (pendulum, 6D dict):", metrics)

    # ---- Option 1: single test segment
    # show_figures_pendulum(X, K, mu, std, dt=dt, horizon_steps=200)

    # ---- Option 2: multi-trajectory evaluation
    show_figures_pendulum(X, K, mu, std,
                          dt=dt,
                          horizon_steps=1000,
                          multi_traj=True,
                          traj_list=traj_list)
