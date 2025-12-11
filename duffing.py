import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Duffing dynamics (CT)
# =========================
def f_duffing(x, delta=0.2, alpha=-1.0, beta=1.0):
    """
    Damped, unforced Duffing oscillator.
    state x = [position, velocity] = [x, v]
    x'' + delta x' + alpha x + beta x^3 = 0
    """
    pos, vel = x
    dpos = vel
    dvel = -delta*vel - alpha*pos - beta*(pos**3)
    return np.array([dpos, dvel], dtype=float)


# RK4 step
def rk4_step(x, dt, dyn=f_duffing, **dyn_params):
    k1 = dyn(x, **dyn_params)
    k2 = dyn(x + 0.5*dt*k1, **dyn_params)
    k3 = dyn(x + 0.5*dt*k2, **dyn_params)
    k4 = dyn(x + dt*k3, **dyn_params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# simulate one trajectory
def simulate_traj(x0, N, dt, dyn=f_duffing, step=rk4_step, **dyn_params):
    X = np.zeros((N+1, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(N):
        X[k+1] = step(X[k], dt, dyn=dyn, **dyn_params)
    return X


# =========================
# 2) Build Koopman pairs
# =========================
def build_one_step_pairs(traj):
    X = traj[:-1].T   # (2, N)
    Xn = traj[1:].T   # (2, N)
    return X, Xn

def stack_pairs(pair_list):
    """Pair list is [(X1,X1n), (X2,X2n), ...]; returns big (2, Ntot)."""
    X_all    = np.hstack([p[0] for p in pair_list])
    Xnext_all= np.hstack([p[1] for p in pair_list])
    return X_all, Xnext_all


# =========================
# 3) Observables (richer polynomial dictionary)
# =========================
def phi_from_X_duffing_rich(X, p_max=5):
    """
    Rich monomial dictionary for Duffing:
    - Includes all monomials x^i v^j with i + j <= p_max
    - p_max = 5 gives 21 features (including constant 1)

    X: (2, N) with rows [x; v]
    Returns: Phi: (n_features, N)
    Feature ordering:
      row 0: 1
      row 1: x
      row 2: v
      then increasing total degree (x^2, x v, v^2, ...).
    """
    x = X[0, :]
    v = X[1, :]

    ones = np.ones_like(x)
    features = [ones, x, v]  # enforce ordering [1, x, v]

    # all monomials x^i v^j with total degree >= 2
    for total_deg in range(2, p_max + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            features.append((x ** i) * (v ** j))

    Phi = np.vstack(features)
    return Phi

# =========================
# 4) Standardization utils
# =========================
def standardize_fit(Phi):
    """
    Standardize all non-constant rows of Phi:
      - row 0 (constant 1) left unchanged (mu=0, std=1).
      - rows 1..m-1: subtract mean, divide by std.
    Returns Phi_std, mu, std.
    """
    m, N = Phi.shape
    mu  = np.zeros((m, 1))
    std = np.ones((m, 1))

    for i in range(1, m):
        mu[i, 0]  = Phi[i, :].mean()
        std[i, 0] = Phi[i, :].std() + 1e-12

    Phi_std = (Phi - mu) / std
    return Phi_std, mu, std


def standardize_apply(Phi, mu, std):
    return (Phi - mu) / std


def unstandardize(Phi_std, mu, std):
    return Phi_std * std + mu


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
# 6) Rollout & decode
# =========================
def rollout_and_decode_duffing(K, Phi0_std, steps, mu, std):
    """
    Advance in standardized feature space, then unstandardize to read x,v rows.
    Returns arrays x_hat, v_hat length steps+1.

    NOTE: For the rich dictionary, we assume:
      row 0: 1
      row 1: x
      row 2: v
    """
    m = Phi0_std.shape[0]
    Phi_std = Phi0_std.reshape(m, 1)
    xs, vs = [], []
    for k in range(steps+1):
        Phi = unstandardize(Phi_std, mu, std)  # back to raw to read x,v rows
        x = Phi[1, 0]   # x row index = 1
        v = Phi[2, 0]   # v row index = 2
        xs.append(x); vs.append(v)
        if k < steps:
            Phi_std = K @ Phi_std
    return np.array(xs), np.array(vs)


# =========================
# 7) Train/test, eval
# =========================
def train_and_eval_duffing(X, Xnext, lam=1e-4, horizon_steps=200, p_max=5):
    """
    Train Koopman operator for Duffing with rich polynomial dictionary.
    """
    N = X.shape[1]
    idx_split = int(0.7 * N)

    # Train
    X_tr, Xp_tr = X[:, :idx_split], Xnext[:, :idx_split]
    Phi_tr   = phi_from_X_duffing_rich(X_tr,  p_max=p_max)
    Phi_tr_p = phi_from_X_duffing_rich(Xp_tr, p_max=p_max)
    Phi_tr_std, mu, std = standardize_fit(Phi_tr)
    Phi_tr_p_std = standardize_apply(Phi_tr_p, mu, std)
    # --- SVD to inspect singular values ---
    U, S, Vt = np.linalg.svd(Phi_tr_std, full_matrices=False)
    print("\nSingular values of Phi_tr_std:")
    print(S)
    print("Minimum singular value:", S.min())
    K = fit_koopman_svd_ridge(Phi_tr_std, Phi_tr_p_std, lam=lam)

    # Test
    X_te, Xp_te = X[:, idx_split:], Xnext[:, idx_split:]
    Phi_te      = phi_from_X_duffing_rich(X_te,  p_max=p_max)
    Phi_te_p    = phi_from_X_duffing_rich(Xp_te, p_max=p_max)
    Phi_te_std  = standardize_apply(Phi_te,   mu, std)
    Phi_te_p_std= standardize_apply(Phi_te_p, mu, std)

    # One-step
    Phi_pred_std = K @ Phi_te_std
    Phi_pred     = unstandardize(Phi_pred_std, mu, std)
    # rows: [1, x, v, ...]
    x_true  = Phi_te_p[1, :];  v_true  = Phi_te_p[2, :]
    x_pred1 = Phi_pred[1, :];  v_pred1 = Phi_pred[2, :]

    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        "one_step_RMSE_x": rmse(x_pred1, x_true),
        "one_step_RMSE_v": rmse(v_pred1, v_true),
    }

    # Multi-step rollout from first test column
    Phi0_std = Phi_te_std[:, 0]
    x_hat, v_hat = rollout_and_decode_duffing(K, Phi0_std, steps=horizon_steps, mu=mu, std=std)
    x_true_ms = Phi_te[1, :horizon_steps+1]
    v_true_ms = Phi_te[2, :horizon_steps+1]
    metrics.update({
        f"multistep_RMSE_x(h{horizon_steps})": rmse(x_hat, x_true_ms),
        f"multistep_RMSE_v(h{horizon_steps})": rmse(v_hat, v_true_ms),
    })

    return K, metrics, (mu, std, idx_split)


# =========================
# 8) Koopman eigenvalue analysis
# =========================
def analyze_koopman_eigs(K):
    """
    Compute and visualize Koopman eigenvalues to study system behavior.
    Eigenvalues |λ| < 1 => stable modes
                 |λ| = 1 => neutral/periodic
                 |λ| > 1 => unstable modes
    """
    eigvals, eigvecs = np.linalg.eig(K)
    
    # Print basic summary
    print("\n=== Koopman Eigenvalue Analysis ===")
    print(f"Number of eigenvalues: {len(eigvals)}")
    print(f"Min |λ| = {np.min(np.abs(eigvals)):.4f},  Max |λ| = {np.max(np.abs(eigvals)):.4f}")
    
    # Count stable/unstable/neutral modes
    stable = np.sum(np.abs(eigvals) < 1.0)
    neutral = np.sum(np.isclose(np.abs(eigvals), 1.0, atol=1e-2))
    unstable = np.sum(np.abs(eigvals) > 1.0)
    
    print(f"Stable modes (|λ| < 1): {stable}")
    print(f"Neutral modes (|λ| ≈ 1): {neutral}")
    print(f"Unstable modes (|λ| > 1): {unstable}")
    
    # Plot in complex plane
    plt.figure(figsize=(6,6))
    plt.scatter(np.real(eigvals), np.imag(eigvals), c='r', label='Eigenvalues')
    
    # Draw unit circle for reference
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', label='|λ|=1')
    
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("Koopman Eigenvalues in Complex Plane")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    return eigvals


# =========================
# 9) Plots (show)
# =========================
def show_figures_duffing(
    X, K, mu, std,
    dt=0.01,
    horizon_steps=200,
    multi_traj=False,
    traj_list=None,
    p_max=5
):
    """
    Visualize Duffing Koopman predictions.

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
        If False, shows single test segment (original behavior).
    traj_list : list of arrays, optional
        Each trajectory array of shape (N_i+1, 2).
    p_max : int
        Maximum polynomial degree in dictionary.
    """

    # ========================
    # SINGLE-TRAJECTORY MODE
    # ========================
    if not multi_traj or traj_list is None:
        N = X.shape[1]
        idx_split = int(0.7 * N)
        X_te = X[:, idx_split:]
        Phi_te = phi_from_X_duffing_rich(X_te, p_max=p_max)
        Phi_te_std = standardize_apply(Phi_te, mu, std)
        Phi0_std = Phi_te_std[:, 0]

        # Truth vs Koopman rollout
        x_hat, v_hat = rollout_and_decode_duffing(K, Phi0_std, steps=horizon_steps, mu=mu, std=std)
        x_true_ms = Phi_te[1, :horizon_steps+1]
        v_true_ms = Phi_te[2, :horizon_steps+1]
        t = np.arange(horizon_steps+1) * dt

        plt.figure()
        plt.plot(t, x_true_ms, label="x true")
        plt.plot(t, x_hat, "--", label="x Koopman")
        plt.xlabel("time (s)"); plt.ylabel("x"); plt.legend()
        plt.title("Duffing position: true vs Koopman (multi-step)")

        plt.figure()
        plt.plot(t, v_true_ms, label="v true")
        plt.plot(t, v_hat, "--", label="v Koopman")
        plt.xlabel("time (s)"); plt.ylabel("v"); plt.legend()
        plt.title("Duffing velocity: true vs Koopman (multi-step)")

        # Phase portrait (x–v plane)
        plt.figure()
        plt.plot(x_true_ms, v_true_ms, label="True trajectory", linewidth=2)
        plt.plot(x_hat, v_hat, '--', label="Koopman trajectory", linewidth=2)
        plt.xlabel("x (position)")
        plt.ylabel("v (velocity)")
        plt.title("Duffing oscillator: phase portrait (x–v plane)")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Error growth
        H = min(horizon_steps, Phi_te.shape[1]-1)
        rmse_x, rmse_v = [], []
        for h in range(1, H+1):
            xh, vh = rollout_and_decode_duffing(K, Phi0_std, steps=h, mu=mu, std=std)
            x_true_h = Phi_te[1, :h+1]
            v_true_h = Phi_te[2, :h+1]
            rmse_x.append(np.sqrt(np.mean((xh - x_true_h)**2)))
            rmse_v.append(np.sqrt(np.mean((vh - v_true_h)**2)))

        horizons = np.arange(1, H+1)*dt
        plt.figure()
        plt.plot(horizons, rmse_x, label="RMSE x")
        plt.plot(horizons, rmse_v, label="RMSE v")
        plt.xlabel("prediction horizon (s)"); plt.ylabel("RMSE")
        plt.title("Duffing: error growth over horizon")
        plt.legend()

        plt.show()
        return  # early exit

    # ========================
    # MULTI-TRAJECTORY MODE
    # ========================
    Phi_list = []
    for tr in traj_list:
        Xi = tr.T
        Phi_i = phi_from_X_duffing_rich(Xi, p_max=p_max)
        Phi_i_std = standardize_apply(Phi_i, mu, std)
        Phi_list.append((Phi_i, Phi_i_std))

    # --- 1) x(t) and v(t) for all trajectories
    plt.figure()
    plt.title("Duffing position: true vs Koopman (multi-step, all trajectories)")
    plt.xlabel("time (s)"); plt.ylabel("x")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        x_hat, _ = rollout_and_decode_duffing(K, Phi0_std, steps=H_i, mu=mu, std=std)
        x_true = Phi_i[1, :H_i+1]
        t = np.arange(H_i+1)*dt
        plt.plot(t, x_true, label=f"True traj {i+1}" if i<5 else None)
        plt.plot(t, x_hat, "--", label=f"Koopman traj {i+1}" if i<5 else None)
    plt.legend(ncol=2, fontsize=9); plt.grid(True, linestyle='--', alpha=0.5)

    plt.figure()
    plt.title("Duffing velocity: true vs Koopman (multi-step, all trajectories)")
    plt.xlabel("time (s)"); plt.ylabel("v")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        _, v_hat = rollout_and_decode_duffing(K, Phi0_std, steps=H_i, mu=mu, std=std)
        v_true = Phi_i[2, :H_i+1]
        t = np.arange(H_i+1)*dt
        plt.plot(t, v_true, label=f"True traj {i+1}" if i<5 else None)
        plt.plot(t, v_hat, "--", label=f"Koopman traj {i+1}" if i<5 else None)
    plt.legend(ncol=2, fontsize=9); plt.grid(True, linestyle='--', alpha=0.5)

    # --- 2) Phase portrait
    plt.figure()
    plt.title("Duffing phase portrait (x–v): true vs Koopman (all trajectories)")
    plt.xlabel("x"); plt.ylabel("v")
    for i, (Phi_i, Phi_i_std) in enumerate(Phi_list):
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        x_hat, v_hat = rollout_and_decode_duffing(K, Phi0_std, steps=H_i, mu=mu, std=std)
        x_true = Phi_i[1, :H_i+1]
        v_true = Phi_i[2, :H_i+1]
        plt.plot(x_true, v_true, label=f"True traj {i+1}" if i<5 else None)
        plt.plot(x_hat, v_hat, "--", label=f"Koopman traj {i+1}" if i<5 else None)
    plt.axis('equal'); plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(ncol=2, fontsize=9)

    # --- 3) Error growth (averaged)
    rmse_x_curves, rmse_v_curves = [], []
    for (Phi_i, Phi_i_std) in Phi_list:
        T_i = Phi_i.shape[1]
        H_i = min(horizon_steps, T_i-1)
        Phi0_std = Phi_i_std[:, 0]
        rx, rv = [], []
        for h in range(1, H_i+1):
            xh, vh = rollout_and_decode_duffing(K, Phi0_std, steps=h, mu=mu, std=std)
            x_true_h = Phi_i[1, :h+1]
            v_true_h = Phi_i[2, :h+1]
            rx.append(np.sqrt(np.mean((xh - x_true_h)**2)))
            rv.append(np.sqrt(np.mean((vh - v_true_h)**2)))
        rmse_x_curves.append(np.array(rx))
        rmse_v_curves.append(np.array(rv))

    H_min = min(len(r) for r in rmse_x_curves)
    horizons = np.arange(1, H_min+1) * dt
    plt.figure()
    for r in rmse_x_curves:
        plt.plot(horizons, r[:H_min], alpha=0.4)
    for r in rmse_v_curves:
        plt.plot(horizons, r[:H_min], alpha=0.25)
    plt.plot(horizons, np.mean([r[:H_min] for r in rmse_x_curves], axis=0), 'k', linewidth=2, label="Mean RMSE x")
    plt.plot(horizons, np.mean([r[:H_min] for r in rmse_v_curves], axis=0), 'r', linewidth=2, label="Mean RMSE v")
    plt.xlabel("prediction horizon (s)"); plt.ylabel("RMSE")
    plt.title("Duffing: error growth over horizon (mean across trajectories)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


# =========================
# 10) Example usage
# =========================
if __name__ == "__main__":
    # Duffing parameters (double well)
    delta, alpha, beta = 0.2, -1.0, 1.0
    dt, T = 0.01, 120.0
    N = int(T / dt)
    p_max = 15   # polynomial degree
    lam   = 0.01 * (0.016038318250232386 ** 2) #1e-4

    # ---- Option A: single trajectory (quick test)
    """
    x0 = [0.5, 0.0]
    traj = simulate_traj(x0, N, dt, dyn=f_duffing,
                         delta=delta, alpha=alpha, beta=beta)
    X, Xnext = build_one_step_pairs(traj)
    traj_list = [traj]  # keep for plotting if desired
    """

    # ---- Option B: multiple trajectories (recommended)
    ics = [[-0.7, 0.0], [-0.3, 0.1], [0.3, 0.0],
            [0.8, -0.05], [1.0, 0.2]]   #[0.0, 0.2]
    
    """
    # Left-well ICs (x ≈ -1)
    icl = [
        [-1.2, 0.1],
        [-1.0, 0.05],
        [-0.8, -0.05],
    ]

    # Right-well ICs (x ≈ +1)
    icr = [
        [0.8, 0.05],
        [1.0, -0.05],
        [1.2, -0.1],
    ]

    # Center-region ICs (near unstable saddle, nonzero velocity)
    icc = [
        [0.0, 0.2],   # the problematic IC
        [0.15,  0.15],
        [-0.15, 0.15],
    ]

    # Near-saddle ICs with zero velocity (v = 0)
    icv = [
        [-0.4, 0.0],
        [-0.2, 0.0],
        [ 0.4, 0.0],
    ]

    # Final list of initial conditions
    ics = icl + icr + icc + icv

    print("Number of ICs:", len(ics))
    print("IC list:", ics)
    """

    traj_list = []
    pairs = []
    for x0 in ics:
        tr = simulate_traj(x0, N, dt, dyn=f_duffing,
                           delta=delta, alpha=alpha, beta=beta)
        traj_list.append(tr)
        pairs.append(build_one_step_pairs(tr))

    # Stack all (current,next) pairs across trajectories
    X, Xnext = stack_pairs(pairs)
    print("Shapes:", X.shape, Xnext.shape)

    # Fit & evaluate Koopman operator
    K, metrics, (mu, std, idx_split) = train_and_eval_duffing(
        X, Xnext, lam=lam, horizon_steps=200, p_max=p_max
    )
    print("Koopman metrics (Duffing, rich dict):", metrics)

    # Analyze Koopman eigenvalues
    eigvals = analyze_koopman_eigs(K)

    # ---- Option 1: single test segment (quick check)
    # show_figures_duffing(X, K, mu, std, dt=dt, horizon_steps=200, p_max=p_max)

    # ---- Option 2: multi-trajectory evaluation (recommended)
    show_figures_duffing(X, K, mu, std,
                         dt=dt,
                         horizon_steps=1000,
                         multi_traj=True,  # True / False
                         traj_list=traj_list,
                         p_max=p_max)
