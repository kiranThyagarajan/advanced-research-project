import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Continuous-time dynamics
# ----------------------------
def f_pendulum(x, g=9.81, L=1.0, c=0.1):
    """Damped, unforced pendulum: x = [theta, omega]."""
    theta, omega = x
    dtheta = omega
    domega = -(g/L)*np.sin(theta) - c*omega
    return np.array([dtheta, domega], dtype=float)

# ----------------------------
# 2) One RK4 step
# ----------------------------
def rk4_step(x, dt, dyn=f_pendulum, **dyn_params):
    k1 = dyn(x, **dyn_params)
    k2 = dyn(x + 0.5*dt*k1, **dyn_params)
    k3 = dyn(x + 0.5*dt*k2, **dyn_params)
    k4 = dyn(x + dt*k3, **dyn_params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ----------------------------
# 3) Simulate one trajectory
# ----------------------------
def simulate_traj(x0, N, dt, dyn=f_pendulum, step=rk4_step, **dyn_params):
    """Returns array of shape (N+1, 2): states from k=0..N."""
    X = np.zeros((N+1, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(N):
        X[k+1] = step(X[k], dt, dyn=dyn, **dyn_params)
    return X

# ----------------------------
# 4) Build Koopman pairs
# ----------------------------
def build_one_step_pairs(traj):
    """Given traj (N+1,2), return X (2,N) and Xnext (2,N)."""
    X     = traj[:-1].T  # shape (2, N)
    Xnext = traj[1:].T   # shape (2, N)
    return X, Xnext

def stack_pairs(pair_list):
    """Stack multiple (X, Xnext) pairs column-wise."""
    X_all     = np.hstack([p[0] for p in pair_list])
    Xnext_all = np.hstack([p[1] for p in pair_list])
    return X_all, Xnext_all

# ----------------------------
# 5) Observables (hand-picked, physics-informed)
# ----------------------------
def phi_from_X(X):
    """
    X: (2, N) with rows [theta; omega]
    returns Phi: (6, N) with rows:
    [1, sin(theta), cos(theta), omega, omega**2, sin(theta)*omega]
    """
    theta = X[0, :]
    omega = X[1, :]
    ones  = np.ones_like(theta)
    sinth = np.sin(theta)
    costh = np.cos(theta)
    Phi = np.vstack([ones, sinth, costh, omega, omega**2, sinth*omega])
    return Phi

# ----------------------------
# 6) Fit Koopman (Gram + ridge)
# ----------------------------
def fit_koopman(Phi, Phi_next, lam=1e-6):
    """
    Phi, Phi_next: (m, N)
    returns K: (m, m)
    """
    G = Phi @ Phi.T                 # (m, m)
    A = Phi_next @ Phi.T            # (m, m)
    m = Phi.shape[0]
    K = A @ np.linalg.inv(G + lam*np.eye(m))
    return K

# ----------------------------
# 7) Rollout in lifted space and decode
# ----------------------------
def rollout_and_decode(K, Phi0, steps):
    """
    Phi0: (m,) initial observables
    returns arrays (theta_hat, omega_hat) length (steps+1)
    """
    m = Phi0.shape[0]
    Phi_hat = Phi0.reshape(m, 1)
    thetas, omegas = [], []
    for k in range(steps+1):
        sinth = Phi_hat[1, 0]
        costh = Phi_hat[2, 0]
        omega = Phi_hat[3, 0]
        theta = np.arctan2(sinth, costh)
        thetas.append(theta)
        omegas.append(omega)
        if k < steps:
            Phi_hat = K @ Phi_hat
    return np.array(thetas), np.array(omegas)

# ----------------------------
# 8) Train/test split, fit, eval
# ----------------------------
def train_and_eval(X, Xnext, lam=1e-6, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)

    # Training
    X_tr, Xp_tr = X[:, :idx_split], Xnext[:, :idx_split]
    Phi_tr      = phi_from_X(X_tr)
    Phi_tr_p    = phi_from_X(Xp_tr)
    K = fit_koopman(Phi_tr, Phi_tr_p, lam=lam)

    # Testing (for plotting/eval)
    X_te, Xp_te = X[:, idx_split:], Xnext[:, idx_split:]
    Phi_te      = phi_from_X(X_te)
    Phi_te_p    = phi_from_X(Xp_te)

    # One-step predictions
    Phi_te_pred = K @ Phi_te
    theta_true  = np.arctan2(Phi_te[1, :],      Phi_te[2, :])
    omega_true  = Phi_te[3, :]
    theta_pred1 = np.arctan2(Phi_te_pred[1, :], Phi_te_pred[2, :])
    omega_pred1 = Phi_te_pred[3, :]

    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        "one_step_RMSE_theta": rmse(theta_pred1, theta_true),
        "one_step_RMSE_omega": rmse(omega_pred1, omega_true),
    }

    # Multi-step rollout from first test sample
    Phi0 = Phi_te[:, 0]
    theta_hat, omega_hat = rollout_and_decode(K, Phi0, steps=horizon_steps)
    theta_true_ms = np.arctan2(Phi_te[1, :horizon_steps+1],  Phi_te[2, :horizon_steps+1])
    omega_true_ms = Phi_te[3, :horizon_steps+1]
    metrics.update({
        f"multistep_RMSE_theta(h{horizon_steps})": rmse(theta_hat, theta_true_ms),
        f"multistep_RMSE_omega(h{horizon_steps})": rmse(omega_hat, omega_true_ms),
    })

    return K, metrics, (theta_true_ms, omega_true_ms, theta_hat, omega_hat, idx_split)

# ----------------------------
# 9) Plots (show, don't save)
# ----------------------------
def show_figures(X, K, dt=0.01, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)
    X_te  = X[:, idx_split:]
    Phi_te = phi_from_X(X_te)
    Phi0   = Phi_te[:, 0]

    # Truth vs Koopman rollout
    theta_hat, omega_hat = rollout_and_decode(K, Phi0, steps=horizon_steps)
    theta_true_ms = np.arctan2(Phi_te[1, :horizon_steps+1], Phi_te[2, :horizon_steps+1])
    omega_true_ms = Phi_te[3, :horizon_steps+1]
    t = np.arange(horizon_steps+1) * dt

    plt.figure()
    plt.plot(t, theta_true_ms, label=r"$\theta$ true")
    plt.plot(t, theta_hat, "--", label=r"$\theta$ Koopman")
    plt.xlabel("time (s)"); plt.ylabel(r"$\theta$ (rad)"); plt.legend()
    plt.title("Pendulum angle: true vs. Koopman (multi-step)")

    plt.figure()
    plt.plot(t, omega_true_ms, label=r"$\omega$ true")
    plt.plot(t, omega_hat, "--", label=r"$\omega$ Koopman")
    plt.xlabel("time (s)"); plt.ylabel(r"$\omega$ (rad/s)"); plt.legend()
    plt.title("Pendulum angular velocity: true vs. Koopman (multi-step)")

    # Error growth vs horizon
    H = min(horizon_steps, Phi_te.shape[1]-1)
    rmse_theta, rmse_omega = [], []
    for h in range(1, H+1):
        th_h, om_h = rollout_and_decode(K, Phi0, steps=h)
        th_true = np.arctan2(Phi_te[1, :h+1], Phi_te[2, :h+1])
        om_true = Phi_te[3, :h+1]
        rmse_theta.append(np.sqrt(np.mean((th_h - th_true)**2)))
        rmse_omega.append(np.sqrt(np.mean((om_h - om_true)**2)))

    horizons = np.arange(1, H+1) * dt
    plt.figure()
    plt.plot(horizons, rmse_theta, label=r"RMSE $\theta$")
    plt.plot(horizons, rmse_omega, label=r"RMSE $\omega$")
    plt.xlabel("prediction horizon (s)")
    plt.ylabel("RMSE")
    plt.title("Error growth over prediction horizon")
    plt.legend()

    plt.show()

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Physical & numerical params
    g, L, c = 9.81, 0.7, 0.08   # gravity, length, damping (c=b/(mL^2))
    dt      = 0.01              # time step (s)
    T       = 120.0             # total time (s)
    N       = int(T/dt)         # number of steps

    # ---------- Option A: single trajectory ----------
    x0 = [0.9, 0.0]             # theta (rad), omega (rad/s)
    traj = simulate_traj(x0, N, dt, dyn=f_pendulum, g=g, L=L, c=c)
    X, Xnext = build_one_step_pairs(traj)

    # ---------- Option B: multiple trajectories ----------
    """
    ics = [
        [0.9, 0.0],
        [1.5, -0.2],
        [-1.0, 0.1],
        [0.5, 0.3],
    ]
    pairs = []
    for x0 in ics:
        traj = simulate_traj(x0, N, dt, dyn=f_pendulum, g=g, L=L, c=c)
        pairs.append(build_one_step_pairs(traj))
    X, Xnext = stack_pairs(pairs)
    """
    print("Shapes:", X.shape, Xnext.shape)

    # Fit Koopman and evaluate
    K, metrics, bundle = train_and_eval(X, Xnext, lam=1e-6, horizon_steps=200)
    print("Koopman metrics:", metrics)

    # Show figures on screen
    show_figures(X, K, dt=dt, horizon_steps=200)
