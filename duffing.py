import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1) Duffing dynamics
# ----------------------------
def f_duffing(x, delta=0.2, alpha=-1.0, beta=1.0):
    """
    Damped, unforced Duffing oscillator.
    x = [position, velocity] = [x, v]
    """
    pos, vel = x
    dpos = vel
    dvel = -delta*vel - alpha*pos - beta*(pos**3)
    return np.array([dpos, dvel], dtype=float)

# ----------------------------
# 2) One RK4 step
# ----------------------------
def rk4_step(x, dt, dyn=f_duffing, **dyn_params):
    k1 = dyn(x, **dyn_params)
    k2 = dyn(x + 0.5*dt*k1, **dyn_params)
    k3 = dyn(x + 0.5*dt*k2, **dyn_params)
    k4 = dyn(x + dt*k3, **dyn_params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ----------------------------
# 3) Simulate one trajectory
# ----------------------------
def simulate_traj(x0, N, dt, dyn=f_duffing, step=rk4_step, **dyn_params):
    X = np.zeros((N+1, 2), dtype=float)
    X[0] = np.array(x0, dtype=float)
    for k in range(N):
        X[k+1] = step(X[k], dt, dyn=dyn, **dyn_params)
    return X

# ----------------------------
# 4) Build one-step pairs
# ----------------------------
def build_one_step_pairs(traj):
    X = traj[:-1].T   # (2, N)
    Xn = traj[1:].T   # (2, N)
    return X, Xn

# ----------------------------
# 5) Observables for Duffing
# ----------------------------
def phi_from_X_duffing(X):
    """
    X: (2, N) rows [x; v]
    Phi rows: [1, x, x^2, x^3, v, x*v]
    """
    x = X[0, :]
    v = X[1, :]
    ones = np.ones_like(x)
    Phi = np.vstack([ones, x, x**2, x**3, v, x*v])
    return Phi

# ----------------------------
# 6) Fit Koopman (Gram + ridge)
# ----------------------------
def fit_koopman(Phi, Phi_next, lam=1e-6):
    G = Phi @ Phi.T
    A = Phi_next @ Phi.T
    m = Phi.shape[0]
    K = A @ np.linalg.inv(G + lam*np.eye(m))
    return K

# ----------------------------
# 7) Rollout in lifted space and decode
# ----------------------------
def rollout_and_decode_duffing(K, Phi0, steps):
    """
    Decode x from the 'x' row (= row 1) and v from 'v' row (= row 4) of Phi.
    Row order: [1, x, x^2, x^3, v, x*v] -> indices [0,1,2,3,4,5]
    """
    m = Phi0.shape[0]
    Phi_hat = Phi0.reshape(m, 1)
    xs, vs = [], []
    for k in range(steps+1):
        x = Phi_hat[1, 0]    # x
        v = Phi_hat[4, 0]    # v
        xs.append(x); vs.append(v)
        if k < steps:
            Phi_hat = K @ Phi_hat
    return np.array(xs), np.array(vs)

# ----------------------------
# 8) Train/test, eval, plots
# ----------------------------
def train_and_eval_duffing(X, Xnext, lam=1e-6, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)

    # Train
    X_tr, Xp_tr = X[:, :idx_split], Xnext[:, :idx_split]
    Phi_tr      = phi_from_X_duffing(X_tr)
    Phi_tr_p    = phi_from_X_duffing(Xp_tr)
    K = fit_koopman(Phi_tr, Phi_tr_p, lam=lam)

    # Test
    X_te, Xp_te = X[:, idx_split:], Xnext[:, idx_split:]
    Phi_te      = phi_from_X_duffing(X_te)
    Phi_te_p    = phi_from_X_duffing(Xp_te)

    # One-step
    Phi_pred = K @ Phi_te
    x_true   = Phi_te_p[1, :]
    v_true   = Phi_te_p[4, :]
    x_pred1  = Phi_pred[1, :]
    v_pred1  = Phi_pred[4, :]

    rmse = lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        "one_step_RMSE_x": rmse(x_pred1, x_true),
        "one_step_RMSE_v": rmse(v_pred1, v_true),
    }

    # Multi-step rollout from first test column
    Phi0 = Phi_te[:, 0]
    x_hat, v_hat = rollout_and_decode_duffing(K, Phi0, steps=horizon_steps)
    x_true_ms = Phi_te[1, :horizon_steps+1]
    v_true_ms = Phi_te[4, :horizon_steps+1]
    metrics.update({
        f"multistep_RMSE_x(h{horizon_steps})": rmse(x_hat, x_true_ms),
        f"multistep_RMSE_v(h{horizon_steps})": rmse(v_hat, v_true_ms),
    })

    return K, metrics

def show_figures_duffing(X, K, dt=0.01, horizon_steps=200):
    N = X.shape[1]
    idx_split = int(0.7 * N)
    X_te  = X[:, idx_split:]
    Phi_te = phi_from_X_duffing(X_te)
    Phi0   = Phi_te[:, 0]

    # Truth vs Koopman rollout
    x_hat, v_hat = rollout_and_decode_duffing(K, Phi0, steps=horizon_steps)
    x_true_ms = Phi_te[1, :horizon_steps+1]
    v_true_ms = Phi_te[4, :horizon_steps+1]
    t = np.arange(horizon_steps+1) * dt

    plt.figure()
    plt.plot(t, x_true_ms, label="x true")
    plt.plot(t, x_hat, "--", label="x Koopman")
    plt.xlabel("time (s)"); plt.ylabel("x"); plt.legend()
    plt.title("Duffing position: true vs. Koopman (multi-step)")

    plt.figure()
    plt.plot(t, v_true_ms, label="v true")
    plt.plot(t, v_hat, "--", label="v Koopman")
    plt.xlabel("time (s)"); plt.ylabel("v"); plt.legend()
    plt.title("Duffing velocity: true vs. Koopman (multi-step)")

    # Error growth
    H = min(horizon_steps, Phi_te.shape[1]-1)
    rmse_x, rmse_v = [], []
    for h in range(1, H+1):
        xh, vh = rollout_and_decode_duffing(K, Phi0, steps=h)
        x_true_h = Phi_te[1, :h+1]
        v_true_h = Phi_te[4, :h+1]
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

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Duffing parameters (double well): delta>0, beta>0, alpha<0
    delta, alpha, beta = 0.2, -1.0, 1.0
    dt, T = 0.01, 120.0
    N = int(T/dt)

    # Initial condition (try a few around the wells)
    x0 = [0.5, 0.0]  # [x, v]

    # Simulate
    traj = simulate_traj(x0, N, dt, dyn=f_duffing, delta=delta, alpha=alpha, beta=beta)

    # Build pairs
    X, Xnext = build_one_step_pairs(traj)
    print("Shapes:", X.shape, Xnext.shape)

    # Fit & evaluate
    K, metrics = train_and_eval_duffing(X, Xnext, lam=1e-6, horizon_steps=200)
    print("Koopman metrics (Duffing):", metrics)

    # Show figures on screen
    show_figures_duffing(X, K, dt=dt, horizon_steps=200)
