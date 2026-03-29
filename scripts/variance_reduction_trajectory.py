import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# Problem Setup 
H = np.diag([3.0, 1.0])
n = 40
sigma = 1.2

noise_vecs = np.random.randn(n, 2) * sigma
noise_vecs -= noise_vecs.mean(0)

def grad_fi(x, i):
    return H @ x + noise_vecs[i]

def full_grad(x):
    return H @ x

# Algorithms
def run_sgd(x0, alpha, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    traj = [x.copy()]
    for _ in range(n_steps):
        i = rng.integers(n)
        x = x - alpha * grad_fi(x, i)
        traj.append(x.copy())
    return np.array(traj)

def run_svrg(x0, alpha, m, n_epochs, seed=0):
    rng = np.random.default_rng(seed)
    x_tilde = x0.copy()
    traj  = [x0.copy()]
    snaps = [x0.copy()]
    for _ in range(n_epochs):
        g_tilde = full_grad(x_tilde)
        x = x_tilde.copy()
        for _ in range(m):
            i = rng.integers(n)
            g = grad_fi(x, i) - grad_fi(x_tilde, i) + g_tilde
            x = x - alpha * g
            traj.append(x.copy())
        x_tilde = x.copy()
        snaps.append(x_tilde.copy())
    return np.array(traj), np.array(snaps)

# Run
x0 = np.array([1.55, 1.10])
alpha_sgd = 0.10
alpha_svrg = 0.08
m_inner = 20
n_epochs = 14
n_sgd = m_inner * n_epochs

sgd_traj = run_sgd(x0, alpha_sgd, n_sgd, seed=5)
svrg_traj, snap_pts = run_svrg(x0, alpha_svrg, m_inner, n_epochs, seed=5)

# Contour Grid
x1g = np.linspace(-0.55, 1.75, 400)
x2g = np.linspace(-0.65, 1.35, 400)
X1, X2 = np.meshgrid(x1g, x2g)
Fv = 0.5 * (3*X1**2 + X2**2)

# Plotting 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, traj, title, line_color in [
    (ax1, sgd_traj, 'SGD (constant $\\alpha$)', 'steelblue'),
    (ax2, svrg_traj, 'SVRG (constant $\\alpha$)', 'tomato')]:

    ax.contour(X1, X2, Fv, levels=10, colors='gray', linewidths=0.8, alpha=0.5)

    # Trajectory
    ax.plot(traj[:, 0], traj[:, 1], lw=0.9, alpha=0.7, color=line_color, label='Trajectory')

    # Start / End / Optimum
    ax.scatter(*traj[0], s=60, color='black', zorder=5, label=r'$x_0$')
    ax.scatter(*traj[-1], s=60, color=line_color, zorder=5, ec='black', lw=0.8, label='Final iterate')
    ax.scatter(0, 0, s=120, color='gold', marker='*', zorder=6, ec='black', lw=0.5, label=r'$x^*$')

    ax.set_xlim(-0.52, 1.72)
    ax.set_ylim(-0.62, 1.32)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_facecolor('lemonchiffon')
    ax.set_title(title)

    ax.legend(frameon=True)

# SVRG: mark epoch snapshots (with label only once)
for i, pt in enumerate(snap_pts):
    ax2.scatter(*pt, s=40, color='teal', marker='D', zorder=7, ec='black', lw=0.5, label='Snapshot' if i == 0 else None)

ax2.legend(frameon=True)

plt.suptitle('SGD vs SVRG Trajectories', fontsize=13)
plt.tight_layout()
plt.savefig('variance_reduction_trajectory.png', dpi=150, facecolor='lightsteelblue')
print("Done")