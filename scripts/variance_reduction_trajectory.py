import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HESSIAN = np.diag([3.0, 1.0])
N_SAMPLES = 40

_noise_rng = np.random.RandomState(7)
NOISE_VECTORS = _noise_rng.randn(N_SAMPLES, 2) * 1.2
NOISE_VECTORS -= NOISE_VECTORS.mean(axis=0)


def component_gradient(x, index):
    return HESSIAN @ x + NOISE_VECTORS[index]


def full_gradient(x):
    return HESSIAN @ x


def run_sgd(x0, alpha, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    trajectory = [x.copy()]
    for _ in range(n_steps):
        index = rng.integers(N_SAMPLES)
        x = x - alpha * component_gradient(x, index)
        trajectory.append(x.copy())
    return np.asarray(trajectory)


def run_svrg(x0, alpha, m_inner, n_epochs, seed=0):
    rng = np.random.default_rng(seed)
    snapshot = x0.copy()
    trajectory = [x0.copy()]
    snapshots = [x0.copy()]
    for _ in range(n_epochs):
        snapshot_gradient = full_gradient(snapshot)
        x = snapshot.copy()
        for _ in range(m_inner):
            index = rng.integers(N_SAMPLES)
            estimator = (
                component_gradient(x, index)
                - component_gradient(snapshot, index)
                + snapshot_gradient
            )
            x = x - alpha * estimator
            trajectory.append(x.copy())
        snapshot = x.copy()
        snapshots.append(snapshot.copy())
    return np.asarray(trajectory), np.asarray(snapshots)


def build_figure():
    x0 = np.array([1.55, 1.10])
    m_inner = 20
    n_epochs = 14
    sgd_trajectory = run_sgd(x0, alpha=0.10, n_steps=m_inner * n_epochs, seed=5)
    svrg_trajectory, snapshot_points = run_svrg(x0, alpha=0.08, m_inner=m_inner, n_epochs=n_epochs, seed=5)

    x1_values = np.linspace(-0.55, 1.75, 400)
    x2_values = np.linspace(-0.65, 1.35, 400)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    objective = 0.5 * (3 * x1_grid**2 + x2_grid**2)

    figure, (sgd_axis, svrg_axis) = plt.subplots(1, 2, figsize=(12, 5))
    for axis, trajectory, title, color in [
        (sgd_axis, sgd_trajectory, r"SGD (constant $\alpha$)", "steelblue"),
        (svrg_axis, svrg_trajectory, r"SVRG (constant $\alpha$)", "tomato"),
    ]:
        axis.contour(x1_grid, x2_grid, objective, levels=10, colors="gray", linewidths=0.8, alpha=0.5)
        axis.plot(trajectory[:, 0], trajectory[:, 1], lw=0.9, alpha=0.7, color=color, label="Trajectory")
        axis.scatter(*trajectory[0], s=60, color="black", zorder=5, label=r"$x_0$")
        axis.scatter(*trajectory[-1], s=60, color=color, zorder=5, edgecolor="black", lw=0.8, label="Final iterate")
        axis.scatter(0, 0, s=120, color="gold", marker="*", zorder=6, edgecolor="black", lw=0.5, label=r"$x^*$")
        axis.set_xlim(-0.52, 1.72)
        axis.set_ylim(-0.62, 1.32)
        axis.set_xlabel(r"$x_1$")
        axis.set_ylabel(r"$x_2$")
        axis.set_facecolor("lemonchiffon")
        axis.set_title(title)
        axis.legend(frameon=True)

    for index, point in enumerate(snapshot_points):
        svrg_axis.scatter(
            *point, s=40, color="teal", marker="D", zorder=7,
            edgecolor="black", lw=0.5, label="Snapshot" if index == 0 else None,
        )
    svrg_axis.legend(frameon=True)
    figure.suptitle("SGD vs SVRG Trajectories", fontsize=13)
    figure.tight_layout()
    return figure


def parse_args():
    parser = argparse.ArgumentParser(description="Render the SGD/SVRG trajectory illustration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "figures" / "variance_reduction_trajectory.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output.expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    figure.savefig(output_path, dpi=150, facecolor="lightsteelblue")
    plt.close(figure)
    print(f"Saved {output_path.resolve()}")


if __name__ == "__main__":
    main()
