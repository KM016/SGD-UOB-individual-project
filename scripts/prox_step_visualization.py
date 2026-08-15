import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_figure():
    alpha = 0.18
    lam = 0.55
    x0 = np.array([1.7, 1.05])
    gradient = np.array([x0[0], 3 * x0[1]])
    gradient_step = x0 - alpha * gradient
    prox_step = np.sign(gradient_step) * np.maximum(np.abs(gradient_step) - alpha * lam, 0.0)

    x1 = np.linspace(1.0, 2.0, 300)
    x2 = np.linspace(0.2, 1.3, 300)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    objective = 0.5 * (x1_grid**2 + 3 * x2_grid**2)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.contour(x1_grid, x2_grid, objective, levels=10, colors="gray", linewidths=0.8, alpha=0.6)
    axis.annotate(
        "", xy=gradient_step, xytext=x0,
        arrowprops={"arrowstyle": "-|>", "color": "steelblue", "lw": 1.8},
    )
    axis.annotate(
        "", xy=prox_step, xytext=gradient_step,
        arrowprops={"arrowstyle": "-|>", "color": "tomato", "lw": 1.8},
    )

    axis.scatter(*x0, s=60, color="black", zorder=5)
    axis.scatter(*gradient_step, s=60, color="steelblue", zorder=5)
    axis.scatter(*prox_step, s=60, color="tomato", zorder=5)
    axis.text(x0[0] + 0.03, x0[1] + 0.03, r"$x^{(t)}$", fontsize=12)
    axis.text(gradient_step[0] + 0.03, gradient_step[1] - 0.06, r"$z$", fontsize=12, color="steelblue")
    axis.text(prox_step[0] - 0.07, prox_step[1] + 0.03, r"$x^{(t+1)}$", fontsize=12, color="tomato")

    legend = [
        Line2D([0], [0], color="steelblue", lw=1.8, label=r"Gradient step: $z=x-\alpha\nabla F(x)$"),
        Line2D(
            [0], [0], color="tomato", lw=1.8,
            label=r"Proximal step: $x^{(t+1)}=\mathrm{prox}_{\alpha\lambda}(z)$",
        ),
    ]
    axis.legend(handles=legend, fontsize=9, loc="upper right")
    axis.set_xlabel(r"$x_1$")
    axis.set_ylabel(r"$x_2$")
    axis.set_facecolor("lemonchiffon")
    axis.set_title("Proximal gradient step")
    figure.tight_layout()
    return figure


def parse_args():
    parser = argparse.ArgumentParser(description="Render the proximal-gradient step illustration.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "figures" / "prox_step_visualisation.png",
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
