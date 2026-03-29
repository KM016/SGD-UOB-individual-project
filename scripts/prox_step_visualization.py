import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Parameters 
alpha = 0.18
lam = 0.55
x0 = np.array([1.7, 1.05])
grad = np.array([x0[0], 3 * x0[1]]) # ∇F(x0) for F = 0.5*(x1^2 + 3*x2^2)
z = x0 - alpha * grad   # gradient step
xnew = np.sign(z) * np.maximum(np.abs(z) - alpha * lam, 0.) # soft threshold (prox)

# Contour Grid 
x1 = np.linspace(1.0, 2.0, 300)
x2 = np.linspace(0.2, 1.3, 300)
X1, X2 = np.meshgrid(x1, x2)
Z = 0.5 * (X1**2 + 3 * X2**2)

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))

ax.contour(X1, X2, Z, levels=10, colors='gray', linewidths=0.8, alpha=0.6)

# Arrows: x0 -> z (gradient step), z -> xnew (proximal step)
ax.annotate("", xy=z, xytext=x0, arrowprops=dict(arrowstyle="-|>", color="steelblue", lw=1.8))
ax.annotate("", xy=xnew, xytext=z, arrowprops=dict(arrowstyle="-|>", color="tomato", lw=1.8))

# Points
ax.scatter(*x0, s=60, color="black", zorder=5)
ax.scatter(*z, s=60, color="steelblue", zorder=5)
ax.scatter(*xnew, s=60, color="tomato", zorder=5)

# Labels
ax.text(x0[0] + 0.03, x0[1] + 0.03, r"$x^{(t)}$", fontsize=12)
ax.text(z[0]  + 0.03, z[1]  - 0.06, r"$z$", fontsize=12, color="steelblue")
ax.text(xnew[0] - 0.07, xnew[1] + 0.03, r"$x^{(t+1)}$", fontsize=12, color="tomato")

# Legend
legend_elements = [
    Line2D([0], [0], color="steelblue", lw=1.8, label=r"Gradient step: $z = x - \alpha\nabla F(x)$"),
    Line2D([0], [0], color="tomato", lw=1.8, label=r"Proximal step: $x^{(t+1)} = \mathrm{prox}_{\alpha\lambda}(z)$"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_facecolor('lemonchiffon')
ax.set_title("Proximal gradient step")

plt.tight_layout()
plt.savefig("prox_step_visualisation.png", dpi=150, facecolor='lightsteelblue')
print('done')