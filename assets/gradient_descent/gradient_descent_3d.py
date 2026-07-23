"""
Simulació visual del descens de gradient sobre una funció 3D.

Funció utilitzada (una "bowl" no simètrica, per fer-ho una mica més interessant
que un paraboloide perfecte):

    f(x, y) = x^2 + 2*y^2 + x*y - 3*x

Es calcula el gradient de manera analítica i s'aplica la regla d'actualització:

    theta_{t+1} = theta_t - eta * grad(f)(theta_t)

El resultat es guarda com una animació GIF que mostra:
  1. La superfície 3D de la funció.
  2. La trajectòria de la bola (paràmetres) baixant cap al mínim.
  3. Un panell inferior amb l'evolució del valor de la funció de cost.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessari per a projection='3d')


# ---------------------------------------------------------------------------
# 1. Definició de la funció de cost i del seu gradient
# ---------------------------------------------------------------------------
def f(x, y):
    """Funció de cost J(x, y)."""
    return x**2 + 2 * y**2 + x * y - 3 * x


def grad_f(x, y):
    """Gradient analític de f: (df/dx, df/dy)."""
    df_dx = 2 * x + y - 3
    df_dy = 4 * y + x
    return np.array([df_dx, df_dy])


# ---------------------------------------------------------------------------
# 2. Descens de gradient
# ---------------------------------------------------------------------------
def gradient_descent(start, lr=0.1, n_iter=60):
    """Retorna la trajectòria (llista de punts (x, y, f(x,y))) del descens de gradient."""
    theta = np.array(start, dtype=float)
    trajectory = [theta.copy()]

    for _ in range(n_iter):
        grad = grad_f(theta[0], theta[1])
        theta = theta - lr * grad
        trajectory.append(theta.copy())

    return np.array(trajectory)


# Punt inicial (lluny del mínim) i hiperparàmetres
start_point = (-3.5, 3.0)
learning_rate = 0.15
n_iterations = 50

trajectory = gradient_descent(start_point, lr=learning_rate, n_iter=n_iterations)
z_traj = f(trajectory[:, 0], trajectory[:, 1])


# ---------------------------------------------------------------------------
# 3. Preparar la superfície 3D
# ---------------------------------------------------------------------------
x_range = np.linspace(-5, 5, 120)
y_range = np.linspace(-5, 5, 120)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)


# ---------------------------------------------------------------------------
# 4. Figura amb dos subplots: superfície 3D + evolució del cost
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117")

ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.set_facecolor("#0f1117")

ax_cost = fig.add_subplot(1, 2, 2)
ax_cost.set_facecolor("#161925")

# --- Superfície 3D ---
ax3d.plot_surface(
    X, Y, Z, cmap=cm.viridis, alpha=0.55, linewidth=0, antialiased=True, zorder=1
)
ax3d.contour(X, Y, Z, levels=15, cmap=cm.viridis, offset=Z.min() - 5, alpha=0.4)

ax3d.set_xlabel("x", color="white", labelpad=8)
ax3d.set_ylabel("y", color="white", labelpad=8)
ax3d.set_zlabel("f(x, y)", color="white", labelpad=8)
ax3d.set_title("Descens de gradient sobre f(x, y)", color="white", fontsize=13, pad=15)
ax3d.tick_params(colors="#9aa0ac")
ax3d.xaxis.pane.set_facecolor("#0f1117")
ax3d.yaxis.pane.set_facecolor("#0f1117")
ax3d.zaxis.pane.set_facecolor("#0f1117")
ax3d.view_init(elev=32, azim=-60)

# Punt (bola) que es mourà, i traça acumulada
ball, = ax3d.plot([], [], [], "o", color="#ff6b6b", markersize=9, zorder=5)
path_line, = ax3d.plot([], [], [], "-", color="#ffd166", linewidth=2, zorder=4)

# --- Panell d'evolució del cost ---
ax_cost.set_title("Evolució de la funció de cost J", color="white", fontsize=13)
ax_cost.set_xlabel("Iteració", color="white")
ax_cost.set_ylabel("J(x, y)", color="white")
ax_cost.tick_params(colors="#9aa0ac")
for spine in ax_cost.spines.values():
    spine.set_color("#3a3f4b")
ax_cost.grid(alpha=0.2)

cost_line, = ax_cost.plot([], [], color="#4dd0e1", linewidth=2)
cost_point, = ax_cost.plot([], [], "o", color="#ff6b6b", markersize=7)

ax_cost.set_xlim(0, n_iterations)
ax_cost.set_ylim(z_traj.min() - 1, z_traj.max() + 1)

fig.tight_layout()


# ---------------------------------------------------------------------------
# 5. Funció d'animació
# ---------------------------------------------------------------------------
def update(frame):
    xs = trajectory[: frame + 1, 0]
    ys = trajectory[: frame + 1, 1]
    zs = z_traj[: frame + 1]

    # Bola: posició actual
    ball.set_data([xs[-1]], [ys[-1]])
    ball.set_3d_properties([zs[-1]])

    # Traça del camí recorregut
    path_line.set_data(xs, ys)
    path_line.set_3d_properties(zs)

    # Evolució del cost
    cost_line.set_data(np.arange(frame + 1), zs)
    cost_point.set_data([frame], [zs[-1]])

    return ball, path_line, cost_line, cost_point


anim = FuncAnimation(
    fig, update, frames=len(trajectory), interval=120, blit=False, repeat=True
)

# ---------------------------------------------------------------------------
# 6. Exportar com a GIF
# ---------------------------------------------------------------------------
output_path = "/mnt/user-data/outputs/descens_gradient_3d.gif"
anim.save(output_path, writer=PillowWriter(fps=10))

print(f"Animació guardada a: {output_path}")
print(f"Punt inicial: {start_point}")
print(f"Punt final aproximat: ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})")
print(f"Valor final de f: {z_traj[-1]:.4f}")
