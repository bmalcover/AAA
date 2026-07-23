"""
Descens de gradient sobre una funció amb DIVERSOS MÍNIMS LOCALS.

Funció utilitzada:

    f(x, y) = sin(x) * cos(y) + 0.1*(x^2 + y^2)

El terme sin(x)*cos(y) crea una "ondulació" amb múltiples mínims i màxims
locals, mentre que el terme 0.1*(x^2+y^2) actua com una "conca" suau que
manté la funció acotada i assegura que hi hagi un mínim global identificable
dins la regió que visualitzem.

Aquest exemple permet observar un fet important: segons el punt d'inici,
el descens de gradient pot convergir a mínims LOCALS diferents, no
necessàriament al mínim GLOBAL. Per això s'executen diverses trajectòries
en paral·lel, cadascuna des d'un punt d'inici diferent.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# 1. Funció de cost amb múltiples mínims locals i el seu gradient analític
# ---------------------------------------------------------------------------
def f(x, y):
    """Funció de cost J(x, y) amb diversos mínims locals."""
    return np.sin(x) * np.cos(y) + 0.1 * (x**2 + y**2)


def grad_f(x, y):
    """Gradient analític de f: (df/dx, df/dy)."""
    df_dx = np.cos(x) * np.cos(y) + 0.2 * x
    df_dy = -np.sin(x) * np.sin(y) + 0.2 * y
    return np.array([df_dx, df_dy])


# ---------------------------------------------------------------------------
# 2. Descens de gradient
# ---------------------------------------------------------------------------
def gradient_descent(start, lr=0.1, n_iter=80):
    theta = np.array(start, dtype=float)
    trajectory = [theta.copy()]
    for _ in range(n_iter):
        grad = grad_f(theta[0], theta[1])
        theta = theta - lr * grad
        trajectory.append(theta.copy())
    return np.array(trajectory)


# Diversos punts d'inici -> poden acabar en mínims locals DIFERENTS
learning_rate = 0.2
n_iterations = 70

start_points = [
    (-4.0, 3.5),
    (3.0, -3.5),
    (0.5, 4.0),
    (-3.5, -2.0),
]
colors = ["#ff6b6b", "#4dd0e1", "#ffd166", "#a78bfa"]

trajectories = [gradient_descent(sp, lr=learning_rate, n_iter=n_iterations) for sp in start_points]
z_trajs = [f(t[:, 0], t[:, 1]) for t in trajectories]


# ---------------------------------------------------------------------------
# 3. Superfície 3D
# ---------------------------------------------------------------------------
x_range = np.linspace(-5, 5, 150)
y_range = np.linspace(-5, 5, 150)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)


# ---------------------------------------------------------------------------
# 4. Figura
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117")

ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.set_facecolor("#0f1117")

ax_cost = fig.add_subplot(1, 2, 2)
ax_cost.set_facecolor("#161925")

ax3d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.55, linewidth=0, antialiased=True)
ax3d.contour(X, Y, Z, levels=20, cmap=cm.viridis, offset=Z.min() - 2, alpha=0.35)

ax3d.set_xlabel("x", color="white", labelpad=8)
ax3d.set_ylabel("y", color="white", labelpad=8)
ax3d.set_zlabel("f(x, y)", color="white", labelpad=8)
ax3d.set_title("Descens de gradient — funció amb mínims locals", color="white", fontsize=12, pad=15)
ax3d.tick_params(colors="#9aa0ac")
ax3d.xaxis.pane.set_facecolor("#0f1117")
ax3d.yaxis.pane.set_facecolor("#0f1117")
ax3d.zaxis.pane.set_facecolor("#0f1117")
ax3d.view_init(elev=38, azim=-55)

balls = []
paths = []
for c in colors:
    ball, = ax3d.plot([], [], [], "o", color=c, markersize=8, zorder=5)
    path, = ax3d.plot([], [], [], "-", color=c, linewidth=1.8, alpha=0.85, zorder=4)
    balls.append(ball)
    paths.append(path)

ax_cost.set_title("Evolució de J per a cada trajectòria", color="white", fontsize=12)
ax_cost.set_xlabel("Iteració", color="white")
ax_cost.set_ylabel("J(x, y)", color="white")
ax_cost.tick_params(colors="#9aa0ac")
for spine in ax_cost.spines.values():
    spine.set_color("#3a3f4b")
ax_cost.grid(alpha=0.2)

cost_lines = [ax_cost.plot([], [], color=c, linewidth=2)[0] for c in colors]
cost_points = [ax_cost.plot([], [], "o", color=c, markersize=6)[0] for c in colors]

ax_cost.set_xlim(0, n_iterations)
all_z = np.concatenate(z_trajs)
ax_cost.set_ylim(all_z.min() - 0.5, all_z.max() + 0.5)

fig.tight_layout()


# ---------------------------------------------------------------------------
# 5. Animació
# ---------------------------------------------------------------------------
def update(frame):
    artists = []
    for i in range(len(start_points)):
        xs = trajectories[i][: frame + 1, 0]
        ys = trajectories[i][: frame + 1, 1]
        zs = z_trajs[i][: frame + 1]

        balls[i].set_data([xs[-1]], [ys[-1]])
        balls[i].set_3d_properties([zs[-1]])

        paths[i].set_data(xs, ys)
        paths[i].set_3d_properties(zs)

        cost_lines[i].set_data(np.arange(frame + 1), zs)
        cost_points[i].set_data([frame], [zs[-1]])

        artists += [balls[i], paths[i], cost_lines[i], cost_points[i]]
    return artists


anim = FuncAnimation(
    fig, update, frames=n_iterations + 1, interval=120, blit=False, repeat=True
)

# ---------------------------------------------------------------------------
# 6. Exportar
# ---------------------------------------------------------------------------
output_path = "/mnt/user-data/outputs/descens_gradient_minims_locals.gif"
anim.save(output_path, writer=PillowWriter(fps=10))

print(f"Animació guardada a: {output_path}\n")
for sp, traj, z in zip(start_points, trajectories, z_trajs):
    print(f"Inici {sp} -> Final ({traj[-1,0]:.2f}, {traj[-1,1]:.2f}), J final = {z[-1]:.3f}")
