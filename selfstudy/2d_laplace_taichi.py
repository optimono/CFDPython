import numpy as np
from matplotlib import pyplot, cm
import taichi as ti

ti.init(arch=ti.metal)

##variable declarations
nx = 201
ny = 201
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

##initial conditions
p_py = np.zeros((ny, nx), dtype=np.float32)  # create a XxY vector of 0's

##plotting aids
x_py = np.linspace(0, 2, nx, dtype=np.float32)
y_py = np.linspace(0, 1, ny, dtype=np.float32)
y = ti.field(dtype=ti.f32, shape=[ny])
y.from_numpy(y_py)

##boundary conditions
p_py[:, 0] = 0  # p = 0 @ x = 0
p_py[:, -1] = y_py  # p = y @ x = 2
p_py[0, :] = p_py[1, :]  # dp/dy = 0 @ y = 0
p_py[-1, :] = p_py[-2, :]  # dp/dy = 0 @ y = 1

p = ti.field(dtype=ti.f32, shape=(ny, nx))
pn = ti.field(dtype=ti.f32, shape=(ny, nx))
rgb_buf = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))


###Assign initial conditions
# @ti.kernel
def init_p():
    p.from_numpy(p_py)


@ti.func
def visualize_scalar(val):
    return ti.Vector([val, val, val])


@ti.func
def visualize_vec(vec):
    e = vec.norm()
    return ti.Vector([e, e, e])


@ti.kernel
def render(v: ti.template()):
    for i, j in rgb_buf:
        rgb_buf[i, j] = 5 * visualize_scalar(v[i, j])


@ti.kernel
def update():
    for j in range(1, ny):
        for i in range(1, nx):
            p[j, i] = (dy ** 2 * (pn[j, i + 1] + pn[j, i - 1]) + dx ** 2 * (pn[j + 1, i] + pn[j - 1, i])) / \
                      (2 * (dx ** 2 + dy ** 2))

    for j in range(ny):
        p[j, 0] = 0
        p[j, nx - 1] = y[j]

    for i in range(nx):
        p[0, i] = p[1, i]
        p[ny - 1, i] = p[ny - 2, i]


def plot2D(x, y, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, p[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    pyplot.show()


if __name__ == '__main__':
    gui = ti.GUI('2d laplace equation', res=(ny, nx))
    init_p()
    pn = p
    # print(p_py)
    # print('-' * 10)
    # print(p)
    plot2D(x_py, y_py, p.to_numpy())

    # while gui.running:
    for _ in range(4935):
        update()
        pn = p
        render(p)
        gui.set_image(rgb_buf)
        gui.show()

    plot2D(x_py, y_py, p.to_numpy())
