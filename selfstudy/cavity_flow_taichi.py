import numpy as np
from matplotlib import pyplot, cm
import taichi as ti
import time

ti.init(arch=ti.metal)

nx = 101
ny = 101
nit = 50
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)

rho = 1
nu = .1
dt = .001

u = ti.Vector.field(n=2, dtype=ti.float32, shape=(ny, nx))
un = ti.Vector.field(n=2, dtype=ti.float32, shape=(ny, nx))
p = ti.field(dtype=ti.f32, shape=(ny, nx))
pn = ti.field(dtype=ti.f32, shape=(ny, nx))
b = ti.field(dtype=ti.f32, shape=(ny, nx))
rgb_buf = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))
rgb_buf_90 = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))


@ti.func
def build_up_b():
    for j in range(1, ny):
        for i in range(1, nx):
            b[j, i] = rho * (
                    1 / dt * (
                    (u[j, i + 1][0] - u[j, i - 1][0]) / (2 * dx) + (u[j + 1, i][1] - u[j - 1, i][1]) / (2 * dy)) -
                    ((u[j, i + 1][0] - u[j, i - 1][0]) / (2 * dx)) ** 2 -
                    2 * ((u[j + 1, i][0] - u[j - 1, i][0]) / (2 * dy) * (u[j, i + 1][1] - u[j, i - 1][1]) / (2 * dx)) -
                    ((u[j + 1, i][1] - u[j - 1, i][1]) / (2 * dy)) ** 2
            )


@ti.func
def pressure_update():
    build_up_b()

    for q in range(nit):
        for j in range(1, ny):
            for i in range(1, nx):
                p[j, i] = ((pn[j, i + 1] + pn[j, i - 1]) * dy ** 2 + (pn[j + 1, i] + pn[j - 1, i]) * dx ** 2) / \
                          (2 * (dx ** 2 + dy ** 2)) - \
                          dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) * b[j, i]

        for j in range(ny):
            p[j, nx - 1] = p[j, nx - 2]
            p[j, 0] = p[j, 1]

        for i in range(nx):
            p[0, i] = p[1, i]
            p[ny - 1, i] = 0

        for j in range(1, ny):
            for i in range(1, nx):
                pn[j, i] = p[j, i]


@ti.kernel
def update():
    vel_update()


@ti.kernel
def init():
    for j in range(ny):
        for i in range(nx):
            p[j, i] = 0
            u[j, i] = [0, 0]

    for i in range(nx):
        u[0, i] = [1, 0]  # set velocity on cavity lid equal to 1


@ti.func
def visualize_scalar(val):
    return ti.Vector([val, val, val])


@ti.func
def visualize_vec(vec):
    e = vec.norm()
    return ti.Vector([e, e, e])


@ti.func
def rot90():
    """taichi gui index start from left-bottom"""
    for i in range(ny):
        for j in range(nx):
            rgb_buf_90[j, ny - 1 - i] = rgb_buf[i, j]


@ti.kernel
def render(v: ti.template()):
    for j in range(ny):
        for i in range(nx):
            rgb_buf[j, i] = 10 * visualize_vec(v[j, i])

    rot90()


@ti.func
def vel_update():
    pressure_update()

    for j in range(1, ny):
        for i in range(1, nx):
            u[j, i][0] = (un[j, i][0] -
                          un[j, i][0] * dt / dx * (un[j, i][0] - un[j, i - 1][0]) -
                          un[j, i][1] * dt / dy * (un[j, i][0] - un[j - 1, i][0]) -
                          dt / (2 * rho * dx) * (p[j, i + 1] - p[j, i - 1]) +
                          nu * (dt / dx ** 2 * (un[j, i + 1][0] - 2 * un[j, i][0] + un[j, i - 1][0]) +
                                dt / dy ** 2 * (un[j + 1, i][0] - 2 * un[j, i][0] + un[j - 1, i][0])))
            u[j, i][1] = (un[j, i][1] -
                          un[j, i][0] * dt / dx * (un[j, i][1] - un[j, i - 1][1]) -
                          un[j, i][1] * dt / dy * (un[j, i][1] - un[j - 1, i][1]) -
                          dt / (2 * rho * dy) * (p[j + 1, i] - p[j - 1, i]) +
                          nu * (dt / dx ** 2 * (un[j, i + 1][1] - 2 * un[j, i][1] + un[j, i - 1][1]) +
                                dt / dy ** 2 * (un[j + 1, i][1] - 2 * un[j, i][1] + un[j - 1, i][1])))

    for i in range(nx):
        u[0, i] = [1, 0]
        u[ny - 1, i] = [0, 0]  # set velocity on cavity lid equal to 1

    for j in range(ny):
        u[j, 0] = [0, 0]
        u[j, nx - 1] = [0, 0]


def im_show_v(u, v):
    m = np.sqrt(u ** 2 + v ** 2)
    pyplot.imshow(m)
    pyplot.colorbar()
    pyplot.show()


def im_show_p(p):
    pyplot.imshow(p)
    pyplot.colorbar()
    pyplot.show()


if __name__ == '__main__':
    gui = ti.GUI('cavity flow', res=(ny, nx))
    init()

    un = u

    # while gui.running:
    for _ in range(50):
        update()
        un = u
        render(u)
        gui.set_image(rgb_buf_90)
        gui.show()
        time.sleep(.1)

    x_vel_py = u.to_numpy()[:, :, 0]
    y_vel_py = u.to_numpy()[:, :, 1]

    print(x_vel_py.shape)
    print(y_vel_py.shape)

    im_show_v(x_vel_py, y_vel_py)
    im_show_p(p.to_numpy())
