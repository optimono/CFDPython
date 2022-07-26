import numpy
from matplotlib import pyplot, cm
import taichi as ti

ti.init(arch=ti.metal)

###variable declarations
nx = 401
ny = 401
nu = .05
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .25
dt = sigma * dx * dy / nu

velocity = ti.field(dtype=ti.f32, shape=(ny, nx))
velocity_old = ti.field(dtype=ti.f32, shape=(ny, nx))
rgb_buf = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))


###Assign initial conditions
@ti.kernel
def init():
    for i, j in velocity:
        v = 1
        ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
        if 0.5 < i * dy < 1 and 0.5 < j * dx < 1:
            v = 2
        velocity[i, j] = v


@ti.func
def visualize_scalar(val):
    return ti.Vector([val, val, val])


@ti.kernel
def render(v: ti.template()):
    for i, j in rgb_buf:
        rgb_buf[i, j] = 0.2 * visualize_scalar(v[i, j])


@ti.kernel
def update():
    for j in range(1, ny):
        for i in range(1, nx):
            velocity[j, i] = (velocity_old[j, i] +
                              nu * dt / dx ** 2 *
                              (velocity_old[j, i + 1] - 2 * velocity_old[j, i] + velocity_old[j, i - 1]) +
                              nu * dt / dy ** 2 *
                              (velocity_old[j + 1, i] - 2 * velocity_old[j, i] + velocity_old[j - 1, i]))
    for i in range(ny):
        velocity[i, 0] = 1
        velocity[i, nx - 1] = 1

    for i in range(nx):
        velocity[0, i] = 1
        velocity[ny - 1, i] = 1


if __name__ == '__main__':
    gui = ti.GUI('linear convection', res=(ny, nx))
    init()
    velocity_old = velocity

    while gui.running:
        update()
        velocity_old = velocity
        render(velocity)
        gui.set_image(rgb_buf)
        gui.show()
