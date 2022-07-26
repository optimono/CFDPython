from matplotlib import pyplot, cm
import numpy
import taichi as ti

ti.init(arch=ti.metal)

###variable declarations
nx = 201
ny = 201
nt = 80
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

u = ti.Vector.field(n=2, dtype=ti.f32, shape=(ny, nx))
un = ti.Vector.field(n=2, dtype=ti.f32, shape=(ny, nx))
rgb_buf = ti.Vector.field(3, dtype=ti.f32, shape=(ny, nx))


###Assign initial conditions
@ti.kernel
def init():
    for i, j in u:
        e = ti.Vector([1, 1])
        ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
        if 0.5 < i * dy < 1 and 0.5 < j * dx < 1:
            e = [2, 2]

        u[i, j] = e


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
        rgb_buf[i, j] = 0.2 * visualize_vec(v[i, j])


@ti.kernel
def update():
    for j in range(1, ny):
        for i in range(1, nx):
            u[j, i][0] = un[j, i][0] - \
                         un[j, i][0] * c * dt / dx * (un[j, i][0] - un[j, i - 1][0]) - \
                         un[j, i][1] * c * dt / dy * (un[j, i][0] - un[j - 1, i][0])

            u[j, i][1] = un[j, i][1] - \
                         un[j, i][0] * c * dt / dx * (un[j, i][1] - un[j, i - 1][1]) - \
                         un[j, i][1] * c * dt / dy * (un[j, i][1] - un[j - 1, i][1])

    for i in range(ny):
        u[i, 0] = [1, 1]
        u[i, nx - 1] = [1, 1]

    for i in range(nx):
        u[0, i] = [1, 1]
        u[ny - 1, i] = [1, 1]


if __name__ == '__main__':
    gui = ti.GUI('2d convection', res=(ny, nx))
    init()
    un = u

    # while gui.running:
    for _ in range(nt):
        update()
        un = u
        render(u)
        gui.set_image(rgb_buf)
        gui.show()
