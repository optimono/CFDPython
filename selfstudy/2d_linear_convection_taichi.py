from matplotlib import pyplot, cm
import numpy
import taichi as ti

ti.init(arch=ti.metal)

###variable declarations
nx = 501
ny = 501
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx

x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)

velocity = numpy.ones((ny, nx))  ##create a 1xn vector of 1's
velocity_old = numpy.ones((ny, nx))  ##

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
def visualize_norm(val):
    return ti.Vector([val, val, val])


@ti.kernel
def render(v: ti.template()):
    for i, j in rgb_buf:
        rgb_buf[i, j] = 0.2 * visualize_norm(v[i, j])


@ti.kernel
def update():
    for j in range(1, ny):
        for i in range(1, nx):
            velocity[j, i] = (velocity_old[j, i] - (c * dt / dx * (velocity_old[j, i] - velocity_old[j, i - 1])) -
                              (c * dt / dy * (velocity_old[j, i] - velocity_old[j - 1, i])))
            # velocity[0, :] = 1
            # velocity[-1, :] = 1
            # velocity[:, 0] = 1
            # velocity[:, -1] = 1


###Plot Initial Condition
##the figsize parameter can be used to produce different sized images
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
