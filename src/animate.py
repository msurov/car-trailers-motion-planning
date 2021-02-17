from matplotlib.patches import Rectangle, Polygon, Circle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

font = {'size': 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['text.usetex'] = True

trailers_colors = [
    'forestgreen',
    'orchid',
    'darkcyan',
    'plum',
    'lightsteelblue',
]

def rotmat(a):
    sin = np.sin(a)
    cos = np.cos(a)
    return np.array([
        [cos, -sin],
        [sin, cos]
    ])


def rotate_contour(c, angle):
    return c @ rotmat(angle).T


def shift_contour(c, dx, dy):
    return c + np.array([dx, dy])


def affine(c, dx=0., dy=0., alpha=0.):
    return shift_contour(rotate_contour(c, alpha), dx, dy)


def wheel_contour():
    return np.array([
        [-0.05, 0.0],
        [-0.05, 0.05],
        [-0.2, 0.05],
        [-0.2, 0.15],
        [0.2, 0.15],
        [0.2, 0.05],
        [0.05, 0.05],
        [0.05, 0.0],
    ])


def trailer_body_points():
    return np.array([
        [-0.25, 0.4],
        [ 0.50, 0.4],
        [ 1.00, 0.0],
        [ 0.50,-0.4],
        [-0.25,-0.4],
        [-0.25, 0.4],
    ])


def circle(cx, cy, r):
    t = np.linspace(0, 2*np.pi, 64)
    return np.array([cx + r*np.sin(t), cy + r*np.cos(t)]).T


def axis():
    return np.array([
        [-0.01,-0.45],
        [-0.01, 0.45],
        [ 0.01, 0.45],
        [ 0.01,-0.45],
        [-0.01,-0.45],
    ])


class Part:
    def __init__(self, cont, color, parent):
        self.parent = parent
        self.pts = cont
        self.poly = Polygon(cont, True, zorder=3, alpha=0.6, color=color)
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0

    def move(self, x=0, y=0, a=0):
        R'''
            move part to the pose (x,y,angle) wrt parent
        '''
        self.x = x
        self.y = y
        self.a = a
        self.update()

    def update(self):
        px = self.parent.x
        py = self.parent.y
        pa = self.parent.a
        pR = rotmat(pa)

        wa = pa + self.a
        wx, wy = [px, py] + pR @ [self.x, self.y]

        pts = affine(self.pts, wx, wy, wa)
        self.poly.set_xy(pts)

    def patch():
        return self.poly


def car_body_points():
    return np.array([
        [-0.25, 0.4],
        [ 1.20, 0.4],
        [ 1.25, 0.0],
        [ 1.20,-0.4],
        [-0.25,-0.4],
        [-0.25, 0.4],
    ])


class Trailer:
    def __init__(self, color):
        self.x = 0
        self.y = 0
        self.a = 0
        self.d = 0.4
        wheel1 = Part(wheel_contour(), 'black', self)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(), 'black', self)
        wheel2.move(y=-self.d, a=np.pi)
        body = Part(trailer_body_points(), color, self)
        o = Part(circle(0, 0, 0.1), 'black', self)
        a = Part(axis(), 'black', self)
        self.parts = [wheel1, wheel2, body, o, a]

    def move(self, x, y, theta):
        self.x = x
        self.y = y
        self.a = theta
        for p in self.parts:
            p.update()

    def patches(self):
        return [p.poly for p in self.parts]


class Car:
    def __init__(self, color):
        self.x = 0
        self.y = 0
        self.a = 0
        self.s = 0

        self.b = 1.0
        self.d = 0.4

        wheel1 = Part(wheel_contour(), 'black', self)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(), 'black', self)
        wheel2.move(y=-self.d, a = np.pi)

        wheel3 = Part(wheel_contour(), 'black', self)
        wheel3.move(y=self.d, x=self.b)
        wheel4 = Part(wheel_contour(), 'black', self)
        wheel4.move(y=-self.d, x=self.b, a=np.pi)

        body = Part(car_body_points(), color, self)

        a1 = Part(axis(), 'black', self)
        a2 = Part(axis(), 'black', self)
        a2.move(x=self.b)

        o = Part(circle(0, 0, 0.1), 'black', self)

        self.parts = [wheel1, wheel2, wheel3, wheel4, body, a1, a2, o]

    def move(self, x, y, theta, phi):
        self.x = x
        self.y = y
        self.a = theta
        self.s = phi

        phi1 = np.arctan2(self.b * np.tan(phi), (self.b - self.d * np.tan(phi)))
        phi2 = np.arctan2(self.b * np.tan(phi), (self.b + self.d * np.tan(phi)))

        wheel3 = self.parts[2]
        wheel4 = self.parts[3]
        wheel3.a = phi1
        wheel4.a = phi2 + np.pi

        for p in self.parts:
            p.update()

    def patches(self):
        return [p.poly for p in self.parts]


class CarTrailers:
    def __init__(self, ntrailers):
        self.car = Car(trailers_colors[0])
        self.trailers = [Trailer(trailers_colors[i]) for i in range(1, ntrailers)]
        self.move(0, 0, 0, np.zeros(ntrailers))
    
    def get_trailer_pose(self, i, x0, y0, phi, thetas):
        dx = np.sum([np.cos(thetas[j]) for j in range(1, i+1)])
        dy = np.sum([np.sin(thetas[j]) for j in range(1, i+1)])
        return x0 - dx, y0 - dy, thetas[i]

    def move(self, x0, y0, phi, thetas):
        ntrailers = len(self.trailers) + 1
        self.car.move(x0, y0, thetas[0], phi)

        for i in range(1, ntrailers):
            x,y,theta = self.get_trailer_pose(i, x0, y0, phi, thetas)
            self.trailers[i-1].move(x, y, theta)

    def patches(self):
        patches = self.car.patches()
        for t in self.trailers:
            patches = patches + t.patches()
        return patches


def animate(traj):
    ntrailers = traj['ntrailers']
    cartrailers = CarTrailers(ntrailers)

    t = traj['t']
    x0 = traj['x0']
    y0 = traj['y0']
    phi = traj['phi']
    thetas = np.array([traj['theta%d' % i] for i in range(ntrailers)]).T
    npts = len(t)
    xs = [traj['x%d' % i] for i in range(ntrailers)]
    ys = [traj['y%d' % i] for i in range(ntrailers)]

    xmin = np.min(np.concatenate(xs)) - 1
    xmax = np.max(np.concatenate(xs)) + 1
    ymin = np.min(np.concatenate(ys)) - 1
    ymax = np.max(np.concatenate(ys)) + 1

    fig = plt.figure(figsize=(12,9))
    plt.axis('equal')
    ax = plt.gca()
    plt.grid()

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for i in range(ntrailers):
        plt.plot(xs[i], ys[i], '.', alpha=0.1, color=trailers_colors[i])

    pathes = cartrailers.patches()
    cartrailers.move(x0[0], y0[0], phi[0], thetas[0,:])

    for p in pathes:
        ax.add_patch(p)

    def init():
        patches = cartrailers.patches()
        [ax.add_patch(p) for p in patches]
        return patches

    def update(frameidx):
        i = frameidx
        cartrailers.move(x0[i], y0[i], phi[i], thetas[i,:])
        return cartrailers.patches()

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=npts, blit=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=60, metadata=dict(artist='Maksim Surov'), bitrate=400*60)
    anim.save('data/anim.mp4', writer)
    # plt.show()


if __name__ == '__main__':
    from misc.format.serialize import load_dict
    traj = load_dict('data/traj.npz')
    animate(traj)
