from matplotlib.patches import Rectangle, Polygon, Circle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline
import tempfile
from os.path import join


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
    def __init__(self, cont, color, parent, alpha=0.8):
        self.parent = parent
        self.pts = cont
        self.poly = Polygon(cont, True, zorder=3, alpha=alpha, color=color)
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0

    @property
    def alpha(self):
        return self.poly.alpha

    @alpha.setter
    def alpha(self, value):
        self.poly.alpha = value

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
    def __init__(self, color, grayed=False):
        self.x = 0
        self.y = 0
        self.a = 0
        self.d = 0.4
        alpha = 0.2 if grayed else 0.8
        wheel1 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel2.move(y=-self.d, a=np.pi)
        body = Part(trailer_body_points(), color, self, alpha=alpha)
        o = Part(circle(0, 0, 0.1), 'black', self, alpha=alpha)
        a = Part(axis(), 'black', self, alpha=alpha)
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
    def __init__(self, color, grayed=False):
        self.x = 0
        self.y = 0
        self.a = 0
        self.s = 0

        self.b = 1.0
        self.d = 0.4

        alpha = 0.2 if grayed else 0.8

        wheel1 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel2.move(y=-self.d, a = np.pi)

        wheel3 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel3.move(y=self.d, x=self.b)
        wheel4 = Part(wheel_contour(), 'black', self, alpha=alpha)
        wheel4.move(y=-self.d, x=self.b, a=np.pi)

        body = Part(car_body_points(), color, self, alpha=alpha)

        a1 = Part(axis(), 'black', self, alpha=alpha)
        a2 = Part(axis(), 'black', self, alpha=alpha)
        a2.move(x=self.b)

        o = Part(circle(0, 0, 0.1), 'black', self, alpha=alpha)

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
    def __init__(self, ntrailers, grayed=False):
        if grayed:
            self.car = Car('gray', grayed=True)
            self.trailers = [Trailer('gray', grayed=True) for i in range(1, ntrailers)]
        else:
            self.car = Car(trailers_colors[0])
            self.trailers = [Trailer(trailers_colors[i]) for i in range(1, ntrailers)]
        self.move(0, 0, 0, np.zeros(ntrailers))

    def get_trailer_pose(self, i, x0, y0, thetas):
        dx = np.sum(np.cos(thetas[1:i+1]))
        dy = np.sum(np.sin(thetas[1:i+1]))
        return x0 - dx, y0 - dy, thetas[i]

    def move(self, x0, y0, phi, thetas):
        ntrailers = len(self.trailers) + 1
        self.car.move(x0, y0, thetas[0], phi)

        for i in range(1, ntrailers):
            x,y,theta = self.get_trailer_pose(i, x0, y0, thetas)
            self.trailers[i-1].move(x, y, theta)

    def patches(self):
        patches = self.car.patches()
        for t in self.trailers:
            patches = patches + t.patches()
        return patches


def to_cartesian(x, y, thetas):
    '''
        Returns cartesian trajectories of the trails
    '''
    ntrailers, nt = np.shape(thetas)
    trailers = np.zeros((ntrailers, nt, 2), float)
    trailers[0,:,0] = x
    trailers[0,:,1] = y

    for i in range(1, ntrailers):
        trailers[i,:,0] = trailers[i-1,:,0] - np.cos(thetas[i])
        trailers[i,:,1] = trailers[i-1,:,1] - np.sin(thetas[i])

    return trailers


def animate(traj, fps=60, animtime=None, speedup=None, filepath=None):
    ntrailers = traj['ntrailers']

    t = traj['t']
    x = traj['x']
    y = traj['y']

    phi = traj['phi']
    thetas = np.array(traj['theta'])
    npts = len(t)

    trailers = to_cartesian(x, y, thetas)

    xmin = np.min(trailers[:,:,0]) - ntrailers
    xmax = np.max(trailers[:,:,0]) + ntrailers
    ymin = np.min(trailers[:,:,1]) - ntrailers
    ymax = np.max(trailers[:,:,1]) + ntrailers

    # fig = plt.figure(figsize=(16,9))
    fig = plt.figure(figsize=(8,4))
    plt.axis('equal')
    ax = plt.gca()
    plt.grid(True)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    for i in range(ntrailers):
        plt.plot(trailers[i,:,0], trailers[i,:,1], '.', alpha=0.1, color=trailers_colors[i])

    cartrailers = CarTrailers(ntrailers)
    pathes = cartrailers.patches()
    cartrailers.move(x[0], y[0], phi[0], thetas[:,0])

    for p in pathes:
        ax.add_patch(p)

    cartrailers0 = CarTrailers(ntrailers, grayed=True)
    cartrailers0.move(x[0], y[0], phi[0], thetas[:,0])
    for p in cartrailers0.patches():
        ax.add_patch(p)

    cartrailersf = CarTrailers(ntrailers, grayed=True)
    cartrailersf.move(x[-1], y[-1], phi[-1], thetas[:,-1])
    for p in cartrailersf.patches():
        ax.add_patch(p)

    def init():
        patches = cartrailers.patches()
        [ax.add_patch(p) for p in patches]
        return patches
    
    state = np.zeros((len(t), ntrailers + 3))
    state[:,0] = x
    state[:,1] = y
    state[:,2] = phi
    state[:,3:] = thetas.T
    sp = make_interp_spline(t, state, k=1)

    def update(i):
        x,y,phi,*thetas = sp(t[0] + i * speedup / fps)
        cartrailers.move(x, y, phi, thetas)
        return cartrailers.patches()
    
    if speedup is not None:
        assert speedup > 0
        nframes = int((t[-1] - t[0]) * fps / speedup)
    elif animtime is not None:
        assert animtime > 0
        nframes = int(animtime * fps)
        speedup = (t[-1] - t[0]) / animtime
    else:
        nframes = int((t[-1] - t[0]) * fps)
        speedup = 1
    
    fig.subplots_adjust(bottom=0.06, top=0.99, left=0.06, right=0.99,)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=nframes, blit=True)
    if filepath is not None:
        if filepath.endswith('.gif'):
            writer='imagemagick'
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Maksim Surov'), bitrate=400*60)
        anim.save(filepath, writer)
    else:
        plt.show()


if __name__ == '__main__':
    datadir = tempfile.gettempdir()
    data = np.load(
        join(datadir, 'traj-2.npy'),
        allow_pickle=True
    )
    traj = data.item()
    animate(traj, animtime=5)
