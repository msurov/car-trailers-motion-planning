from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from car_trailer_kinematics import CarTrailerKinematic, get_trailer_pose, get_car_pose
from dataclasses import dataclass

def rotmat(a):
    sa = np.sin(a)
    ca = np.cos(a)
    return np.array([
        [ca, -sa],
        [sa, ca]
    ])


def rotate_contour(c, angle):
    return c @ rotmat(angle).T


def shift_contour(c, dx, dy):
    return c + np.array([dx, dy])


def affine(c, dx=0., dy=0., alpha=0.):
    return shift_contour(rotate_contour(c, alpha), dx, dy)


def wheel_contour(width):
    return width * np.array([
        [-0.05, 0.0],
        [-0.05, 0.05],
        [-0.2, 0.05],
        [-0.2, 0.15],
        [0.2, 0.15],
        [0.2, 0.05],
        [0.05, 0.05],
        [0.05, 0.0],
    ])


def trailer_body_points(width, length):
    return np.array([
        [-0.25 * length, 0.45 * width],
        [ 0.50 * length, 0.45 * width],
        [ 1.00 * length, 0.00 * width],
        [ 0.50 * length,-0.45 * width],
        [-0.25 * length,-0.45 * width],
        [-0.25 * length, 0.45 * width],
    ])


def circle(cx, cy, r):
    t = np.linspace(0, 2*np.pi, 64)
    return np.array([cx + r*np.sin(t), cy + r*np.cos(t)]).T


def axis(width):
    return width * np.array([
        [-0.01,-0.55],
        [-0.01, 0.55],
        [ 0.01, 0.55],
        [ 0.01,-0.55],
        [-0.01,-0.55],
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

    def patch(self):
        return self.poly


def car_body_points(width, length):
    return np.array([
        [-0.25 * length, 0.45 * width],
        [ 1.20 * length, 0.45 * width],
        [ 1.25 * length, 0.05 * width],
        [ 1.20 * length,-0.45 * width],
        [-0.25 * length,-0.45 * width],
        [-0.25 * length, 0.45 * width],
    ])


class Trailer:
    def __init__(self, cfg : CarTrailerKinematic, color : str):
        self.x = 0
        self.y = 0
        self.a = 0
        self.b = cfg.trailer_base
        width = cfg.wheels_diastance
        self.d = width / 2
        alpha = 0.8
        wheel1 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel2.move(y=-self.d, a=np.pi)
        body = Part(trailer_body_points(width, self.b), color, self, alpha=alpha)
        # o = Part(circle(0, 0, 0.1), 'black', self, alpha=alpha)
        a = Part(axis(width), 'black', self, alpha=alpha)
        self.parts = [wheel1, wheel2, body, a]

    def move(self, x, y, theta):
        self.x = x
        self.y = y
        self.a = theta
        for p in self.parts:
            p.update()

    def patches(self):
        return [p.poly for p in self.parts]

class Car:
    def __init__(self, cfg : CarTrailerKinematic, color : str):
        self.x = 0
        self.y = 0
        self.a = 0
        self.s = 0

        self.b = cfg.wheel_base
        width = cfg.wheels_diastance
        self.d = width / 2
        self.m = cfg.pivot_distance

        alpha = 0.8

        wheel1 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel1.move(y=self.d)
        wheel2 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel2.move(y=-self.d, a = np.pi)

        wheel3 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel3.move(y=self.d, x=self.b)
        wheel4 = Part(wheel_contour(width), 'black', self, alpha=alpha)
        wheel4.move(y=-self.d, x=self.b, a=np.pi)

        body = Part(car_body_points(width, self.b), color, self, alpha=alpha)

        a1 = Part(axis(width), 'black', self, alpha=alpha)
        a2 = Part(axis(width), 'black', self, alpha=alpha)
        a2.move(x=self.b)

        o = Part(circle(-self.m, 0, 0.05 * width), 'black', self, alpha=alpha)

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


class CarTrailer:
    def __init__(self, cfg : CarTrailerKinematic):
        self.cfg = cfg
        self.car = Car(cfg, 'magenta')
        self.trailer = Trailer(cfg, 'lightblue')
        self.move(0, 0, 0, 0, 0)

    def get_trailer_pose(self, x0, y0, theta, psi):
        L1 = self.cfg.wheel_base
        L2 = self.cfg.trailer_base
        d = self.cfg.pivot_distance
        x1 = x0 - np.cos(theta) * d
        y1 = y0 - np.sin(theta) * d
        x2 = x1 - np.cos(theta + psi) * L2
        y2 = y1 - np.sin(theta + psi) * L2
        return x2, y2, theta + psi

    def move(self, x0, y0, theta, psi, phi):
        self.car.move(x0, y0, theta, phi)
        xt,yt,at = self.get_trailer_pose(x0, y0, theta, psi)
        self.trailer.move(xt, yt, at)

    def patches(self):
        return self.car.patches() + \
            self.trailer.patches()


if __name__ == '__main__':
    cfg = CarTrailerKinematic(
        2.0, 1.3, 0.4, 1.0
    )
    cartrailer = CarTrailer(cfg)
    plt.gca().set_aspect('equal')
    plt.gca().set_xlim(-5, 5)
    plt.gca().set_ylim(-5, 5)
    [plt.gca().add_patch(p) for p in cartrailer.patches()]
    cartrailer.move(1, 2, 0.5, 0, -0.1)
    plt.show()
