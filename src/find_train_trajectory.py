from dynamics import Dynamics
from bezier.bezier import interpolate, bezier2poly, eval_bezier, eval_bezier_length
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from dynamics import Dynamics
from time import time


t0 = time()

def bezier2bspline(bzr):
    n,d,_ = bzr.shape
    args = np.linspace(0, 1, d)
    vals = [eval_bezier(c, args) for c in bzr]
    vals = np.concatenate([vals[0]] + [v[1:] for v in vals[1:]])
    lengths = [eval_bezier_length(c, args) for c in bzr]
    for i in range(1, len(lengths)):
        lengths[i] += lengths[i-1][-1]
    lengths = np.concatenate([lengths[0]] + [v[1:] for v in lengths[1:]])
    n,_ = vals.shape
    sp = make_interp_spline(lengths, vals, d-1)
    return sp


def fix_conitnuity(arr):
    for i in range(1, len(arr)):
        if arr[i-1] - arr[i] > np.pi:
            arr[i] += 2*np.pi
        elif arr[i-1] - arr[i] < -np.pi:
            arr[i] -= 2*np.pi


def interpolate_waypoints(waypts):
    waypts = np.array(waypts)
    bzr = interpolate(waypts, 7)
    sp = bezier2bspline(bzr)
    return sp


def find_trajectory(waypts):
    R'''
        TODO: implement for arbitrary trailers
    '''
    sp = interpolate_waypoints(waypts)
    t = np.linspace(sp.t[0], sp.t[-1], 1000)

    p2 = sp(t)
    v2 = sp(t, 1)
    theta2 = np.arctan2(v2[:,1], v2[:,0]) + np.pi
    fix_conitnuity(theta2)

    p1 = p2 + np.array([np.cos(theta2), np.sin(theta2)]).T
    fp1 = make_interp_spline(t, p1, 5)
    v1 = fp1(t, 1)
    theta1 = np.arctan2(v1[:,1], v1[:,0]) + np.pi
    fix_conitnuity(theta1)

    p0 = p1 + np.array([np.cos(theta1), np.sin(theta1)]).T
    fp0 = make_interp_spline(t, p0, 5)
    v0 = fp0(t, 1)
    theta0 = np.arctan2(v0[:,1], v0[:,0]) + np.pi
    fix_conitnuity(theta0)

    u1 = np.cos(theta0) * v0[:,0] + np.sin(theta0) * v0[:,1]
    ftheta0 = make_interp_spline(t, theta0, 5)
    dtheta0 = ftheta0(t, 1)
    phi = np.arctan2(dtheta0, u1) + np.pi
    fix_conitnuity(phi)
    fphi = make_interp_spline(t, phi, 5)
    u2 = fphi(t, 1)

    return {
        'ntrailers': 3,
        't': t,
        'x0': p0[:,0],
        'y0': p0[:,1],
        'x1': p1[:,0],
        'y1': p1[:,1],
        'x2': p2[:,0],
        'y2': p2[:,1],
        'theta0': theta0,
        'theta1': theta1,
        'theta2': theta2,
        'phi': phi,
        'u1': u1,
        'u2': u2
    }


def test_trajectory(traj):
    from misc.math.integrate import integrate

    ntrailers = traj['ntrailers']
    t = traj['t']
    x0 = traj['x0']
    y0 = traj['y0']
    phi = traj['phi']
    theta0 = traj['theta0']
    theta1 = traj['theta1']
    theta2 = traj['theta2']
    u1 = traj['u1']
    u2 = traj['u2']
    npts = len(t)
    fu = make_interp_spline(t, np.array([u1, u2]).T, k=3)

    dynamics = Dynamics(ntrailers)

    def rhs(t, state):
        u = fu(t)
        return dynamics.rhs(*state, *u)

    state0 = [
        x0[0], y0[0], phi[0], theta0[0], theta1[0], theta2[0]
    ]
    _, state = integrate(rhs, state0, [t[0], t[-1]], step=t[-1]/1000)

    plt.gca().set_prop_cycle(None)
    plt.plot(state[:,0], state[:,1])
    plt.gca().set_prop_cycle(None)
    plt.plot(x0, y0, '--')
    plt.show()


if __name__ == '__main__':
    from misc.format.serialize import save

    print(time() - t0)

    traj = find_trajectory([
        [0, 0],
        [3, 0],
        [3, 3],
        [0, 3],
        [0, -3],
        [-3, -3],
        [-3, 0],
        [0, 0],
    ])
    save('data/traj.npz', traj)

    print(time() - t0)

    test_trajectory(traj)

    print(time() - t0)

    x0 = traj['x0']
    y0 = traj['y0']
    x1 = traj['x1']
    y1 = traj['y1']
    x2 = traj['x2']
    y2 = traj['y2']

    plt.axis('equal')
    plt.plot(x0, y0, 'o', color='blue', alpha=0.1)
    plt.plot(x1, y1, 'o', color='green', alpha=0.1)
    plt.plot(x2, y2, 'o', color='pink', alpha=0.1)
    plt.grid(True)
    plt.show()
