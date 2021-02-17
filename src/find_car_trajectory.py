from dynamics import Dynamics
from bezier.bezier import interpolate, bezier2poly, eval_bezier
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from dynamics import Dynamics


def bezier2bspline(bzr):
    n,d,_ = bzr.shape
    args = np.linspace(0, 1, d)
    vals = [eval_bezier(c, args) for c in bzr]
    vals = np.concatenate([vals[0]] + [v[1:] for v in vals[1:]])
    n,_ = vals.shape
    args = np.linspace(0, 1, n)
    sp = make_interp_spline(args, vals, d-1)
    return sp


def fix_conitnuity(arr):
    for i in range(1, len(arr)):
        if arr[i-1] - arr[i] > np.pi:
            arr[i] += 2*np.pi
        elif arr[i-1] - arr[i] < -np.pi:
            arr[i] -= 2*np.pi


def find_trajectory(waypts):
    R'''
        TODO: implement for arbitrary trailers
    '''
    waypts = np.array(waypts)
    bzr = interpolate(waypts, 7)
    sp = bezier2bspline(bzr)
    t = np.linspace(0, 1, 1000)

    p0 = sp(t)
    v0 = sp(t, 1)
    theta0 = np.arctan2(v0[:,1], v0[:,0])
    fix_conitnuity(theta0)

    u1 = np.cos(theta0) * v0[:,0] + np.sin(theta0) * v0[:,1]
    ftheta0 = make_interp_spline(t, theta0, 5)
    dtheta0 = ftheta0(t, 1)
    phi = np.arctan2(dtheta0, u1)
    fix_conitnuity(phi)
    fphi = make_interp_spline(t, phi, 3)
    u2 = fphi(t, 1)

    return {
        'ntrailers': 1,
        't': t,
        'x0': p0[:,0],
        'y0': p0[:,1],
        'theta0': theta0,
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
    u1 = traj['u1']
    u2 = traj['u2']
    npts = len(t)
    fu = make_interp_spline(t, np.array([u1, u2]).T, k=3)

    dynamics = Dynamics(ntrailers)

    def rhs(t, state):
        u = fu(t)
        return dynamics.rhs(*state, *u)

    state0 = [
        x0[0], y0[0], phi[0], theta0[0]
    ]
    _, state = integrate(rhs, state0, [t[0], t[-1]], step=1e-3)

    plt.gca().set_prop_cycle(None)
    plt.plot(state[:,0], state[:,1])
    plt.gca().set_prop_cycle(None)
    plt.plot(x0, y0, '--')
    plt.show()


if __name__ == '__main__':
    from misc.format.serialize import save

    traj = find_trajectory([
        [0, 0],
        [3, 1],
        [6, -1],
        [9, 1],
        [12, -1],
        [15, 1],
        [18, 0],
        [18, -3],
        [0, -1]
    ])
    save('data/traj.npz', traj)

    test_trajectory(traj)

    x0 = traj['x0']
    y0 = traj['y0']

    plt.axis('equal')
    plt.plot(x0, y0, 'o', color='blue', alpha=0.1)
    plt.grid(True)
    plt.show()
