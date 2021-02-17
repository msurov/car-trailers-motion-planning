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


def interpolate_waypoints(waypts, k=5):
    waypts = np.array(waypts)
    bzr = interpolate(waypts, k)
    sp = bezier2bspline(bzr)
    return sp


def find_trajectory(waypts, ntrailers, reverse=False):
    R'''
        TODO: implement for arbitrary trailers
    '''
    assert ntrailers > 0
    delta = np.pi if reverse else 0

    sp = interpolate_waypoints(waypts)
    t = np.linspace(sp.t[0], sp.t[-1], 1000)
    trailers = []

    # train N
    p = sp(t, 0)
    v = sp(t, 1)
    theta = np.arctan2(v[:,1], v[:,0]) + delta
    fix_conitnuity(theta)
    trailer = (sp, p, v, theta)
    trailers += [trailer]

    for i in range(1, ntrailers):
        prev_trailer = trailers[-1]
        _, prev_p, prev_v, prev_theta = prev_trailer
        p = prev_p + np.array([np.cos(prev_theta), np.sin(prev_theta)]).T
        sp =  make_interp_spline(t, p, 5)
        v = sp(t, 1)
        theta = np.arctan2(v[:,1], v[:,0]) + delta
        fix_conitnuity(theta)
        trailer = (sp, p, v, theta)
        trailers += [trailer]

    trailers = trailers[::-1]
    _, p, v, theta = trailers[0]
    
    u1 = np.cos(theta) * v[:,0] + np.sin(theta) * v[:,1]
    stheta = make_interp_spline(t, theta, 5)
    dtheta = stheta(t, 1)
    phi = np.arctan2(dtheta, u1) + delta
    fix_conitnuity(phi)
    fphi = make_interp_spline(t, phi, 5)
    u2 = fphi(t, 1)

    keys = ['ntrailers', 't', 'phi', 'u1', 'u2'] + \
        ['x%d' % i for i in range(ntrailers)] + \
        ['y%d' % i for i in range(ntrailers)] + \
        ['theta%d' % i for i in range(ntrailers)]
    
    values = [ntrailers, t, phi, u1, u2] + \
        [trailer[1][:,0] for trailer in trailers] + \
        [trailer[1][:,1] for trailer in trailers] + \
        [trailer[3] for trailer in trailers]

    traj = dict(zip(keys, values))
    return traj


def test_trajectory(traj):
    from misc.math.integrate import integrate

    ntrailers = traj['ntrailers']
    t = traj['t']
    x0 = traj['x0']
    y0 = traj['y0']
    phi = traj['phi']
    thetas = np.array([traj['theta%d' % i] for i in range(ntrailers)])
    u1 = traj['u1']
    u2 = traj['u2']
    fu = make_interp_spline(t, np.array([u1, u2]).T, k=3)

    dynamics = Dynamics(ntrailers)

    def rhs(t, state):
        u = fu(t)
        return dynamics.rhs(*state, *u)

    state0 = [x0[0], y0[0], phi[0]] + list(thetas[:,0])
    _, state = integrate(rhs, state0, [t[0], t[-1]], step=t[-1]/1000)

    plt.gca().set_prop_cycle(None)
    plt.plot(state[:,0], state[:,1])
    plt.gca().set_prop_cycle(None)
    plt.plot(x0, y0, '--')
    plt.show()


if __name__ == '__main__':
    from misc.format.serialize import save

    traj = find_trajectory([
        [0, 0],
        [3, 0],
        [3, 3],
        [0, 3],
        [0, -3],
        [-3, -3],
        [-3, 0],
        [0, 0],
    ], 4)
    save('data/traj.npz', traj)
    test_trajectory(traj)

    plt.axis('equal')
    for i in range(traj['ntrailers']):
        plt.plot(traj['x%d' % i], traj['y%d' % i], '.', alpha=0.1)
    plt.grid(True)
    plt.show()
