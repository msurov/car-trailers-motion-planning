from dynamics import Dynamics
from bezier import interpolate, bezier2poly, eval_bezier, eval_bezier_length
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import factorial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from transforms import get_flat_boundary_values, get_maps
from numpy.polynomial.polynomial import polyval, polyder
from casadi import Function


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


class PolyCurve:
    def __init__(self, coefs, tspan):
        self.coefs = coefs
        self.t1, self.t2 = tspan

    def __call__(self, t, nder=0):
        if nder > 0:
            p = polyder(self.coefs, nder)
        elif nder == 0:
            p = self.coefs
        else:
            assert False
        return polyval(t, p, True).T


def get_phase_trajectory(out : PolyCurve):
    t = np.linspace(out.t1, out.t2, 100)


def find_trajectory(waypts, ntrailers, reverse=False):
    R'''
        TODO: implement for arbitrary trailers
    '''
    assert ntrailers > 0
    delta = np.pi if reverse else 0

    sp = interpolate_waypoints(waypts, 7)
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
        r = dynamics.rhs(*state, *u)
        r = np.reshape(r, (-1,))
        return r

    state0 = [x0[0], y0[0], phi[0]] + list(thetas[:,0])
    ans = solve_ivp(rhs, [t[0], t[-1]], state0, t_eval=t)

    plt.gca().set_prop_cycle(None)
    plt.plot(ans.y[0], ans.y[1])
    plt.gca().set_prop_cycle(None)
    plt.plot(x0, y0, '--')
    plt.show()


def test1():
    waypoints = [
        [0, 0],
        [5, 0],
        [5, 5],
        [0, 5],
        [0, 0],
    ]

    traj = find_trajectory(waypoints, 5, True)
    np.save('/tmp/traj.npy', traj)
    test_trajectory(traj)

    plt.axis('equal')
    for i in range(traj['ntrailers']):
        plt.plot(traj['x%d' % i], traj['y%d' % i], '.', alpha=0.1)
    plt.grid(True)
    plt.show()


def test2():
    np.set_printoptions(suppress=True, linewidth=200)

    ntrailers = 5

    pose1 = [
        0., 0., 0., 0.0, 0., 0., 0.
    ]

    # pose2 = [
    #     5., 5., 0, 0, 0, 0, 0
    # ]

    pose2 = [
        5., 8., 0., 1.57, 1.57, 1.57, 1.57
    ]

    out1, out2 = get_flat_boundary_values(pose1, pose2)

    m,_ = out1.shape
    neqs = 2*m
    A = np.zeros((neqs, neqs))
    for i in range(0, m):
        A[i,i] = factorial(i)

    for i in range(0, m):
        for j in range(i, 2*m):
            A[m+i,j] = factorial(j) / factorial(j - i)

    B = np.zeros((neqs, 2))
    for i in range(0, m):
        B[i,:] = out1[i,:]
        B[m+i,:] = out2[i,:]

    coefs = np.linalg.solve(A, B)

    t = np.linspace(0, 1, 600)
    nt = len(t)
    exprs = get_maps(ntrailers)

    nderivatives,_ = exprs['output'].shape

    args = exprs['output'].T.reshape((-1,1))
    # u1 = Function('u1', [args], [exprs['u1']])
    # u2 = Function('u2', [args], [exprs['u2']])
    phi = Function('phi', [args], [exprs['phi']])
    theta0 = Function('theta0', [args], [exprs['theta0']])
    theta1 = Function('theta1', [args], [exprs['theta1']])
    theta2 = Function('theta2', [args], [exprs['theta2']])
    theta3 = Function('theta3', [args], [exprs['theta3']])
    theta4 = Function('theta4', [args], [exprs['theta4']])
    x = Function('x', [args], [exprs['x']])
    y = Function('y', [args], [exprs['y']])

    theta0_vals = np.zeros(nt)
    theta1_vals = np.zeros(nt)
    theta2_vals = np.zeros(nt)
    theta3_vals = np.zeros(nt)
    theta4_vals = np.zeros(nt)
    x_vals = np.zeros(nt)
    y_vals = np.zeros(nt)
    phi_vals = np.zeros(nt)

    for i in range(nt):
        output = np.zeros((nderivatives, 2))
        output[0,:] = polyval(t[i], coefs)

        for j in range(1, nderivatives):
            p = polyder(coefs,j)
            output[j,:] = polyval(t[i], p)

        a = np.reshape(output, (-1,))
        theta0_vals[i] = theta0(a)
        theta1_vals[i] = theta1(a)
        theta2_vals[i] = theta2(a)
        theta3_vals[i] = theta3(a)
        theta4_vals[i] = theta4(a)
        phi_vals[i] = phi(a)
        x_vals[i] = x(a)
        y_vals[i] = y(a)

    traj = {
        'ntrailers': ntrailers,
        't': t,
        'phi': phi_vals,
        'x0': x_vals,
        'y0': y_vals,
        'theta0': theta0_vals,
        'theta1': theta1_vals,
        'theta2': theta2_vals,
        'theta3': theta3_vals,
        'theta4': theta4_vals
    }

    np.save('/tmp/traj.npy', traj)


if __name__ == '__main__':
    test2()
