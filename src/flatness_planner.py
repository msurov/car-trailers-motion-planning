from dynamics import Dynamics
from bezier.bezier import interpolate, eval_bezier, eval_bezier_length
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import factorial
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from flat_coordinates import eval_flat_derivs, get_flat_maps
from numpy.polynomial.polynomial import polyval, polyder
from casadi import Function, vcat, DM
import tempfile
from os.path import join


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


def fix_conitnuity(arr, period):
    helfperiod = period/2
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] > helfperiod:
            arr[i] -= period
        elif arr[i] - arr[i-1] < -helfperiod:
            arr[i] += period


def make_poly_bv(left_bv, right_bv):
    '''
        Find coefficients of the polynomial P(x) s.t.
        P(0) = left_bv[0]
        P'(0) = left_bv[1]
        ...
        P(1) = right_bv[0]
        P'(1) = right_bv[1]
        ...
    '''
    n = len(left_bv)
    assert n == len(right_bv)
    neqs = 2*n

    A = np.zeros((neqs, neqs))
    for i in range(0, n):
        A[i,i] = factorial(i)

    for i in range(0, n):
        for j in range(i, neqs):
            A[n+i,j] = factorial(j) / factorial(j - i)

    B = np.zeros(neqs)
    B[0:n] = left_bv
    B[n:2*n] = right_bv

    return np.linalg.solve(A, B)


def get_poly_derivatives(x, poly, nder):
    xshape = np.shape(x)
    result = np.zeros((nder + 1,) + xshape)
    for k in range(0, nder + 1):
        result[k,...] = polyval(x, polyder(poly, k))
    return result


def interpolate_waypoints(waypts, k=5):
    waypts = np.array(waypts)
    bzr = interpolate(waypts, k)
    sp = bezier2bspline(bzr)
    return sp


def test_trajectory(traj):
    ntrailers = traj['ntrailers']
    t = traj['t']
    x = traj['x']
    y = traj['y']
    phi = traj['phi']
    theta = np.array(traj['theta'])
    u1 = traj['u1']
    u2 = traj['u2']
    fu = make_interp_spline(t, np.array([u1, u2]).T, k=3)

    dynamics = Dynamics(ntrailers)

    def rhs(t, state):
        u = fu(t)
        r = dynamics.rhs(*state, *u)
        r = np.reshape(r, (-1,))
        return r

    state0 = [x[0], y[0], phi[0]] + list(theta[:,0])
    ans = solve_ivp(rhs, [t[0], t[-1]], state0, t_eval=t, max_step=1e-3)

    plt.gca().set_prop_cycle(None)
    plt.plot(ans.y[0], ans.y[1])
    plt.gca().set_prop_cycle(None)
    plt.plot(x, y, '--')
    plt.show()


def trajectory_through_waypoints(waypts, ntrailers, reverse=False):
    R'''
        `waypts` is an array of Cartesian waypoints to follow through \
    '''
    assert ntrailers > 0
    delta = np.pi if reverse else 0

    sp = interpolate_waypoints(waypts, 7)
    t = np.linspace(0, 1, 1000)
    poly = make_poly_bv([sp.t[0], 0], [sp.t[-1], 0])
    s = polyval(t, poly)

    # train N
    p = sp(s, 0)
    v = sp(s, 1)
    theta = np.arctan2(v[:,1], v[:,0]) + delta
    fix_conitnuity(theta, 2*np.pi)
    trailer = (sp, p, v, theta)
    trailers = [trailer]

    for i in range(1, ntrailers):
        prev_trailer = trailers[-1]
        _, prev_p, prev_v, prev_theta = prev_trailer
        p = prev_p + np.array([np.cos(prev_theta), np.sin(prev_theta)]).T
        sp = make_interp_spline(s, p, 5)
        v = sp(s, 1)
        theta = np.arctan2(v[:,1], v[:,0]) + delta
        fix_conitnuity(theta, 2*np.pi)
        trailer = (sp, p, v, theta)
        trailers += [trailer]

    trailers = trailers[::-1]
    _, p, v, theta = trailers[0]
    
    u1 = np.cos(theta) * v[:,0] + np.sin(theta) * v[:,1]
    stheta = make_interp_spline(s, theta, 5)
    dtheta = stheta(s, 1)
    phi = np.arctan2(dtheta, u1) + delta
    fix_conitnuity(phi, 2*np.pi)
    fphi = make_interp_spline(s, phi, 5)
    u2 = fphi(s, 1)

    traj = {
        'ntrailers': ntrailers,
        't': t, 
        'x': p[:,0],
        'y': p[:,1],
        'phi': phi,
        'theta': np.array([tr[3] for tr in trailers]),
        'u1': u1,
        'u2': u2
    }

    return traj


def trajectory_2pts(initial_state, final_state):
    '''
        Find point-to-point trajectory
    '''
    assert np.shape(initial_state) == np.shape(final_state)
    ntrailers = len(initial_state) - 3
    flat = get_flat_maps(ntrailers)

    args = flat.flat_derivs.reshape((-1,1))
    state_expr = vcat((
        flat.positions[0][0], # x
        flat.positions[0][1], # y
        flat.phi,
        *flat.theta
    ))
    inp_expr = vcat((
        flat.u1,
        flat.u2,
    ))
    state_fun = Function('state', [args], [state_expr])
    inp_fun = Function('input', [args], [inp_expr])

    flat_val1 = eval_flat_derivs(initial_state, flat)
    flat_val2 = eval_flat_derivs(final_state, flat)

    poly1 = make_poly_bv(flat_val1[:,0], flat_val2[:,0])
    poly2 = make_poly_bv(flat_val1[:,1], flat_val2[:,1])

    t = np.linspace(0, 1, 1000)
    poly = make_poly_bv([0, 0], [1, 0])
    s = polyval(t, poly)
    nderivs = flat.flat_derivs.shape[0] - 1
    out1_vals = get_poly_derivatives(s, poly1, nderivs)
    out2_vals = get_poly_derivatives(s, poly2, nderivs)
    out_vals = np.concatenate([out1_vals, out2_vals])

    state_vals = state_fun(out_vals)
    state_vals = np.array(state_vals.T, float)
    for i in range(ntrailers):
        fix_conitnuity(state_vals[:,3+i], 2*np.pi)

    inp_vals = inp_fun(out_vals)
    inp_vals = np.array(inp_vals.T, float)
    traj = {
        'ntrailers': ntrailers,
        't': t, 
        'x': state_vals[:,0],
        'y': state_vals[:,1],
        'phi': state_vals[:,2],
        'theta': state_vals[:,3:].T,
        'u1': inp_vals[:,0],
        'u2': inp_vals[:,1]
    }
    return traj


def test1():
    st1 = [0, 0, 0, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
    st2 = [7, -2, 0, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]

    traj = trajectory_2pts(st1, st2)
    test_trajectory(traj)
    np.save(join(tempfile.gettempdir(), 'traj-1.npy'), traj)


def reverse_trajectory(traj):
    return {
        'x': traj['x'][::-1],
        'y': traj['y'][::-1],
        'phi': traj['phi'][::-1],
        'theta': traj['theta'][:,::-1],
        'u1': -traj['u1'][::-1],
        'u2': -traj['u2'][::-1],
    }


def test2():
    waypts = [
        [0, 0],
        [5, 5],
        [10, 0],
        [15, 5],
        [20, 0],
    ]
    traj = trajectory_through_waypoints(waypts, 3)
    test_trajectory(traj)
    np.save(join(tempfile.gettempdir(), 'traj-2.npy'), traj)


if __name__ == '__main__':
    # test1()
    test2()
