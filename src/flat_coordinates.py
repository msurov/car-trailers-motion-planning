from casadi import SX, DM, horzcat, vertcat, Function, \
    arctan2, sin, cos, tan, jacobian, jtimes, rootfinder, \
    nlpsol, norm_2, substitute, sum1, solve, det, pinv
import numpy as np
import scipy as sp
from collections import namedtuple


def sxvec(*args):
    n = len(args)
    v = SX.zeros(n)
    for i in range(n):
        v[i] = args[i]
    return v


def make_tuple(name : str, **kwargs):
    T = namedtuple(name, kwargs.keys())
    return T(*kwargs.values())


def get_flat_maps(ntrailers):
    '''
        returns expressions for state variables [x, y, phi, theta...]
        as functions of outputs [y0, y1, Dy0, Dy1, ...]
    '''
    out = SX.zeros(ntrailers + 3, 2)
    out[0,0] = SX.sym(f'x{ntrailers - 1}')
    out[0,1] = SX.sym(f'y{ntrailers - 1}')

    for i in range(1, ntrailers + 3):
        out[i,0] = SX.sym(f'D{i}x{ntrailers-1}')
        out[i,1] = SX.sym(f'D{i}y{ntrailers-1}')

    args = out[1:-1,:].T.reshape((-1,1))
    dargs = out[2:,:].T.reshape((-1,1))

    # 1. find all thetas
    p = out[0,:].T
    v = sxvec(args[0], args[1])
    theta = arctan2(v[1], v[0])
    dtheta = jtimes(theta, args, dargs)
    thetas = [theta]
    velocities = [v]
    positions = [p]
    dthetas = [dtheta]

    for i in range(ntrailers - 1):
        p_next = p
        v_next = v
        theta_next = theta
        dtheta_next = dtheta

        p = p_next + sxvec(cos(theta_next), sin(theta_next))
        v = v_next + sxvec(-sin(theta_next), cos(theta_next)) * dtheta_next
        theta = arctan2(v[1], v[0])
        dtheta = jtimes(theta, args, dargs)

        thetas += [theta]
        velocities += [v]
        dthetas += [dtheta]
        positions += [p]

    positions = positions[::-1]
    velocities = velocities[::-1]
    dthetas = dthetas[::-1]
    thetas = thetas[::-1]

    # 2. find phi
    v0 = velocities[0]
    dtheta0 = dthetas[0]
    theta0 = thetas[0]
    phi = arctan2(dtheta0, cos(theta0) * v0[0] + sin(theta0) * v0[1])

    # 3. find controls
    u1 = v0.T @ sxvec(cos(theta0), sin(theta0))
    u2 = jtimes(phi, args, dargs)

    return make_tuple(
        'FlatMaps',
        flat_out = out[0],
        flat_derivs = out,
        u1 = u1,
        u2 = u2,
        phi = phi,
        theta = thetas,
        velocities = velocities,
        positions = positions
    )


def eval_flat_derivs(pose, maps):
    R'''
        Find flat variable and its derivatives corresponding the given pose
    '''
    x,y,phi,*thetas = pose
    ntrailers = len(thetas)
    eqs = [maps.theta[i] - thetas[i] for i in range(ntrailers)]
    eqs += [
        maps.positions[0][0] - x,
        maps.positions[0][1] - y,
        maps.phi - phi,
        maps.velocities[0].T @ maps.velocities[0] - 1
    ]
    decision_variable = maps.flat_derivs.T.reshape((-1,1))
    prob = {
        'f': decision_variable.T @ decision_variable,
        'x': decision_variable,
        'g': vertcat(*eqs)
    }
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=np.ones(np.shape(decision_variable)), lbg=-1e-5, ubg=1e-5)    
    flat_drivs_value = substitute(maps.flat_derivs, decision_variable, sol['x'])
    flat_drivs_value = DM(flat_drivs_value)
    flat_drivs_value = np.array(flat_drivs_value, float)
    return flat_drivs_value


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    maps = get_flat_maps(2)
    pose1 = [0, 0, 0, 0, 0]
    flat_val1 = eval_flat_derivs(pose1, maps)
