from casadi import SX, DM, horzcat, vertcat, Function, \
    arctan2, sin, cos, tan, jacobian, jtimes, rootfinder, \
    nlpsol, norm_2, substitute, sum1, solve, det, pinv
import numpy as np
import scipy as sp


def sx_vec(*args):
    n = len(args)
    v = SX.zeros(n)
    for i in range(n):
        v[i] = args[i]
    return v


def get_maps(ntrailers):
    '''
        returns expressions for state variables
        x, y, phi, theta...
        as functions of outputs
        y0, y1, Dy0, Dy1, ...
    '''
    out = SX.zeros(ntrailers + 3, 2)
    out[0,0] = SX.sym('x%d' % ntrailers)
    out[0,1] = SX.sym('y%d' % ntrailers)

    for i in range(1, ntrailers + 3):
        out[i,0] = SX.sym('D%dx%d' % (i, ntrailers))
        out[i,1] = SX.sym('D%dy%d' % (i, ntrailers))

    args = out[1:-1,:].T.reshape((-1,1))
    dargs = out[2:,:].T.reshape((-1,1))

    # 1. find all thetas
    p = out[0,:].T
    v = sx_vec(args[0], args[1])
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

        p = p_next + sx_vec(cos(theta_next), sin(theta_next))
        v = v_next + sx_vec(-sin(theta_next), cos(theta_next)) * dtheta_next
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
    p0 = positions[0]
    v0 = velocities[0]
    dtheta0 = dthetas[0]
    theta0 = thetas[0]
    phi = arctan2(dtheta0, cos(theta0) * v0[0] + sin(theta0) * v0[1])

    # 3. find controls
    u1 = v0.T @ sx_vec(cos(theta0), sin(theta0))
    u2 = jtimes(phi, args, dargs)

    names = ['theta%d' % i for i in range(ntrailers)]

    result = {
        'u1': u1,
        'u2': u2,
        'output': out,
        'x': p0[0],
        'y': p0[1],
        'phi': phi
    }
    result.update(dict(zip(names, thetas)))
    return result

def gen_equations(ntrailers):
    out = SX.sym('y', 2 * (ntrailers + 1))
    args = out[:-2]
    dargs = out[2:]

    # vn, thetan, dthetan
    v = SX.zeros(2)
    v[0] = args[0]
    v[1] = args[1]
    theta = arctan2(v[1], v[0])
    dtheta = jtimes(theta, args, dargs)
    thetas = [theta]
    velocities = [v]
    dthetas = [dtheta]

    for i in range(ntrailers - 1):
        v_next = v
        theta_next = theta
        dtheta_next = dtheta

        q = SX.zeros(2)
        q[0] = -sin(theta_next)
        q[1] = cos(theta_next)
        v = v_next + q * dtheta_next
        theta = arctan2(v[1], v[0])
        dtheta = jtimes(theta, args, dargs)
        thetas += [theta]
        velocities += [v]
        dthetas += [dtheta]

    velocities = velocities[::-1]
    dthetas = dthetas[::-1]
    thetas = thetas[::-1]

    v0 = velocities[0]
    dtheta0 = dthetas[0]
    theta0 = thetas[0]
    phi = arctan2(dtheta0, cos(theta0) * v0[0] + sin(theta0) * v0[1])

    return phi, thetas, velocities, out


def get_flat_boundary_values(pose1, pose2):
    R'''
        for given initial and final poses (`pose` = [`x`, `y`, `phi`, `theta_1`, `theta_2`, ...])
        finds initial output [y1, Dy1, D2y1, ...] and final output
    '''
    x1 = pose1[0]
    y1 = pose1[1]
    phi1 = pose1[2]
    thetas1 = pose1[3:]
    ntrailers = len(thetas1)

    x2 = pose2[0]
    y2 = pose2[1]
    phi2 = pose2[2]
    thetas2 = pose2[3:]

    phi_expr, thetas_expr, velocities_expr, out_var = gen_equations(ntrailers)

    # 1. solve for pose1
    eqs = [norm_2(velocities_expr[0]) - 1]
    eqs += [theta_expr - theta for theta_expr,theta in zip(thetas_expr, thetas1)]
    eqs += [phi_expr - phi1]
    prob = {
        'f': out_var[2:].T @ out_var[2:],
        'x': out_var,
        'g': vertcat(*eqs)
    }
    nvars,_ = out_var.shape
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=np.ones(nvars), lbg=-1e-5, ubg=1e-5)    
    out1 = np.concatenate([
        [[x1 - sum(np.cos(thetas1[1:])), y1 - sum(np.sin(thetas1[1:]))]],
        np.reshape(sol['x'], (-1, 2))
    ], axis=0)

    # 2. solve for pose2
    eqs = [norm_2(velocities_expr[0]) - 1]
    eqs += [theta_expr - theta for theta_expr,theta in zip(thetas_expr, thetas2)]
    eqs += [phi_expr - phi2]
    prob = {
        'f': out_var[2:].T @ out_var[2:],
        'x': out_var,
        'g': vertcat(*eqs)
    }
    nvars = out_var.shape
    solver = nlpsol('solver', 'ipopt', prob)
    sol = solver(x0=np.ones(nvars), lbg=-1e-5, ubg=1e-5)    
    out2 = np.concatenate([
        [[x2 - sum(np.cos(thetas2[1:])), y2 - sum(np.sin(thetas2[1:]))]],
        np.reshape(sol['x'], (-1, 2))
    ], axis=0)

    return out1, out2


def solve_linear(expr, args):
    nargs,_ = args.shape
    f = Function('f', [args], [expr])
    z = SX.zeros(nargs)
    fz = f(z)
    cols = []
    for i in range(nargs):
        e = SX.zeros(nargs)
        e[i] = 1
        col = f(e) - fz
        cols += [col]

    A = np.array(DM(horzcat(*cols)))
    B = np.reshape(DM(-fz), (-1,))
    x,_,_,_ = np.linalg.lstsq(A, B, rcond=-1)
    x = np.reshape(x, (-1,))

    print(A @ x - B)

    return x


def get_flat_values(v, p, phi, thetas):
    R'''
        `v` is the absolute value of the velocity of the car
        `p` is the position of the car
        `phi` is steering angle
    '''
    ntrailers = len(thetas)
    positions = np.zeros((ntrailers, 2))
    positions[0] = p
    velocities = np.zeros((ntrailers, 2))
    velocities[0,0] = v * np.cos(thetas[0])
    velocities[0,1] = v * np.sin(thetas[0])
    dthetas = np.zeros(ntrailers)

    # 1. find positions of all the trailers
    for i in range(1, ntrailers):
        positions[i] = positions[i-1] - np.array([np.cos(thetas[i]), np.sin(thetas[i])])

    # 2. find velocities and angular velocities of all the trailers
    for i in range(1, ntrailers):
        A = np.array([
            [1, 0, -np.sin(thetas[i])],
            [0, 1, np.cos(thetas[i])],
            [np.sin(thetas[i]), -np.cos(thetas[i]), 0]
        ])
        B = np.array([
            [velocities[i-1,0]],
            [velocities[i-1,1]],
            [0]
        ])
        print('det A =', np.linalg.det(A))
        ans,_,_,_ = np.linalg.lstsq(A, B, rcond=None)
        velocities[i] = ans[0:2,0]
        dthetas[i] = ans[2,0]
    
    # 3. find output derivaties
    out_derivs, vel_exprs = get_velocities(ntrailers)
    out_deriv_values = np.zeros(out_derivs.shape)

    for i in range(ntrailers):
        eq = velocities[i] - vel_exprs[i]
        for j in range(i):
            eq = substitute(eq, out_derivs[j,:].T, out_deriv_values[j,:])
        vals = solve_linear(eq, out_derivs[i,:].T)
        out_deriv_values[i,:] = vals

    # eq = velocities[0] - exprs[0]
    # D1 = solve_linear(eq, args[0:2])
    # print(D1)

    # eq = velocities[1] - exprs[1]
    # eq = substitute(eq, args[0:2], D1)
    # D2 = solve_linear(eq, args[2:4])
    # print(D2)

    # eq = velocities[2] - exprs[2]
    # eq = substitute(eq, args[0:2], D1)
    # eq = substitute(eq, args[2:4], D2)
    # D3 = solve_linear(eq, args[4:6])
    # print(D3)


def get_velocities(ntrailers):
    out_derivs = SX.zeros(ntrailers, 2)

    for i in range(ntrailers):
        out_derivs[i, 0] = SX.sym('D%dx%d' % (i+1, ntrailers))
        out_derivs[i, 1] = SX.sym('D%dy%d' % (i+1, ntrailers))

    args = out_derivs[:-1,:].T.reshape((-1,1))
    dargs = out_derivs[1:,:].T.reshape((-1,1))

    theta = arctan2(out_derivs[0,1], out_derivs[0,0])
    v = out_derivs[0,:].T

    velocities = [v]
    thetas = [theta]

    for i in range(ntrailers-1, -1, -1):
        vnext = velocities[-1]
        thetanext = thetas[-1]
        dthetanext = jtimes(thetanext, args, dargs)
        v = vnext + sx_vec(-sin(thetanext), cos(thetanext)) * dthetanext
        theta = arctan2(v[1], v[0])
        velocities += [v]
        thetas += [theta]

    return out_derivs, velocities


if __name__ == '__main__':
    get_flat_values(-1., [0., 0.], 0., [0.5, 0.3, 0.8])
