from casadi import SX, horzcat, vertcat, Function, \
    arctan2, sin, cos, tan, jacobian, jtimes, rootfinder, nlpsol, norm_2, substitute
import numpy as np


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


def find_flat_values(pose1, pose2):
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
    out1_val = sol['x']

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
    out2_val = sol['x']

    print(out1_val)
    print(out2_val)


def testsolve():
    x = SX.sym('x')
    y = SX.sym('y')
    
    prob = {
        'f': 0,
        'x': vertcat(x, y),
        'g': vertcat(tan(x) - y, y - 1)
    }
    solver = nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=[0., 0.], lbx=-10., ubx=10., lbg=-1e-5, ubg=1e-5)
    print(sol['x'])

    # solver = rootfinder('RF', 'newton', f)
    # ans = solver(0.7, 1.)
    # print(ans, f(*ans))

find_flat_values([0, 0, 0.0, 0.1, 0.1], [1, 2, 0.0, 0.1, 0.1])
