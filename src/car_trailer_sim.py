from car_trailer_kinematics import CarTrailerKinematic
from casadi import SX, jtimes, jacobian, DM, sin, cos, tan, Function, vertcat
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def get_dynamics(cfg : CarTrailerKinematic):
    L1 = cfg.wheel_base
    L2 = cfg.trailer_base
    d = cfg.pivot_distance

    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    psi = SX.sym('psi')
    v = SX.sym('v')
    phi = SX.sym('phi')

    dx = cos(theta) * v
    dy = sin(theta) * v
    dtheta = tan(phi) * v / L1
    dpsi = -(sin(psi) / L2 + (cos(psi) * d / L2 + 1) * tan(phi) / L1) * v

    dst = vertcat(dx, dy, dtheta, dpsi)
    st = vertcat(x, y, theta, psi)
    inp = vertcat(v, phi)

    dstfun = Function('rhs', [st, inp], [dst])
    return dstfun


if __name__ == '__main__':
    cfg = CarTrailerKinematic(
        1.4, 1.8, 0.2, 1.0
    )
    dynamics = get_dynamics(cfg)

    def control_inp(t):
        v = 5 * np.ones(np.shape(t))
        phi = 0.4 * np.sign(np.cos(0.75*t))
        return v, phi

    def rhs(t, st):
        inp = control_inp(t)
        dst = dynamics(st, inp)
        return np.reshape(dst, (-1,))

    st0 = np.zeros(4)
    t_eval = np.arange(0, 10, 3e-2)
    ans = solve_ivp(rhs, [0, 10], st0, max_step=1e-2, t_eval=t_eval)
    v,phi = control_inp(ans.t)
    traj = {
        't': ans.t,
        'x': ans.y[0],
        'y': ans.y[1],
        'theta': ans.y[2],
        'psi': ans.y[3],
        'v': v,
        'phi': phi
    }
    plt.plot(traj['x'], traj['y'])
    plt.axis('equal')
    plt.show()
    data = {
        'cfg': cfg,
        'trajectory': traj
    }
    np.save('data/data.npy', data, allow_pickle=True)
