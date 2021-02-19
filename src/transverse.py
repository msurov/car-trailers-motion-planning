from casadi import SX, horzcat, substitute, jtimes, norm_2, Function
import numpy as np
from dynamics import Dynamics


P1 = np.array([
    [ 0, 1,  0, 0],
    [-1, 0,  0, 0],
    [ 0, 0,  0, 1],
    [ 0, 0, -1, 0]

])

P2 = np.array([
    [ 0, 0,  1, 0],
    [ 0, 0,  0,-1],
    [-1, 0,  0, 0],
    [ 0, 1,  0, 0]
])

P3 = np.array([
    [ 0, 0,  0, 1],
    [ 0, 0,  1, 0],
    [ 0,-1,  0, 0],
    [-1, 0,  0, 0]
])


def get_perp(v):
    return horzcat(P1 @ v, P2 @ v, P3 @ v)


def get_linsys(dynamics):
    R'''
        Finds the matrix functions A,B of linearized transverse dynamics
    '''
    f = dynamics.f
    g = dynamics.g
    x = dynamics.state

    n,_ = x.shape
    m,_ = dynamics.u.shape

    u_star = SX.sym('u_star', m)
    du_star = SX.sym('du_star', m)
    x_star = SX.sym('x_star', n)

    dx_star = substitute(f + g @ u_star, x, x_star)
    ddx_star = substitute(
        jtimes(f, x, dx_star) + jtimes(g, x, dx_star) @ u_star + g @ du_star, 
        x, x_star
    )
    v = dx_star / norm_2(dx_star)
    dv = (SX.eye(4) - v @ v.T) @ ddx_star / norm_2(dx_star)
    E = get_perp(v)
    dE = get_perp(dv)
    A = dE.T @ E + E.T @ substitute(jtimes(f, x, E), x, x_star)
    B = E.T @ substitute(g, x, x_star)
    A = Function('A', [x_star, u_star, du_star], [A])
    B = Function('B', [x_star, u_star], [B])

    return A, B


if __name__ == '__main__':
    from misc.format.serialize import load_dict
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline

    traj = load_dict('data/traj.npz')
    t = traj['t']
    x = np.array([traj['x0'], traj['y0'], traj['phi'], traj['theta0']]).T
    u = np.array([traj['u1'], traj['u2']]).T
    su = make_interp_spline(t, u)
    du = su(t, 1)
    nt = len(t)

    d = Dynamics(1)
    A,B = get_linsys(d)

    Avals = np.zeros((nt, 3, 3))
    Bvals = np.zeros((nt, 3, 2))

    for i in range(nt):
        Avals[i,:,:] = A(x[i,:], u[i,:], du[i,:])
        Bvals[i,:,:] = B(x[i,:], u[i,:])
    
    plt.plot(t, Avals[:,0,:])
    plt.plot(t, Avals[:,1,:])
    plt.plot(t, Avals[:,2,:])
    plt.show()
        
