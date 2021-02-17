import casadi as ca



def product(f, i1, i2):
    P = 1
    for i in range(i1, i2+1):
        P = P * f(i)
    return P


def sum(f, i1, i2):
    S = 0
    for i in range(i1, i2+1):
        S = S + f(i)
    return S


class Dynamics:
    R'''
        The first car has 4 wheels;
        `phi` is the angle of the steering wheels (wrt x-axis of the first wheel) \
        `thetaI` is the orientation of I-th trailer \
        `x,y` are thr cartesian coordinates of the centre of car's forward wheels \
    '''

    def __init__(self, ntrailers):
        nstates = 3 + ntrailers
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        phi = ca.SX.sym('phi')
        thetas = [ca.SX.sym('theta%d' %i) for i in range(ntrailers)]

        self.ntrailers = ntrailers
        self.state = [x,y,phi] + thetas
        self.x = x
        self.y = y
        self.phi = phi
        self.thetas = thetas

        g1 = ca.SX.zeros((nstates, 1))
        g1[0] = ca.cos(thetas[0])
        g1[1] = ca.sin(thetas[0])

        if ntrailers > 0:
            g1[3] = 1 * ca.tan(phi)

        for i in range(1, ntrailers):
            fun = lambda j: ca.cos(thetas[j-1] - thetas[j])
            g1[3+i] = product(fun, 1, i-1) * ca.sin(thetas[i-1] - thetas[i])
        
        g2 = ca.SX.zeros((nstates, 1))
        g2[2] = 1

        self.g = ca.horzcat(g1, g2)
        self.f = ca.SX.zeros((nstates, 1))

        u1 = ca.SX.sym('u1')
        u2 = ca.SX.sym('u2')
        self.rhs = ca.Function('RHS', [x, y, phi] + thetas + [u1, u2], [g1 * u1 + g2 * u2])


    def trailer_position(self, i):
        assert i >= 0 and i < self.ntrailers
        x = self.x - sum(lambda j: ca.cos(self.thetas[j]), 1, i)
        y = self.y - sum(lambda j: ca.sin(self.thetas[j]), 1, i)
        return x,y
