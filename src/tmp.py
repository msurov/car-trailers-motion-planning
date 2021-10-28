from casadi import DM, SX, horzcat, vertcat, Function, \
    arctan2, sin, cos, tan, jacobian, jtimes, rootfinder, nlpsol, norm_2, substitute, sum1


a = SX.sym('a')


def sx_vec(*args):
    n = len(args)
    v = SX.zeros(n)
    for i in range(n):
        v[i] = args[i]
    return v

print(sx_vec(sin(a), cos(a)))
