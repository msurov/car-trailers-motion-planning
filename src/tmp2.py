from casadi import *

g = 9.81
L = 0.2
m = 1
mcart = 0.5

T = 2
N = 160 # number of control intervals
dt = T/N # length of 1 control interval [s]

# System is composed of 4 states
nx = 4

degree = 4
method = 'radau';

'''
    continuous system dot(x)=f(x,u)
    Construct a CasADi function for the ODE right-hand side
'''

x = MX.sym('x', nx) # pos, theta, dpos, dtheta
u = MX.sym('u') # control force [N]
ddpos = ((u+m*L*x[3]*x[3]*sin(x[1])-m*g*sin(x[1])*cos(x[1]))/(mcart+m-m*cos(x[1])*cos(x[1])))
rhs = vertcat(
    x[2],
    x[3],
    ddpos,
    g/L*sin(x[1])-cos(x[1])*ddpos
)

# Continuous system dynamics as a CasADi Function
f = Function('f', [x, u], [rhs])

'''
    Discrete system x_next = F(x,u)
'''

# differential equation
dae = {
    'x': x, 'p': u, 'ode': f(x,u)
}

# integrator options
iop = {}
iop['number_of_finite_elements'] = 1
iop['tf'] = dt / iop['number_of_finite_elements'];

# Reference Runge-Kutta implementation
intg = integrator(
    'intg', 'rk', dae, iop
)

res = intg(x0=x, p=u)

# % Discretized (sampling time dt) system dynamics as a CasADi Function
F = Function('F', [x, u], [res['xf']])

'''
    Optimal control problem, multiple shooting
'''



# % Path constraints
# opti.subject_to(-3  <= pos <= 3);
# opti.subject_to(-1.2 <= U   <= 1.2);

# % Initial and terminal constraints
# opti.subject_to(X(:,1)==[1;0;0;0]);
# opti.subject_to(X(:,end)==[0;0;0;0]);

# % Objective: regularization of controls
# opti.minimize(sumsqr(U));

# % solve optimization problem
# opti.solver('ipopt')

# sol = opti.solve();

# %%
# % -----------------------------------------------
# %    Post-processing: plotting
# % -----------------------------------------------

# pos_opt = sol.value(pos);
# theta_opt = sol.value(theta);
# dpos_opt = sol.value(dpos);
# dtheta_opt = sol.value(dtheta);

# u_opt = sol.value(U);

# % time grid for printing
# tgrid = linspace(0,T, N+1);

# figure;
# subplot(3,1,1)
# hold on
# plot(tgrid, theta_opt, 'b')
# plot(tgrid, pos_opt, 'b')
# legend('theta [rad]','pos [m]')
# xlabel('Time [s]')
# subplot(3,1,2)
# hold on
# plot(tgrid, dtheta_opt, 'b')
# plot(tgrid, dpos_opt, 'b')
# legend('dtheta [rad/s]','dpos [m/s]')
# xlabel('Time [s]')
# subplot(3,1,3)
# stairs(tgrid(1:end-1), u_opt, 'b')
# legend('u [m/s^2]')
# xlabel('Time [s]')

# cart = [pos;0*pos];
# ee   = [pos+L*sin(theta);L*cos(theta)];

# cart_sol = sol.value(cart);
# ee_sol   = sol.value(ee);

# figure
# hold on
# for k=1:8:N+1
#     line([cart_sol(1,k) ee_sol(1,k)],[cart_sol(2,k) ee_sol(2,k)],'LineWidth',1)
# end
# axis equal
