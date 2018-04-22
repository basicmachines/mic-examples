import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec

# Constants for both Odeint and GEKKO model

Mw = 10    # mass of wheel
Mr = 2     # mass of robot
R  = 4     # radius of wheel
L  = 6     # length from end to center of mass of the rod
Iw = 1/2*Mw*R**2        # Inertia of wheel
Ir = 1/3*Mr*(2*L)**2    # Inertia of robot
G  = 9.81  # gravity (9.81 m/s)
Fw = 0.0   # Friction, wheel, opposing θw_dot
Fr = 0.0   # Friction, robot, opposing θr_dot

# Initial conditions for variables
θr_init, θr_dot_init = [0.0, 0.0]
xr_init, xr_dot_init = [0.0, 0.0]
θw_init, θw_dot_init = [0.0, xr_dot_init/R]
u_init = 0.0
xw_init = xr_init - L*np.sin(θr_init)

def modelDerivFunc(xz, tz, u):

    [θr, θr_dot, θw, θw_dot] = xz # unpack inputs.

    A = np.array([
                 [Mr*R*L*np.cos(θr), (Iw + (Mw + Mr)*R**2)],
                 [(Ir + Mr*L**2)   ,  Mr*R*L*np.cos(θr)]
                 ])

    B = np.array([
                 [Mr*R*L*θr_dot**2*np.sin(θr) + u - θr_dot*Fr],
                 [Mr*G*L*np.sin(θr)        - u - θw_dot*Fw]
                 ])

    # Solve for X where AX=B
    X = np.linalg.solve(A, B)

    [θr_ddot, θw_ddot] = X # un-pack X

    return [θr_dot, θr_ddot, θw_dot, θw_ddot]

# Init model and set time values
m = GEKKO()

#tend = 6.0
#endMotionTime = tend
m.time = np.linspace(0.0, 0.35, 11)

#tfinal = np.zeros(len(m.time))
#for i in range(len(m.time)):
#    if m.time[i] >= tend:
#        tfinal[i] = 1
#m.tf = m.Param(value=tfinal)

# Setup model

# Parameters
m.mw = m.Param(name='mass_wheel', value=Mw)
m.mr = m.Param(name='mass_robot', value=Mr)
m.l = m.Param(name='length_cm', value=L)
m.R = m.Param(name='radius_wheel', value=R)
m.g = m.Param(name='gravity', value=G)

# Fixed Variables
m.Iw = m.FV(name='Inertia_wheel', value=Iw)  # TODO: What is an FV for?
m.Ir = m.FV(name='Inertia_robot', value=Ir)

# State Variables - Wheel angle not controlled but torque of motor
m.phi = m.SV(name='angle_wheel')
m.phi_d = m.SV(name='angle_wheel_dot', value=0) # TODO: Needs to be set
m.phi_dd = m.SV(name='angle_wheel_dotdot', value=0) # TODO: Needs to be set
m.xw = m.Var(name='xWheel', value=0)

# Controlled variables
m.xr = m.CV(name='xRobot', value=xr_init)
# m.xr_d = m.CV(name='x_position_wheel_dot', value=xr_dot_init)   # TODO: Jim has no CV for this
#m.theta = m.CV(name='angle_robot', value=θr_init)
m.theta = m.CV(name='angle_robot', lb=-np.pi/4, ub=np.pi/4)   # TODO: Jim's
m.theta_d = m.SV(name='angle_robot_dot', value=θr_dot_init)   # TODO: What is SV?
m.theta_dd = m.SV(name='angle_robot_dotdot', value=0)

# Manipulated variables
m.tau = m.MV(name='torque', value=u_init, lb=-1000, ub=1000)

# Define parameter options
m.tau.STATUS = 1
m.tau.DCOST = 0

# add (STATUS=1) setpoint to obj function
m.xr.STATUS = 1
m.theta.STATUS = 1
#m.theta_d.STATUS = 1   # TODO: Jim does not have this
#m.xr_d.STATUS = 1      # TODO: Jim does not have this

def new_setpoint(var, value, weight=None):
    var.SPHI = value
    var.SPLO = value
    if weight is not None:
        var.WSP = weight

# setpoints for robot angle to 0
new_setpoint(m.theta, 0, weight=2)
#new_setpoint(m.theta_d, 0, weight=1)       # TODO: Jim does not set this...
#new_setpoint(m.theta_dd, 0, weight=10)

# setpoints for robot position to 0
#new_setpoint(m.xr, 0)                  # TODO: Do we need a weight?
#new_setpoint(m.xr_d, 0, weight=1)
#new_setpoint(m.xr_dd, 0, weight=10)

# set FSTATUS = 1 to follow measurement
m.phi.FSTATUS = 1
m.phi_d.FSTATUS = 1
m.phi_dd.FSTATUS = 0
m.theta.FSTATUS = 1
m.theta_d.FSTATUS = 1
m.theta_dd.FSTATUS = 0
m.xr.TR_INIT = 1       # TODO: What are these?
m.xr.TR_OPEN = 1       # What are these?
m.xr.TAU = 0.02        # What are these?

# Intermediates and Equations
# Intermediate for Force & tau for later implementation
m.Equation(
    (
        (m.Iw + (m.mw+m.mr)*m.R**2)*m.phi_dd +
        (m.mr*m.R*m.l*m.cos(m.theta))*m.theta_dd
    ) == m.mr*m.R*m.l*(m.theta_d**2)*m.sin(m.theta) + m.tau
)

m.Equation(
    (
        (m.mr*m.R*m.l*m.cos(m.theta))*m.phi_dd + (m.Ir +
        m.mr*m.l**2)*m.theta_dd
    ) == m.mr*m.g*m.l*m.sin(m.theta) - m.tau
)

m.Equation(m.phi.dt() == m.phi_d)
m.Equation(m.phi_d.dt() == m.phi_dd)
m.Equation(m.theta.dt() == m.theta_d)
m.Equation(m.theta_d.dt() == m.theta_dd)
m.Equation(m.xw == m.R * m.phi)    # Radius of Wheel times rotation speed
m.Equation(m.xr == m.xw + L*m.sin(m.theta)) # Position of robot

# Constraints
#m.theta.LOWER = -0.5*np.pi   # TODO: Limit the angle to +/- 90 deg
#m.theta.UPPER = 0.5*np.pi

# Solver options
m.options.SOLVER = 3
m.options.CV_TYPE = 1  # 1 for linear error model
m.options.IMODE = 6
#m.options.CV_WGT_SLOPE = 0.75   # TODO: Consider this or CV_WGT_START
# m.solve(remote=False)

# Plot
if m.options.APPSTATUS != 1:
    raise RuntimeError('gekko.solve() error')

t = np.linspace(0, 14, 401)

# Create arrays for storage.
θrz = np.empty_like(t)
θrz_dot = np.empty_like(t)
θwz  = np.empty_like(t)
θwz_dot = np.empty_like(t)
uz = np.empty_like(t)
xwz = np.empty_like(t)
xrz = np.empty_like(t)
yrz = np.empty_like(t)

# Define setpoints
xrSpz = np.array(
    [(0 if x<=0.5 else -0.5 if x<4 else 0.5 if x<6 else -0.5 if x<8 else 0.5) for x in t]
)

# Set variables to initial conditions
i = 0
θrz[i], θrz_dot[i] = θr_init, θr_dot_init
θwz[i], θwz_dot[i] = θw_init, θw_dot_init
xrz[i] = xr_init
#xrz_dot[0] = xr_dot_init
uz[i] = u_init

'''
Run simulation, looping over all t[].
unpack the result using transpose operation (.T).
'''
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(3, 3)

subPlot_a = plt.subplot(gs[0,0])
subPlot_u = plt.subplot(gs[1,0])
subPlot_y = plt.subplot(gs[2,0])
subPlat_anim = plt.subplot(gs[:,1:])

plt.ion()
plt.show()

for i in range(len(t) - 1):

    # TODO: Added this from Jim's version
    xwz[i] = R * θwz[i] # Calculate wheel's x-position based on angle of wheel.
    xrz[i] = xwz[i] + L*np.sin(θrz[i]) # Calculate robot body position
    yrz[i] = L*np.cos(θrz[i])

    Y = odeint(modelDerivFunc,
               [θrz[i], θrz_dot[i], θwz[i], θwz_dot[i]],
               [t[i], t[i+1]], args=(uz[i], ))

    θrz[i+1], θrz_dot[i+1], θwz[i+1], θwz_dot[i+1] = Y[1].T

    # Adjust setpoints
    m.xr.SPHI = xrSpz[i+1]
    m.xr.SPLO = xrSpz[i+1]

    # Make sure segway has not tipped over!
    #assert -0.5*np.pi < θrz[i+1] < 0.5*np.pi  TODO: Do we need this?

    # Read in measurements from the system (odeint)
    m.theta.MEAS = θrz[i+1]
    m.theta_d.MEAS = θrz_dot[i+1]
    m.phi.MEAS = θwz[i+1]
    m.phi_d.MEAS = θwz_dot[i+1]

    # solve MPC
    m.solve(remote=True)
    m.xr.TR_INIT = 1      # TODO: What is this?

    # Readout new manipulated variable values
    uz[i+1] = m.tau.NEWVAL

    #import pdb; pdb.set_trace()

    subPlot_a.cla()
    subPlot_a.grid()
    subPlot_a.plot(t[0:i],θrz[0:i],'b--',linewidth=3,label='rad Robot')
    subPlot_a.plot(t[0:i],θwz[0:i],'k--',linewidth=3,label='rad Wheel')
    subPlot_a.legend(loc='best')
    subPlot_a.set_ylabel('Angles [rad]')

    subPlot_u.cla()
    subPlot_u.grid()
    subPlot_u.plot(t[0:i], uz[0:i], '-',color='Black', label='u')
    subPlot_u.set_ylabel('Torque')
    subPlot_u.legend(loc='best')
    subPlot_u.set_xlabel('Time')

    subPlot_y.cla()
    subPlot_y.grid()
    subPlot_y.plot(t[0:i], xrz[0:i], '-',color='blue', label='xr')
    subPlot_y.plot(t[0:i], xwz[0:i], '-',color='black', label='xw')
    subPlot_y.plot(t[0:i], xrSpz[0:i], '-',color='red', label='SP')
    subPlot_y.legend(loc='best')
    subPlot_y.set_ylabel('x, position')

    subPlat_anim.cla()

    subPlat_anim.plot([xwz[i], xrz[i]], [0, yrz[i]], 'o-', color='blue',
                      linewidth=3, markersize=6, markeredgecolor='blue',
                      markerfacecolor='blue')
    subPlat_anim.plot([-4,4], [0,0],'k--', linewidth=1)

    # display force on mass as a horizontal line emanating from the mass m1
    subPlat_anim.plot([xwz[i], xwz[i] - uz[i]*3/1000.0], [0, 0], '-', color='gray', lw=3)

    # display force on mass as a horizontal line emanating from the mass m1
    subPlat_anim.plot([xrSpz[i], xrSpz[i]], [0, L], '--', color='red', lw=1)
    subPlat_anim.axis('equal')
    subPlat_anim.text(0.5, 0.05, 'time = %.1fs' % t[i])

    ##plt.draw()
    plt.pause(0.02)

plt.ioff()
plt.show()


# Animation Plot
# Effectively loop over all t[] to pre-calculate all x's and y's for the robot.

# Calculate wheel's x-position based on angle of wheel.
xwz = R*θwz    # TODO: Inform Jim anout the *2 here in his script
xrz = xwz + L*np.sin(θrz)

# Calculate robot body position
yrz = L*np.cos(θrz)

xMin = np.min([np.min(xrz), np.min(xwz)])
xMax = np.max([(np.max(xrz), np.max(xwz))])
print(xMin, xMax)

'''
Animate the pendulum - Initialization
http://tiao.io/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
'''
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# First set up the figure, the axis, and the plot element we want to animate
fig, axis = plt.subplots(figsize=(4, 4))

axis.set_xlim(-6*1.1,6*1.1)
axis.set_ylim(-6*1.1,6*1.1)

axis.plot(xrz,yrz, 'o', color='gray', markersize=1) # draw some ghost dots at each position. Don't join the dots with a line.

axis.plot([-6*1.1, 6*1.1], [0, 0], '--', lw=1, color='black')
time_text  = axis.text(0.04, 0.05, '0.0', transform=axis.transAxes)
time_text.set_text('t = %.1fs' % t[0])
line, = axis.plot([xwz[0], xrz[0]], [0, yrz[0]], 'o-', lw=3, markersize=4) # draw the pendulum, a line with 2 markers.

__file__ = 'OdeIntSegwayPid.ipynb' # jupyter doesn't know its name.
plt.savefig(__file__ + '.png') # Save initial conditions.
#%%
'''
Animate the pendulum.
'''
# initialization function: plot the background of each frame
def init_func():
    line.set_data([], [])
    time_text.set_text('t = %.1fs' % t[0])
    return (line, time_text)

# animation function. This is called sequentially
def animate_func(i):
    line.set_data([xwz[i], xrz[i]], [0, yrz[i]])
    time_text.set_text('t = %.1fs' % t[i])
    return (line, time_text)

# call the animator. blit=True means only re-draw the parts that have changed.
fps=60
frameDelay_msec = 10*1000.0/fps # 10x slow motion.
anim = animation.FuncAnimation(fig, animate_func, init_func=init_func, frames=len(t), interval=frameDelay_msec, blit=True)

'''
Display animation. This takes several seconds (about 7) to render.
for 'RuntimeError: No MovieWriters available!',
  do 'conda install -c conda-forge ffmpeg'
  see https://stackoverflow.com/a/44483126/101252
'''
HTML(anim.to_html5_video())