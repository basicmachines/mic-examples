import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec


class MobileInvertedPendulum(object):

    # Default constants (for both Odeint and GEKKO)
    # TODO: Add units
    defaults = {
        'Mw': 10,    # mass of wheel
        'Mr': 2,     # mass of robot
        'R': 4,      # radius of wheel
        'L': 6,      # length from end to center of mass of the rod
        'G': 9.81,   # gravity (9.81 m/s)
        'Fw': 0.0,   # Friction, wheel, opposing θw_dot
        'Fr': 0.0    # Friction, robot, opposing θr_dot
    }

    def __init__(self, name="MIP", constants=None, init_values=None,
                 t=0.0, step_size=1.0):

        if constants:
            self.constants = constants
        else:
            self.constants = self.defaults

        # Inertia of wheel
        self.constants['Iw'] = (0.5*self.constants['Mw'] *
                                self.constants['R']**2)

        # Inertia of robot
        self.constants['Ir'] = (1/3 * self.constants['Mr'] *
                                (2*self.constants['L'])**2)

        # Define variables and initial values
        if init_values:
            self.init_values = init_values
        else:
            self.init_values = {
                'θr': 0.0,
                'θr_dot': 0.0,
                'xr': 0.0,
                'xr_dot': 0.0
            }

        # Additional (derived) variables
        # x-position of wheel  TODO: Make this into a function.
        self.init_values['xw'] = (
            self.init_values['xr'] -
            self.constants['L']*np.sin(self.init_values['θr'])
        )

        # angle of wheel
        self.init_values['θw'] = self.init_values['xw'] / self.constants['R']

        # angular velocity of wheel
        self.init_values['θw_dot'] = (self.init_values['xr_dot']/
                                      self.constants['R'])

        # Calculate robot body position
        self.init_values['yr'] = (self.constants['L']*
                                  np.cos(self.init_values['θr']))

        # Current time period and step-size
        self.t = t
        self.step_size = step_size

        # Manipulated variables
        self.mvs = {
            'tau': 0.0
        }

        # Set variables to initial values
        self.reset()

    def reset(self):

        self.variables = self.init_values.copy()

    def derivFunc(self, xz, tz, u):

        θr, θr_dot, θw, θw_dot = xz # unpack inputs.

        Mw, Mr, R, L, G, Fw, Fr, Iw, Ir = [
            self.constants['Mw'],
            self.constants['Mr'],
            self.constants['R'],
            self.constants['L'],
            self.constants['G'],
            self.constants['Fw'],
            self.constants['Fr'],
            self.constants['Iw'],
            self.constants['Ir']
        ]

        A = np.array([
             [Mr*R*L*np.cos(θr), (Iw + (Mw + Mr)*R**2)],
             [(Ir + Mr*L**2), Mr*R*L*np.cos(θr)]
        ])

        B = np.array([
             [Mr*R*L*θr_dot**2*np.sin(θr) + u - θr_dot*Fr],
             [Mr*G*L*np.sin(θr) - u - θw_dot*Fw]
        ])

        # Solve for X where AX=B
        X = np.linalg.solve(A, B)

        [θr_ddot, θw_ddot] = X

        return [θr_dot, θr_ddot, θw_dot, θw_ddot]

    def next_state(self):

        current_state = [
            self.variables['θr'],
            self.variables['θr_dot'],
            self.variables['θw'],
            self.variables['θw_dot']
        ]

        time_step = (self.t, self.t + self.step_size)

        y = odeint(self.derivFunc,
                   current_state,
                   time_step, args=(self.mvs['tau'], ))

        θr, θr_dot, θw, θw_dot = y[1].T

        self.variables['θr'] = θr
        self.variables['θr_dot'] = θr_dot
        self.variables['θw'] = θw
        self.variables['θw_dot'] = θw_dot

        # Calculate wheel's x-position based on angle of wheel.
        self.variables['xw'] = self.constants['R']*θw
        self.variables['xr'] = self.variables['xw'] + (
                               self.constants['L']*np.sin(θr)
                               )

        # Calculate robot body position
        self.variables['yr'] = self.constants['L']*np.cos(θr)

        self.t += self.step_size


class DataRecorder(object):

    def __init__(self, model, n_steps, params=None, filename=None):

        self.model = model

        if params is None:
            params = {}
        self.params = params

        self.n_steps = n_steps
        self.filename = filename

        self.columns = ['t'] + list(model.variables.keys()) + \
                       list(model.mvs.keys()) + \
                       list(self.params.keys())

        self.data = pd.DataFrame(index=range(n_steps), columns=self.columns)
        self.current_row = 0

    def record_state(self):

        self.data.iloc[self.current_row,:] = (
            [self.model.t] +
            list(self.model.variables.values()) +
            list(self.model.mvs.values()) +
            list(self.params.values())
        ) # TODO: It would be safer to check the keys haven't changed

        self.current_row += 1

        if self.current_row > len(self.data):
            raise NotImplementError("Reached end of dataframe")

    def save_to_csv(self, filename=None):

        if filename is None:
            filename = self.filename

        self.data[:self.current_row].to_csv(filename)

def main():

    # Instantiate model
    model = MobileInvertedPendulum(t=0.0, step_size=0.035)

    # Init solver and set time values
    m = GEKKO()

    # Time horizon for control
    horizon_steps = 10
    m.time = np.linspace(0.0, model.step_size*horizon_steps,
                         horizon_steps + 1)

    # Get parameters from model
    Mw = model.constants['Mw']
    Mr = model.constants['Mr']
    R = model.constants['R']
    L = model.constants['L']
    G = model.constants['G']
    Fw = model.constants['Fw']
    Fr = model.constants['Fr']
    Iw = model.constants['Iw']
    Ir = model.constants['Ir']

    # Get initial values of variables from model
    θr = model.variables['θr']
    θr_dot = model.variables['θr_dot']
    xr = model.variables['xr']
    xr_dot = model.variables['xr_dot']
    tau = model.mvs['tau']

    # Setup model
    # Parameters
    m.mw = m.Param(name='mass_wheel', value=Mw)
    m.mr = m.Param(name='mass_robot', value=Mr)
    m.l = m.Param(name='length_cm', value=L)
    m.R = m.Param(name='radius_wheel', value=R)
    m.g = m.Param(name='gravity', value=G)
    m.Iw = m.Param(name='Inertia_wheel', value=Iw)
    m.Ir = m.Param(name='Inertia_robot', value=Ir)

    # State Variables - Wheel angle not controlled but torque of motor
    m.phi = m.SV(name='angle_wheel')
    m.phi_d = m.SV(name='angle_wheel_dot', value=0) # TODO: Needs to be set
    m.phi_dd = m.SV(name='angle_wheel_dotdot', value=0) # TODO: Needs to be set
    m.theta = m.CV(name='angle_robot', value=θr, lb=-np.pi/4, ub=np.pi/4)   # TODO: Jim's settings
    m.theta_d = m.SV(name='angle_robot_dot', value=θr_dot)
    m.theta_dd = m.SV(name='angle_robot_dotdot', value=0)
    m.xw = m.Var(name='xWheel', value=0)

    # Controlled variables
    m.xr = m.CV(name='xRobot', value=xr)
    # m.xr_d = m.CV(name='x_position_wheel_dot', value=xr_dot_init)   # TODO: Jim has no CV for this

    # Manipulated variables
    m.tau = m.MV(name='torque', value=tau, lb=-1000, ub=1000)

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

    # Setpoint trajectory initialization
    m.xr.TR_INIT = 1    # Setpoint trajectory initialization mode
                        # TR_INIT = 1 makes the initial conditions equal
                        # to the current value
    m.xr.TR_OPEN = 1    # Initial trajectory opening ratio
    m.xr.TAU = 0.02     # Time constant for controlled variable response

    # Intermediates and equations
    # TODO: Need to add friction terms before using Fw, Fr.
    m.Equation(
        (
            (m.Iw + (m.mw+m.mr)*m.R**2)*m.phi_dd +
            m.mr*m.R*m.l*m.cos(m.theta)*m.theta_dd
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
    m.Equation(m.xw == m.R*m.phi)    # Radius of wheel times rotation speed
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

    # Initialize data recorder with a column for the set points
    params = {'xr_sp': 0.0}
    data_recorder = DataRecorder(model, n_steps=401, params=params)

    # Record initial state
    data_recorder.record_state()

    # Define setpoint signal
    def xr_sp_f(t):
        return 0 if t<=0.5 else -0.5 if t<4 \
               else 0.5 if t<6 else -0.5 if t<8 else 0.5

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

    for i in range(0, 401):

        # Run MPC solver to produce control actions
        m.solve(remote=True)

        # Read next value for manipulated variable
        model.mvs['tau'] = m.tau.NEWVAL

        # Move forward to current timestep in the dynamic
        # model simulation
        model.next_state()

        # Adjust setpoints
        m.xr.SPHI = xr_sp_f(model.t)
        m.xr.SPLO = xr_sp_f(model.t)
        data_recorder.params['xr_sp'] = xr_sp_f(model.t)

        # Make sure segway has not tipped over!
        #assert -0.5*np.pi < θr[i+1] < 0.5*np.pi  TODO: Do we need this?

        # Read in measurements from the system (odeint)
        m.theta.MEAS = model.variables['θr']
        m.theta_d.MEAS = model.variables['θr_dot']
        m.phi.MEAS = model.variables['θw']
        m.phi_d.MEAS = model.variables['θw_dot']

        data_recorder.record_state()

        # Plot data from data recorder
        θr = data_recorder.data['θr']
        θw = data_recorder.data['θw']
        tau = data_recorder.data['tau']
        t = data_recorder.data['t']
        xr = data_recorder.data['xr']
        xw = data_recorder.data['xw']
        yr = data_recorder.data['yr']
        xr_sp = data_recorder.data['xr_sp']

        subPlot_a.cla()
        subPlot_a.grid()
        subPlot_a.plot(t[0:i],θr[0:i],'b--', linewidth=2, label='rad Robot')
        subPlot_a.plot(t[0:i],θw[0:i],'k--', linewidth=2, label='rad Wheel')
        subPlot_a.legend(loc='best')
        subPlot_a.set_ylabel('Angles [rad]')

        subPlot_u.cla()
        subPlot_u.grid()
        subPlot_u.plot(t[0:i], tau[0:i], '-', color='Black', label='u')
        subPlot_u.set_ylabel('Torque')
        subPlot_u.legend(loc='best')
        subPlot_u.set_xlabel('Time')

        subPlot_y.cla()
        subPlot_y.grid()
        subPlot_y.plot(t[0:i], xr[0:i], '-',color='blue', label='xr')
        subPlot_y.plot(t[0:i], xw[0:i], '-',color='black', label='xw')
        subPlot_y.plot(t[0:i], xr_sp[0:i], '-',color='red', label='SP')
        subPlot_y.legend(loc='best')
        subPlot_y.set_ylabel('x, position')

        subPlat_anim.cla()

        subPlat_anim.plot([xw[i], xr[i]], [0, yr[i]], 'o-', color='blue',
                          linewidth=3, markersize=6, markeredgecolor='blue',
                          markerfacecolor='blue')
        subPlat_anim.plot([-4, 4], [0, 0], 'k--', linewidth=1)

        # display force on mass as a horizontal line emanating from the mass m1
        subPlat_anim.plot([xw[i], xw[i] - tau[i]*3/1000.0], [0, 0], '-',
                          color='gray', lw=3)

        # display force on mass as a horizontal line emanating from the mass m1
        subPlat_anim.plot([xr_sp[i], xr_sp[i]], [0, L], '--', color='red', lw=1)
        subPlat_anim.axis('equal')
        subPlat_anim.text(0.5, 0.05, 'time = %.1fs' % t[i])

        ##plt.draw()
        plt.pause(0.02)

    plt.ioff()
    plt.show()

    data_recorder.save_to_csv("MIP_data.csv")

    # Animation Plot
    # Effectively loop over all t[] to pre-calculate all x's and y's for the robot.

    # Calculate wheel's x-position based on angle of wheel.
    xw = R*θw    # TODO: Inform Jim anout the *2 here in his script
    xr = xw + L*np.sin(θr)

    # Calculate robot body position
    yr = L*np.cos(θr)

    xMin = np.min([np.min(xr), np.min(xw)])
    xMax = np.max([(np.max(xr), np.max(xw))])
    print(xMin, xMax)

    '''
    Animate the pendulum - Initialization
    http://tiao.io/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/
    '''

    from matplotlib import animation
    from IPython.display import HTML

    # First set up the figure, the axis, and the plot element we want to animate
    fig, axis = plt.subplots(figsize=(4, 4))

    axis.set_xlim(-6*1.1, 6*1.1)
    axis.set_ylim(-6*1.1, 6*1.1)

    # draw some ghost dots at each position. Don't join the dots with a line.
    axis.plot(xr ,yr, 'o', color='gray', markersize=1)

    axis.plot([-6*1.1, 6*1.1], [0, 0], '--', lw=1, color='black')
    time_text  = axis.text(0.04, 0.05, '0.0', transform=axis.transAxes)
    time_text.set_text('t = %.1fs' % t[0])

    # draw the pendulum, a line with 2 markers.
    line, = axis.plot([xw[0], xr[0]], [0, yr[0]], 'o-', lw=3, markersize=4)

    plt.savefig(__file__ + '.png') # Save initial conditions.

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
        line.set_data([xw[i], xr[i]], [0, yr[i]])
        time_text.set_text('t = %.1fs' % t[i])
        return (line, time_text)

    # call the animator. blit=True means only re-draw the parts that have
    # changed.
    fps = 30
    frameDelay_msec = 10*1000.0/fps # 10x slow motion.

    anim = animation.FuncAnimation(fig, animate_func, init_func=init_func,
                                   frames=len(t), interval=frameDelay_msec,
                                   blit=True)

    '''
    Display animation. This takes several seconds (about 7) to render.
    for 'RuntimeError: No MovieWriters available!',
      do 'conda install -c conda-forge ffmpeg'
      see https://stackoverflow.com/a/44483126/101252
    '''
    HTML(anim.to_html5_video())


if __name__ == '__main__':
    main()