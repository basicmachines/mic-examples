from gekko import GEKKO
import nn_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.gridspec import GridSpec
from matplotlib import animation
import timeit

# Use this if using jupyter notebook:
from IPython.display import HTML


previous_settings = {
    'Mw': 10,    # mass of wheel
    'Mr': 2,     # mass of robot
    'R': 4,      # radius of wheel
    'L': 6,      # length from end to center of mass of the rod
    'G': 9.81,   # gravity (9.81 m/s)
    'Fw': 0.0,   # Friction, wheel, opposing θw_dot
    'Fr': 0.0    # Friction, robot, opposing θr_dot
}

class MobileInvertedPendulum(object):

    # Default constants (for both Odeint and GEKKO)
    # Estimates based on Beagle Bone specifications
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
                 t=0.0, step_size=0.1):

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

        L = self.constants['L']
        R = self.constants['R']

        self.variables['θr'] = θr
        self.variables['θr_dot'] = θr_dot
        self.variables['θw'] = θw
        self.variables['θw_dot'] = θw_dot

        # Calculate wheel's x-position based on angle of wheel.
        self.variables['xw'] = R*θw

        # Calculate robot body's x-position
        self.variables['xr'] = self.variables['xw'] + (
                               L*np.sin(θr)
                               )
        self.variables['xr_dot'] = R*θw_dot + L*θr_dot

        # Calculate robot body position
        self.variables['yr'] = L*np.cos(θr)

        self.t += self.step_size


class DataRecorder(object):

    def __init__(self, model, n, params=None, filename=None):

        self.model = model

        if params is None:
            params = {}
        self.params = params

        self.n = n
        self.filename = filename

        self.columns = ['t'] + \
                       sorted(list(model.variables.keys())) + \
                       sorted(list(model.mvs.keys())) + \
                       sorted(list(self.params.keys()))

        self.data = pd.DataFrame(index=range(n), columns=self.columns,
                                 dtype=float)
        self.current_row = 0

    def merge_dicts(self, list_of_dicts):
        """Merges the dictionaries into one."""

        new_dict = list_of_dicts[0].copy()
        for e in list_of_dicts[1:]:
            new_dict.update(e)

        return new_dict

    def record_state(self):

        self.data.iloc[self.current_row, :] = self.merge_dicts(
            [
                {'t': self.model.t},
                self.model.variables,
                self.model.mvs,
                self.params
            ]
        ) # TODO: Is there an easier way to do this?

        self.current_row += 1

        if self.current_row > len(self.data):
            raise NotImplementError("Reached end of dataframe")

    def save_to_csv(self, filename=None):

        if filename is None:
            filename = self.filename

        self.data[:self.current_row].to_csv(filename)


class GEKKO_MPC(GEKKO):
    """Subclass of GEKKO.  Creates a GEKKO object with all
    parameters and variables initialized based on model object.
    """

    def __init__(self, model, horizon_steps=10):

        # Init solver and set time values
        GEKKO.__init__(self)

        self.time = np.linspace(0.0, model.step_size*horizon_steps,
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

        # Setup model parameters
        self.mw = self.Param(name='mass_wheel', value=Mw)
        self.mr = self.Param(name='mass_robot', value=Mr)
        self.L = self.Param(name='length_m', value=L)
        self.R = self.Param(name='radius_wheel', value=R)
        self.g = self.Param(name='gravity', value=G)
        self.Iw = self.Param(name='Inertia_wheel', value=Iw)
        self.Ir = self.Param(name='Inertia_robot', value=Ir)

        # State Variables - Wheel angle not controlled but torque of motor
        self.phi = self.SV(name='angle_wheel')

        # TODO: Values should be set based on xr_dot
        self.phi_d = self.SV(name='angle_wheel_dot', value=0)
        self.phi_dd = self.SV(name='angle_wheel_dotdot', value=0)

        self.theta = self.CV(name='angle_robot', value=θr, lb=-np.pi/4,
                             ub=np.pi/4)   # TODO: Check Jim's settings
        self.theta_d = self.SV(name='angle_robot_dot', value=θr_dot)
        self.theta_dd = self.SV(name='angle_robot_dotdot', value=0)
        self.xw = self.Var(name='xWheel', value=0)

        # Controlled variables
        self.xr = self.CV(name='xRobot', value=xr)
        # TODO: Jim has no CV for this
        # self.xr_d = self.CV(name='x_position_wheel_dot', value=xr_dot_init)

        # Manipulated variables
        self.tau = self.MV(name='torque', value=tau, lb=-1000, ub=1000)

        # Define parameter options
        self.tau.STATUS = 1
        self.tau.DCOST = 0

        # add (STATUS=1) setpoint to obj function
        self.xr.STATUS = 1
        self.theta.STATUS = 1
        #self.theta_d.STATUS = 1   # TODO: Jim does not have this
        #self.xr_d.STATUS = 1      # TODO: Jim does not have this

        # set FSTATUS = 1 to follow measurement
        self.phi.FSTATUS = 1
        self.phi_d.FSTATUS = 1
        self.phi_dd.FSTATUS = 0
        self.theta.FSTATUS = 1
        self.theta_d.FSTATUS = 1
        self.theta_dd.FSTATUS = 0

        # Setpoint trajectory initialization
        self.xr.TR_INIT = 2    # Setpoint trajectory initialization mode
                               # TR_INIT = 1 makes the initial conditions equal
                               # to the current value
        self.xr.TR_OPEN = 5    # Trajectory funnel opening ratio
        self.xr.TAU = 0.02     # Time constant for controlled variable response
        self.theta.TR_INIT = 2
        self.theta.TR_OPEN = 5
        self.theta.TAU = 0.02

        # Intermediates and equations
        # TODO: Need to add friction terms before using Fw, Fr.
        self.Equation(
            (
                (self.Iw + (self.mw+self.mr)*self.R**2)*self.phi_dd +
                self.mr*self.R*self.L*self.cos(self.theta)*self.theta_dd
            ) == self.mr*self.R*self.L*(self.theta_d**2)*self.sin(self.theta) +
                 self.tau
        )

        self.Equation(
            (
                (self.mr*self.R*self.L*self.cos(self.theta))*self.phi_dd +
                (self.Ir + self.mr*self.L**2)*self.theta_dd
            ) == self.mr*self.g*self.L*self.sin(self.theta) - self.tau
        )

        self.Equation(self.phi.dt() == self.phi_d)
        self.Equation(self.phi_d.dt() == self.phi_dd)
        self.Equation(self.theta.dt() == self.theta_d)
        self.Equation(self.theta_d.dt() == self.theta_dd)

        # Calculate position of wheel and robot
        self.Equation(self.xw == self.R*self.phi)
        self.Equation(self.xr == self.xw + L*self.sin(self.theta))

        # Constraints
        #self.theta.LOWER = -0.5*np.pi   # TODO: Limit the angle to +/- 90 deg
        #self.theta.UPPER = 0.5*np.pi

        # Adjustments to objective function
        self.Obj(1.0e-6*self.tau**2)  # Penalize unnecessary movements

        # Solver options
        self.options.SOLVER = 3
        self.options.CV_TYPE = 1  # 1 for linear error model
        self.options.IMODE = 6
        #self.options.CV_WGT_SLOPE = 0.75
        # TODO: Consider this or maybe CV_WGT_START


def create_animation(model, data_recorder, fps=30, figsize=(4, 4),
                     filename=__file__ + '.png'):
    '''Creates a matplotlib animation of the simulation
    and saves it to disk.

    The animation that is returned can also be viewed in
    a Jupyert Notebook using:

    >>> HTML(anim.to_html5_video())

    For more info see:
        http://tiao.io/posts/notebooks/embedding-matplotlib-
                         animations-in-jupyter-notebooks/

    This takes several seconds (about 7) to render.

    If you encouter 'RuntimeError: No MovieWriters available!',
    try the following:

    $ conda install -c conda-forge ffmpeg

    For more information see:
    see https://stackoverflow.com/a/44483126/101252
    '''

    # First calculate x, y coordinates for the robot at
    # all timesteps

    # Wheel's x-position based on angle of wheel.
    R = model.constants['R']
    L = model.constants['L']
    θr = data_recorder.data['θr']
    θw = data_recorder.data['θw']
    xw = R*θw

    # Calculate wheel x-position
    xr = xw + L*np.sin(θr)

    # Calculate robot body x-position
    yr = L*np.cos(θr)

    t = data_recorder.data['t']

    xMin = np.min([np.min(xr), np.min(xw)])
    xMax = np.max([(np.max(xr), np.max(xw))])
    #print(xMin, xMax)

    # First set up the figure, the axis, and the plot element
    # we want to animate
    fig, axis = plt.subplots(figsize=figsize)

    axis.set_xlim(-L*1.1, L*1.1)
    axis.set_ylim(-L*1.1, L*1.1)

    # Draw some ghost dots at each position. Don't join the dots
    # with a line.
    axis.plot(xr ,yr, 'o', color='gray', markersize=1)

    axis.plot([-L*1.1, L*1.1], [0, 0], '--', lw=1, color='black')
    time_text  = axis.text(0.04, 0.05, '0.0', transform=axis.transAxes)
    time_text.set_text('t = %.1fs' % t[0])

    # draw the pendulum, a line with 2 markers.
    line, = axis.plot([xw[0], xr[0]], [0, yr[0]], 'o-', lw=3, markersize=4)

    plt.savefig(filename)

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

    # Call the animator.
    frameDelay_msec = 10*1000.0/fps # 10x slow motion.

    # blit=True means only re-draw the parts that have changed.
    return animation.FuncAnimation(fig, animate_func, init_func=init_func,
                                   frames=len(t), interval=frameDelay_msec,
                                   blit=True)


def main():

    start_time = pd.datetime.now()

    solver_times = []

    # Instantiate dynamic model
    model = MobileInvertedPendulum(t=0.0, step_size=0.035)

    # Choose length of simulation (timesteps)
    n_steps = 401

    # Time horizon for predictive control
    horizon_steps = 10

    # Initialise solver based on model parameters
    m = GEKKO_MPC(model, horizon_steps)

    # Convenience function
    def new_setpoint(var, value, weight=None, tol=0.01):
        var.SPHI = value*(1 + tol)
        var.SPLO = value*(1 - tol)
        if weight is not None:
            var.WSPHI = weight
            var.WSPLO = weight

    # Initialize data recorder to save state data to
    # file with an extra column for the set points
    params = {'xr_sp': 0.0, 'θr_sp': 0.0, 'tau_p1': model.mvs['tau'],
              'tau_p1_nn': model.mvs['tau']}
    data_recorder = DataRecorder(model, n=n_steps+1, params=params)

    '''
    Run simulation, looping over all t[].
    unpack the result using transpose operation (.T).
    '''

    # Define setpoint changes for the demo
    def xr_sp_f(t):
        return (0 if t <= 0.5
                else -0.5 if t < 4
                else 0.5 if t < 6
                else -0.5 if t < 8
                else 0.5)

    def random_setpoint_generator(mu=0.1, sigma=0.25, n=30, init=None):

        if init is not None:
            current_value = init
        else:
            current_value = np.random.normal(mu, sigma)

        while True:
            for i in range(np.random.poisson(n)):
                yield current_value
            current_value = np.random.normal(mu, sigma)

    # Initialize random_setpoint_generator
    # Set init=0.0 to make it start at 0
    xr_sp_random = random_setpoint_generator(mu=0.0, sigma=0.40, n=100)

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(3, 3)

    ax_a = plt.subplot(gs[0, 0])
    ax_u = plt.subplot(gs[1, 0])
    ax_y = plt.subplot(gs[2, 0])
    ax_anim = plt.subplot(gs[:,1:])

    plt.ion()
    plt.show()

    for i in range(0, n_steps + 1):

        # TODO: Remove this when the problem of TR_INIT is fixed
        if i == 1:
            m.theta.TR_INIT = 1
            m.xr.TR_INIT = 1

        # Desired setpoints for robot angle and xr
        theta_sp, xr_sp = 0.0, next(xr_sp_random)

        new_setpoint(m.theta, theta_sp, weight=2)
        new_setpoint(m.xr, xr_sp, weight=1)

        # Store current setpoint values
        data_recorder.params['θr_sp'] = theta_sp
        data_recorder.params['xr_sp'] = xr_sp

        # Other setpoint options
        # TODO: Jim does not set these...
        #new_setpoint(m.theta_d, 0, weight=1)
        #new_setpoint(m.theta_dd, 0, weight=10)

        # setpoints for robot position to 0
        #new_setpoint(m.xr, 0)               # Do we need a weight?
        #new_setpoint(m.xr_d, 0, weight=1)
        #new_setpoint(m.xr_dd, 0, weight=10)

        # Run MPC solver to predict next control actions
        timer1 = timeit.default_timer()

        m.solve(remote=True)
        tau_p1 = m.tau.NEWVAL

        # Or use trained neural network instead
        tau_p1_nn = nn_predict.next_tau(
            tau=model.mvs['tau'],
            θw=model.variables['θw'],
            θw_dot=model.variables['θw_dot'],
            θr=model.variables['θr'],
            θr_dot=model.variables['θr_dot'],
            xr_sp=xr_sp
        )

        timer2 = timeit.default_timer()
        solver_times.append(timer2 - timer1)

        # Read next value for manipulated variable
        data_recorder.params['tau_p1'] = tau_p1
        data_recorder.params['tau_p1_nn'] = tau_p1_nn

        # Save current model state to memory
        data_recorder.record_state()

        # Update dynamic model and simulate next time step
        model.mvs['tau'] = tau_p1   # or tau_p1_nn
        model.next_state()

        # Make sure segway has not tipped over!
        #assert -0.5*np.pi < θr[i+1] < 0.5*np.pi
        # TODO: Do we need this?

        # Read in measurements from the system (odeint)
        m.theta.MEAS = model.variables['θr']
        m.theta_d.MEAS = model.variables['θr_dot']
        m.phi.MEAS = model.variables['θw']
        m.phi_d.MEAS = model.variables['θw_dot']

        # Plot data from data recorder
        θr = data_recorder.data['θr']
        θw = data_recorder.data['θw']
        tau = data_recorder.data['tau']
        t = data_recorder.data['t']
        xr = data_recorder.data['xr']
        xw = data_recorder.data['xw']
        yr = data_recorder.data['yr']
        xr_sp = data_recorder.data['xr_sp']

        ax_a.cla()
        ax_a.grid()
        ax_a.plot(t[0:i], θr[0:i], 'b--', linewidth=2, label='rad Robot')
        ax_a.plot(t[0:i], θw[0:i], 'k--', linewidth=2, label='rad Wheel')
        ax_a.legend(loc='best')
        ax_a.set_ylabel('Angles (rad)')

        ax_u.cla()
        ax_u.grid()
        ax_u.plot(t[0:i], tau[0:i], '-', color='Black', label='u')
        ax_u.set_ylabel('Torque (N.m^2)')
        ax_u.legend(loc='best')

        ax_y.cla()
        ax_y.grid()
        ax_y.plot(t[0:i], xw[0:i], '-', color='black', label='xw')
        ax_y.plot(t[0:i], xr_sp[0:i], '--', color='red', lw=1, label='SP')
        ax_y.plot(t[0:i], xr[0:i], '-', color='blue', label='xr')
        ax_y.legend(loc='best')
        ax_y.set_xlabel('Time (s)')
        ax_y.set_ylabel('x position (m)')

        ax_anim.cla()

        ax_anim.plot([xw[i], xr[i]], [0, yr[i]], 'o-', color='blue',
                     linewidth=3, markersize=6, markeredgecolor='blue',
                     markerfacecolor='blue')
        ax_anim.plot([-4, 4], [0, 0], 'k--', linewidth=1)

        # display force on mass as a grey line
        # TODO: Try an arrow:
        #xs = [xw[i], xw[i] - tau[i]*3/1000.0]
        #xs if tau[i] < 0 else sorted(xs)
        #pyplot.arrow(x, y, dx, dy, hold=None, **kwargs)
        ax_anim.plot([xw[i], xw[i] - tau[i]*3/1000.0], [0, 0], '-',
                     color='gray', lw=3)

        # display body of pendulum
        L = model.constants['L']
        ax_anim.plot([xr_sp[i], xr_sp[i]], [0, L], '--', color='red', lw=1)
        ax_anim.axis('equal')
        ax_anim.text(0.5, 0.05, 'time = %.2fs' % t[i])

        #plt.draw()
        plt.pause(0.02)

    time_elapsed = (pd.datetime.now() - start_time).seconds
    time_stamp = start_time.strftime("%y%m%d%H%M")

    filename = 'MIP_MPC_data ' + time_stamp + '.csv'
    data_recorder.save_to_csv(filename)

    print("\nSimulation finished after %d hours, %d minutes." %
          (time_elapsed // 3600, time_elapsed // 60)
    )

    filename = 'MIP_MPC_plot ' + time_stamp + '.pdf'
    plt.savefig(filename)

    print("Average duration of each iteration: "
          "{:6.0f}ms".format(time_elapsed*1000/n_steps))

    tmin, tmax, tmean = (min(solver_times), max(solver_times),
                         sum(solver_times)/len(solver_times))

    print("APMonitor solver speed (milliseconds):\n"
          "Fastest: {:6.0f}ms\n"
          "Slowest: {:6.0f}ms\n"
          "   Mean: {:6.0f}ms\n".format(tmin*1000, tmax*1000, tmean*1000))

    print("Close plot window to continue.")

    plt.ioff()
    plt.show()

    # Make an animated plot for display in Jupyter Notebook
    # TODO: This needs adjusting to new dimensions
    #animation = create_animation(model, data_recorder)


if __name__ == '__main__':
    main()