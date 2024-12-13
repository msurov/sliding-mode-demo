from scipy.integrate import ode
import numpy as np
from copy import copy
from delay_filter import Delay
from fixed_step_integrator import FixedStepIntegrator
from dataclasses import dataclass

@dataclass
class SimulationResult:
    t : np.ndarray
    x : np.ndarray
    u : np.ndarray
    controller_internal_state : np.ndarray

class ControlSystemSimulator:
    def __init__(self, sysrhs : callable, ctrlinput : callable, 
                    step : float, u_delay_steps=0, x_delay_steps=0, noise=0):
        R'''
            # Parameters
                * `ctrlinput`: is a control input functor of the type `ctrlinput(t, x) -> u`
                * `sysrhs` is the system dynamics functor of the type `sysrhs(t, x, u) -> dx/dt`
                * `step` is the step of discretization in sec
        '''
        self.sysrhs = sysrhs
        self.ctrlinput = ctrlinput
        self.step = step
        self.t = None
        self.u = None
        self.x = None
        self.noise = noise
        self.u_delay_filter = Delay(u_delay_steps)
        self.x_delay_filter = Delay(x_delay_steps)

    def run(self, xstart : np.ndarray, tstart : float, tend : float):
        R'''
            # Parameters
                * `xstart`: initial state
                * `tstart`, `tend` define time interval
        '''
        self.t = float(tstart)
        self.x = np.reshape(xstart, (-1,))
        xdim, = self.x.shape
        self.u = np.reshape(self.ctrlinput(tstart, self.x), (-1,))
        udim, = self.u.shape

        self.x_delay_filter.set_initial_value(tstart, xstart)

        def rhs(t, x):
            return np.reshape(self.sysrhs(t, x, self.u), xdim)

        integrator = FixedStepIntegrator(rhs, self.step, self.t, self.x)
        solt = [self.t]
        solx = [self.x]
        solu = [self.u]

        if hasattr(self.ctrlinput, 'state'):
            solfb = [copy(self.ctrlinput.state)]
        else:
            solfb = []

        while self.t < tend:
            if not integrator.successful():
                print('[warn] integrator doesn\'t feel good')
            integrator.integrate(self.t + self.step)
            self.t = integrator.t
            self.x = integrator.y
            x = self.x_delay_filter(self.t, self.x)
            x += self.noise * np.random.normal(size=x.shape)
            u = self.ctrlinput(self.t, x)
            if u is None:
                break
        
            u_delayed = self.u_delay_filter(self.t, u)
            self.u = np.reshape(u_delayed, udim)
            solt += [self.t]
            solx += [self.x.copy()]
            solu += [self.u.copy()]

            if hasattr(self.ctrlinput, 'state'):
                solfb.append(copy(self.ctrlinput.state))

        return SimulationResult(
            t = np.asanyarray(solt),
            x = np.asanyarray(solx),
            u = np.asanyarray(solu),
            controller_internal_state = solfb
        )


def simulate(sys, fb, x0, t, u_delay=0, x_delay=0):
    step = np.mean(np.diff(t))
    sim = ControlSystemSimulator(sys, fb, step, u_delay_steps=u_delay, x_delay_steps=x_delay)
    return sim.run(x0, t[0], t[-1])
