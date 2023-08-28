import numpy as np
from collections import namedtuple

class Integrator:
    def __init__(self):
        self.u_prev = None
        self.sum = 0.

    def process(self, t, u):
        if self.u_prev is None:
            self.u_prev = u
            self.t_prev = t
            self.sum = 0 * u
        
        self.sum += (u + self.u_prev) * (t - self.t_prev) / 2
        self.u_prev = np.copy(u)
        self.t_prev = t
        return self.sum

    def reset(self):
        self.u_prev = None
        self.sum *= 0

    def __call__(self, t, u):
        return self.process(t, u)


def solve_ivp_fixed(sys, tspan, x0, t_eval=None, max_step=None):
    if t_eval is None:
        t_eval = np.arange(tspan[0], tspan[1], max_step)
    yshape = np.shape(x0) + np.shape(t_eval)
    y = np.zeros(yshape)
    y[...,0] = x0
    for i in range(1, len(t_eval)):
        t_prev = t_eval[i-1]
        x_prev = y[...,i-1]
        dx = np.array(sys(t_prev, x_prev))
        t_new = t_eval[i]
        x_new = x_prev + dx * (t_new - t_prev)
        y[...,i] = x_new
    Solution = namedtuple('IVPSolution', ['t', 'y'])
    return Solution(t_eval, y)
