import numpy as np

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
