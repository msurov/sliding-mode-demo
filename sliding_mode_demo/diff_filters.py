import numpy as np
from scipy.linalg import expm


class DiffFilter:
    R'''
        The second order low pass diff filter with the transfer function

            W(s) = [1, s] / (Ts + 1)**2
    '''
    def __init__(self, time_constant) -> None:
        a2 = time_constant**2
        a1 = 2*time_constant
        a0 = 1
        self.A = np.array([
            [0, 1],
            [-a0/a2, -a1/a2]
        ])
        self.B = np.array([
            0,
            1/a2
        ])
        self.t_prev = None

    def process(self, t, u):
        if self.t_prev is None:
            self.t_prev = t
            self.x = np.zeros(2)
            return self.x
        
        dt = t - self.t_prev
        self.t_prev = t
        expA = expm(self.A * dt)
        I = np.eye(2)
        invA = np.linalg.inv(self.A)
        Ad = expA
        Bd = invA @ (expA - I) @ self.B
        self.x = Ad @ self.x + Bd * u
        return self.x
    
    def value(self):
        return self.x

    def __call__(self, t, u):
        return self.process(t, u)

sign = np.sign

def spow(x, a):
    return sign(x) * pow(abs(x), a)

class SlidingDiffFilter:
    R'''
        The third order sliding mode diff filter
    '''
    def __init__(self, lipschitz_constant):
        self.lam = [1.1, 1.5, 2.]
        self.L = lipschitz_constant
        self.state = np.zeros(3)
        self.t = None

    def process(self, t, u):
        k = 2
        lam = self.lam
        L = self.L
        z = self.state

        if self.t is None:
            self.t = t
            self.state[0] = u

        dz0 = -lam[k] * pow(L, 1 / (k + 1)) * spow(z[0] - u, k / (k + 1)) + z[1]
        dz1 = -lam[k-1] * pow(L, 1 / k) * spow(z[1] - dz0, (k - 1) / k) + z[2]
        dz2 = -lam[k-2] * L * sign(z[2] - dz1)
        dz = np.array([dz0, dz1, dz2])
        self.state += dz * (t - self.t)
        self.t = t
        return np.copy(self.state)

    def value(self):
        return self.state

    def __call__(self, t, u):
        return self.process(t, u)
