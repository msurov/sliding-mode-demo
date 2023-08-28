import numpy as np
import matplotlib.pyplot as plt
from .integrator import solve_ivp_fixed
from scipy.interpolate import make_interp_spline

sign = np.sign

def spow(x, a):
    return sign(x) * pow(abs(x), a)

def derivatives_estimator_test():
    lam = [1.1, 1.5, 2., 3.]
    k = 2
    L = 20.

    t = np.linspace(0, 10, 15)
    x = np.random.rand(*t.shape)
    sp = make_interp_spline(t, x, k=5)

    def f(t, k=0):
       return sp(t, k)
    
    def sys(t, z):
        fval = f(t) + 1e-1 * (np.random.normal() - 0.5)
        dz0 = -lam[k] * pow(L, 1 / (k + 1)) * spow(z[0] - fval, k / (k + 1)) + z[1]
        dz1 = -lam[k-1] * pow(L, 1 / k) * spow(z[1] - dz0, (k - 1) / k) + z[2]
        dz2 = -lam[k-2] * L * sign(z[2] - dz1)
        return [dz0, dz1, dz2]

    z0 = np.zeros(k + 1)
    t = np.arange(0, 10, 0.01)
    sol = solve_ivp_fixed(sys, [t[0], t[-1]], z0, t_eval=t, max_step=1e-3)

    plt.figure('test derivatives estimation')
    plt.subplot(311)
    plt.plot(sol.t, f(sol.t), '--')
    plt.plot(sol.t, sol.y[0])
    plt.grid(True)

    plt.subplot(312)
    plt.plot(sol.t, f(sol.t, 1), '--')
    plt.plot(sol.t, sol.y[1])
    plt.grid(True)

    plt.subplot(313)
    plt.plot(sol.t, f(sol.t, 2), '--')
    plt.plot(sol.t, sol.y[2])
    plt.grid(True)
    plt.show()

def test():
    derivatives_estimator_test()
