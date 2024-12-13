from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .simulation import simulate
from .integrator import Integrator


class SuperTwisting:
    def __init__(self) -> None:
        self.k1 = 0.3
        self.k2 = self.k1**2 * 1.5
        c = 1.0
        n = np.array([c, 1], float)
        self.n = n / np.linalg.norm(n)
        self.integ = Integrator()
        self.state = 0

    def reset(self):
        self.integ.reset()
        self.state = 0

    def __call__(self, t, x):
        z = self.n @ x
        v = -self.k2 * self.integ(t, np.sign(z))
        u = (-self.k1 * np.sqrt(np.abs(z)) * np.sign(z) + v - self.n[0] * x[1]) / self.n[1]
        self.state = z
        return u

class Sliding:
    def __init__(self) -> None:
        self.k = 0.3
        c = 1.0
        n = np.array([c, 1], float)
        self.n = n / np.linalg.norm(n)
        self.state = 0
    
    def reset(self):
        self.state = 0

    def __call__(self, t, x):
        z = self.n @ x
        u = (-self.k * np.sign(z) - self.n[0] * x[1]) / self.n[1]
        self.state = z
        return u

def perp_vec(n):
    x,y = n
    return np.array([y, -x])

def process_tests(sys, fb, tspan, step):
    t = np.arange(*tspan, step)

    xmin = -2.0
    xmax = 2.0
    dxmin = -2.0
    dxmax = 2.0
    xstep = 0.2
    dxstep = 0.2

    plt.subplot(221)
    plt.grid(True)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axline([0, 0], perp_vec(fb.n), color='red', ls='--')

    ic_list = \
        [[x0, xmin] for x0 in np.arange(xmin, xmax, xstep)] + \
        [[x0, xmax] for x0 in np.arange(xmin, xmax, xstep)] + \
        [[dxmin, dx0] for dx0 in np.arange(dxmin, dxmax, dxstep)] + \
        [[dxmax, dx0] for dx0 in np.arange(dxmin, dxmax, dxstep)]

    for ic in ic_list:
        fb.reset()
        solx, solu, solfb = simulate(sys, fb, x0=ic, t=t)
        plt.plot(solx[:,0], solx[:,1], '-', color='blue', lw=0.5, alpha=0.5)

    plt.subplot(222)
    plt.grid(True)
    plt.plot(t, solx[:,0], '-', color='green', lw=1, alpha=0.5, label='$x_1$')
    plt.axhline()
    plt.plot(t, solx[:,1], '-', color='blue', lw=1, alpha=0.5, label='$x_2$')
    plt.xlabel('$t$, sec')
    plt.ylabel('phase coordinates')
    plt.legend()
    plt.axhline()

    plt.subplot(223)
    plt.grid(True)
    plt.plot(t, solu, alpha=0.5)
    plt.xlabel('$t$, sec')
    plt.ylabel('control input, $u$')

    plt.subplot(224)
    plt.grid(True)
    plt.plot(t, solfb, alpha=0.5)
    plt.xlabel('$t$, sec')
    plt.ylabel('plane signed distance, $z$')

    plt.tight_layout(w_pad=2.0)

def sliding_test():
    R'''
        Test usual sliding-mode controller
    '''

    fb = Sliding()

    def sys(_, x, u):
        dx = np.array([x[1], u])
        return dx

    plt.figure('sliding', figsize=(8,6))
    process_tests(sys, fb, [0, 20], 1e-2)
    plt.savefig('out/sliding.png')

def twisting_test():
    R'''
        Test super-twist controller
    '''
    def sys(t, x, u):
        dx = np.array([x[1], u])
        return dx
    
    fb = SuperTwisting()
    plt.figure('super-twist', figsize=(8,6))
    process_tests(sys, fb, [0, 20], 1e-2)
    plt.savefig('out/super-twist.png')

def test():
    twisting_test()
    sliding_test()
    plt.show()
