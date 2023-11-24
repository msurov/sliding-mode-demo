import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from integrator import solve_ivp_fixed
from diff_filters import DiffFilter, SlidingDiffFilter, sign, spow
from sliding_mode_demo.common import load_logs_csv


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
    ax = plt.subplot(311)
    plt.plot(sol.t, f(sol.t), '--')
    plt.plot(sol.t, sol.y[0])
    plt.grid(True)

    ax = plt.subplot(312, sharex=ax)
    plt.plot(sol.t, f(sol.t, 1), '--')
    plt.plot(sol.t, sol.y[1])
    plt.grid(True)

    ax = plt.subplot(313, sharex=ax)
    plt.plot(sol.t, f(sol.t, 2), '--')
    plt.plot(sol.t, sol.y[2])
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def diff_filters_benchmark():
    np.random.seed(0)

    args = np.linspace(0, 10, 15)
    vals = np.random.rand(*args.shape)
    sp = make_interp_spline(args, vals, k=5)

    step = 20e-3
    t = np.arange(sp.t[0], sp.t[-1], step)
    x = sp(t)
    xd = x + 0e-2 * np.random.normal(size=x.shape)

    diff_filter1 = DiffFilter(0.01)
    output = [diff_filter1(*v) for v in zip(t, xd)]
    s0,s1 = zip(*output)

    diff_filter2 = SlidingDiffFilter(10.)
    output = [diff_filter2(*v) for v in zip(t, xd)]
    w0,w1,_ = zip(*output)

    plt.figure('Compare derivative estimators')
    ax = plt.subplot(211)
    plt.plot(t, s0, label=R'$\frac{1}{(Ts+1)^2}$')
    plt.plot(t, w0, label=R'Sliding mode filter')
    plt.plot(t, xd, '-.', label=R'noised')
    plt.plot(t, sp(t), '--', label=R'original')
    plt.grid(True)
    plt.legend()

    ax = plt.subplot(212, sharex=ax)
    plt.plot(t, s1, label=R'$\frac{s}{(Ts+1)^2}$')
    plt.plot(t, w1, label=R'Sliding mode filter')
    plt.plot(t, sp(t, 1), '--', label=R'original')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def process_measurements():
    data = load_logs_csv('data/exp.csv')
    t = data['t']
    theta = data['theta']
    dtheta = data['dtheta']
    torque = data['torque']
    diff_filter = SlidingDiffFilter(1300.)
    out = [diff_filter(*v) for v in zip(t, theta)]
    w0,w1,_ = zip(*out)
    ax = plt.subplot(131)
    plt.grid(True)
    plt.plot(t, theta)
    plt.plot(t, w0)
    plt.subplot(132, sharex=ax)
    plt.grid(True)
    plt.plot(t, dtheta)
    plt.plot(t, w1)
    plt.subplot(133, sharex=ax)
    plt.grid(True)
    plt.plot(t, torque)
    plt.show()

def test():
    # derivatives_estimator_test()
    # diff_filters_benchmark()
    process_measurements()

test()
