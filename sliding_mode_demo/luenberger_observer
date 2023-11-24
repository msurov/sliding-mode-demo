from simulation import simulate
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


class FeedbackLuenberger:
    def __init__(self, a, b) -> None:
        self.z = np.zeros(2)
        self.t = None
        self.a = a
        self.b = b
        self.l1 = 20.0
        self.l2 = a + 10.0
        self.state = [0., 0., 0.]
        self.k = 2.0

        A = np.array([
            [0, 0],
            [1, a]
        ])
        B = np.array([
            [0],
            [b]
        ])
        C = np.array([[0, 1]])
        L = np.array([
            [self.l1],
            [self.l2]
        ])
        K = A - L @ C
        evals,_ = np.linalg.eig(K)
        assert np.all(np.real(evals) < 0)

        self.x_ref = make_interp_spline(
            [0, 1, 2, 4, 6, 8, 9, 10],
            [1, 3, 4, 4, 6, 2, 2, 0],
            k = 1
        )

    def get_ref_traj(self, t):
        v = self.x_ref(t)
        a = self.x_ref(t, 1)
        return v, a

    def __call__(self, t, state):
        y, = state
        l1 = self.l1
        l2 = self.l2
        a = self.a
        b = self.b
        k = self.k

        if self.t is None:
            self.t = t
            self.z = np.array([0, y])

        dt = t - self.t
        self.t = t

        c_bar, y_bar = self.z

        y_ref, dy_ref = self.get_ref_traj(t)
        e = y - y_ref
        u = 1/b * (-k * e - a*y - c_bar + dy_ref)

        dc_bar = -l1 * (y_bar - y)
        dy_bar = a * y_bar + b * u + c_bar - l2 * (y_bar - y)

        c_bar += dc_bar * dt
        y_bar += dy_bar * dt

        self.z = np.array([c_bar, y_bar])
        self.state = [c_bar, y_bar, u, y_ref]
        return u

def test_leunberger():

    a = -0.1
    b = 3.2
    c = 0.33

    def sys(t, x, u):
        return a*x + b*u + c
    
    fb = FeedbackLuenberger(a, b)

    t = np.arange(0, 10, 0.1)
    x0 = [0.]
    x, u, st = simulate(sys, fb, x0, t)
    c1,x1,u1,x_ref = np.array(st).T
    x = x[:,0]

    plt.subplot(311)
    plt.plot(t, x)
    plt.plot(t, x_ref, '--')

    plt.subplot(312)
    plt.plot(t, c1)
    plt.axhline(c, ls='--')

    plt.subplot(313)
    plt.plot(t, x)
    plt.plot(t, x1, '--')

    plt.show()
