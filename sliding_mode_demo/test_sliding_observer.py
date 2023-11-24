from simulation import simulate
import numpy as np
import matplotlib.pyplot as plt


def step_fun(x, alpha=5.):
    return 0.5 + 0.5 * np.tanh(x * alpha)

def soft_sign(x, eps=1e-3):
    return x / (np.abs(x) + eps)

class SlidingObserver1:
    def __init__(self) -> None:
        self.state = [0, 0, 0, 0]
        self.z_est = [0, 0]
        self.u = 0.
        self.t = None

    def __call__(self, t, state):
        if self.t is None:
            self.t = t
        dt = t - self.t
        self.t = t

        y, = state
        q_est, x_est = self.z_est
        u = self.u
        M = -0.5
        Lq = -0.4
        b_est = 3.2

        v = soft_sign(x_est * b_est - y)
        dq_est = Lq * v
        dx_est = q_est + u + M * v
        q_est = np.clip(q_est + dq_est * dt, -1, 1)
        x_est += dx_est * dt
        self.z_est = [q_est, x_est]

        y_ref = step_fun(t - 5) + 3 * step_fun(t - 13) - 4 * step_fun(t - 21)
        u0 = self.z_est[0]
        self.u = -u0 + -0.4 * (y - y_ref)
        self.u = np.clip(self.u, -1, 1)
        self.state = [b_est, q_est, x_est * b_est, y_ref, u]
        return self.u

def test1():
    a = -0.0
    b = 3.2
    c = 0.3

    def sys(t, x, u):
        u = np.clip(u, -1, 1)
        return a*x + b*u + c
    
    fb = SlidingObserver1()

    t = np.arange(0, 30, 0.057)
    x0 = [0.]
    ans = simulate(sys, fb, x0, t, 0, 0)
    t = ans['t']
    x = ans['x']
    st = np.array(ans['feedback_internal_state'])

    b_est, q_est, y_est, y_ref, u = np.array(st).T

    plt.subplot(211)
    plt.plot(t, x, label='y actual')
    plt.plot(t, y_ref, '--', label='y ref')
    plt.plot(t, y_est, '--', label='y est')
    plt.legend()
    plt.subplot(223)
    plt.plot(t, q_est)
    plt.axhline(c / b, ls='--')
    plt.ylabel('c/b')
    plt.subplot(224)
    plt.plot(t, b_est)
    plt.axhline(b, ls='--')
    plt.ylabel('b')
    plt.tight_layout()
    plt.show()


class SlidingObserver2:
    def __init__(self) -> None:
        self.state = [0, 0, 0]
        self.z_est = np.array([0, 1, 0])
        self.u = 0.
        self.t = None

    def __call__(self, t, state):
        if self.t is None:
            self.t = t
        
        dt = t - self.t
        self.t = t

        y, = state
        q_est, b_est, x_est = self.z_est
        u = self.u

        L1 = -0.4
        L2 = -1.2
        M = 0.1

        v = np.sign(x_est * b_est - y)
        dq_est = L1 * v
        db_est = np.sign(u) * L2 * v
        dx_est = u + q_est - M * v

        q_est += dq_est * dt
        b_est += db_est * dt
        x_est += dx_est * dt

        b_est = np.clip(b_est, 1., 5.)
        q_est = np.clip(q_est, -1., 1.)
        self.z_est = np.array([q_est, b_est, x_est])

        y_ref = step_fun(t - 5) + step_fun(t - 12) - step_fun(t - 20) \
             + step_fun(t - 31) + step_fun(t - 39) - 2 * step_fun(t - 47) + step_fun(t - 55) \
            + step_fun(t - 61) + step_fun(t - 67) - 2 * step_fun(t - 77) + step_fun(t - 83)
        self.u = -q_est - 1.0 * (y - y_ref)

        self.state = [b_est, q_est, y_ref, x_est * b_est, self.u]
        return self.u

class SlidingObserver3:
    def __init__(self) -> None:
        self.state = [0, 0, 0, 0, 0]
        self.z_est = [0, 0]
        self.u = 0.
        self.t = None

    def __call__(self, t, state):
        if self.t is None:
            self.t = t
        dt = t - self.t
        self.t = t

        y, = state
        b_est, y_est = self.z_est
        u = self.u
        M = 1.3
        L = -2.2
        c_est = 0.0

        v = np.sign(y_est - y)
        db_est = np.sign(u) * L * v
        dy_est = c_est + b_est * u - M * v

        b_est += db_est * dt
        y_est += dy_est * dt
        b_est = np.clip(b_est, 1., 5.)
        self.z_est = [b_est, y_est]

        y_ref = step_fun(t - 5) + 3 * step_fun(t - 13) - 4 * step_fun(t - 21)
        u0 = -c_est / b_est

        self.u = -u0 + -1.2 * (y - y_ref)
        self.u = np.clip(self.u, -1, 1)
        self.state = [b_est, c_est, y_ref, self.z_est[1], self.u]
        return self.u
    
def test3():

    a = -0.0
    b = 3.2
    c = 0.07

    def sys(t, x, u):
        u = np.clip(u, -1, 1)
        return a*x + b*u + c
    
    fb = SlidingObserver3()

    t = np.arange(0, 30, 0.057)
    x0 = [0.]
    ans = simulate(sys, fb, x0, t, 0, 0)
    t = ans['t']
    x = ans['x']
    st = np.array(ans['feedback_internal_state'])

    b_est, q_est, y_ref, y_est, u = np.array(st).T

    plt.subplot(211)
    plt.plot(t, x, label='y actual')
    plt.plot(t, y_ref, '--', label='y ref')
    plt.plot(t, y_est, '--', label='y est')
    plt.legend()
    plt.subplot(223)
    plt.plot(t, q_est)
    plt.ylabel('c')
    plt.axhline(c, ls='--')
    plt.subplot(224)
    plt.plot(t, b_est)
    plt.axhline(b, ls='--')
    plt.ylabel('b')
    plt.show()


def test2():
    a = -0.0
    b = 3.2
    c = 0.07

    def sys(t, x, u):
        u = np.clip(u, -1, 1)
        return a*x + b*u + c
    
    fb = SlidingObserver2()

    t = np.arange(0, 30, 0.057)
    x0 = [0.]
    ans = simulate(sys, fb, x0, t, 0, 0)
    t = ans['t']
    x = ans['x']
    st = np.array(ans['feedback_internal_state'])

    b_est, q_est, y_ref, y_est, u = np.array(st).T

    plt.subplot(311)
    plt.plot(t, x, label='y actual')
    plt.plot(t, y_ref, '--', label='y ref')
    plt.plot(t, y_est, '--', label='y est')
    plt.legend()
    plt.subplot(312)
    plt.plot(t, q_est)
    plt.axhline(c / b, ls='--')
    plt.subplot(313)
    plt.plot(t, b_est)
    plt.axhline(b, ls='--')
    plt.show()

class SlidingObserver4:
    def __init__(self) -> None:
        self.state = [0, 0, 0, 0, 0]
        self.z_est = [3.0, 0, 0]
        self.u = 0.
        self.t = None

    def __call__(self, t, state):
        if self.t is None:
            self.t = t
        dt = t - self.t
        self.t = t

        y, = state
        b_est, c_est, y_est = self.z_est
        u = self.u
        M = -0.5
        Lb = -0.5
        Lc = -0.2

        v = soft_sign(y_est - y)
        db_est = Lb * np.sign(u) * v
        dc_est = Lc * v
        dy_est = b_est * u + c_est + M * v
        b_est = np.clip(b_est + db_est * dt, 1., 5.)
        c_est = np.clip(c_est + dc_est * dt, -1., 1.)
        y_est = y_est + dy_est * dt
        self.z_est = [b_est, c_est, y_est]

        y_ref = step_fun(t - 5) + 3 * step_fun(t - 13) - 2 * step_fun(t - 21) \
             + step_fun(t - 27) - 3 * step_fun(t - 39)
        u0 = -c_est / b_est
        self.u = u0 - 1.2 * (y - y_ref)

        self.u = np.clip(self.u, -1, 1)
        self.state = [b_est, c_est, y_est, y_ref, u]
        return self.u

def test4():
    a = -0.0
    b = 3.2
    c = 0.12

    def sys(t, x, u):
        u = np.clip(u, -1, 1)
        return a*x + b*u + c
    
    fb = SlidingObserver4()

    t = np.arange(0, 50, 0.057)
    x0 = [0.]
    ans = simulate(sys, fb, x0, t, 0, 0)
    t = ans['t']
    x = ans['x']
    st = np.array(ans['feedback_internal_state'])

    b_est, c_est, y_est, y_ref, u = np.array(st).T

    plt.subplot(311)
    plt.plot(t, x, label='y actual')
    plt.plot(t, y_ref, '--', label='y ref')
    plt.plot(t, y_est, '--', label='y est')
    plt.legend()
    plt.subplot(312)
    plt.plot(t, b_est)
    plt.axhline(b, ls='--')
    plt.ylabel('b')
    plt.subplot(313)
    plt.plot(t, c_est)
    plt.axhline(c, ls='--')
    plt.ylabel('c')
    plt.tight_layout()
    plt.show()



# test1()
# test2()
# test3()
test4()
