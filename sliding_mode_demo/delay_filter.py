import numpy as np
from copy import copy


class Delay:
    def __init__(self, nsteps):
        self.buf = None
        self.bufsz = nsteps + 1
        self.value = None

    def set_initial_value(self, t0, sig0):
        self.buf = [(t0, np.copy(sig0))] * self.bufsz
        self.value = sig0

    def update(self, t, sig):
        if self.buf is None:
            self.buf = [(t, np.copy(sig))] * self.bufsz

        self.buf.append((t, np.copy(sig)))
        n = max(len(self.buf) - self.bufsz, 0)
        self.buf = self.buf[n:]
        self.value = self.buf[0][1]

    def __call__(self, t, sig):
        self.update(t, sig)
        return self.value


def test():
    delay = Delay(0)
    x = list(range(10))
    y = [delay(i, i) for i in x]
    assert np.allclose(x, y)

    delay = Delay(1)
    y = [delay(i, i) for i in x]
    assert np.allclose(x[:-1], y[1:])

    delay = Delay(3)
    y = [delay(i, i) for i in x]
    assert np.allclose(x[:-3], y[3:])


if __name__ == '__main__':
    test()
