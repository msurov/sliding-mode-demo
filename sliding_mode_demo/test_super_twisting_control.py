import numpy as np
from scipy.integrate import ode
from dataclasses import dataclass
from delay_filter import Delay
from simulation import ControlSystemSimulator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


@dataclass
class SystemParameters:
  rolling_friction : float
  static_friction : float
  control_coef : float
  ext_force : float
  u_time_relax : float # 

class SystemDynamics:
  def __init__(self, par : SystemParameters):
    a = par.rolling_friction
    b = par.control_coef
    c = par.ext_force
    T = par.u_time_relax
    q = par.static_friction

    def cb(t, st, u):
      _,v,w = st
      dx = v
      if np.abs(v) < 1e-5 and abs(b * w + c) < q:
        dv = 0.
      else:
        dv = -a * v + b * w + c
      dw = -w/T + u[0]/T
      return np.array([dx,dv,dw])

    self.cb = cb

  def __call__(self, t, st, u):
    return self.cb(t, st, u)

@dataclass
class SuperTwistingParameters:
  lam : float
  alp : float
  umax : float
  surfang : float

def spow(x, deg, eps=1e-3):
  abs_x = np.abs(x)
  return np.power(abs_x, deg) * x / (abs_x + eps)

def smooth_sign(x, eps=1e-3):
  return x / (np.abs(x) + eps)

class SuperTwistingController:
  def __init__(self, par : SuperTwistingParameters):
    self.state = np.zeros(2)
    self.par = par

  def cb(self, t, st):
    lam = self.par.lam
    alp = self.par.alp
    umax = self.par.umax
    surfang = self.par.surfang

    x, v, _ = st
    s = x * np.cos(surfang) + v * np.sin(surfang)

    u1,tprev = self.state
    dt = t - tprev
    u = -lam * spow(s, 0.5, 0.15) + u1

    if np.abs(u) > umax:
      du1 = -u
    else:
      du1 = -alp * smooth_sign(s, 0.15)

    u1 = u1 + dt * du1
    self.state[0] = u1
    self.state[1] = t

    return u

  def __call__(self, t, st):
    return self.cb(t, st)

def main():
  step = 0.053
  syspar = SystemParameters(
    rolling_friction = 0.05,
    static_friction = 0.02,
    control_coef = 3.8,
    ext_force = 0.8,
    u_time_relax = 0.15
  )
  sys = SystemDynamics(syspar)
  ctrpar = SuperTwistingParameters(
    lam = 0.8,
    alp = 0.05,
    umax = 0.8,
    surfang = 1.0
  )
  fb = SuperTwistingController(ctrpar)
  sim = ControlSystemSimulator(sys, fb, step, 
    u_delay_steps=1, x_delay_steps=3, noise=np.array([0., 0.05, 0.05]))
  xstart = [1., 0., 0.]
  simtime = 40
  simres = sim.run(xstart, 0, simtime)
  _,ax = plt.subplots(2, 1, sharex=True)
  plt.sca(ax[0])
  plt.plot(simres.t, simres.x)
  plt.legend(['x', 'v', 'w'])
  plt.grid(True)

  plt.sca(ax[1])
  plt.plot(simres.t, simres.u)
  plt.legend(['u'])
  plt.grid(True)

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
