import numpy as np
from scipy.integrate import ode
from copy import copy


def disturb_variable(u, state_disturbance):
    if state_disturbance is not None:
        return u + np.random.normal(u) * state_disturbance
    return u

def simulate(sys : callable, fb : callable, initial_state : np.ndarray, t_eval : np.ndarray, state_disturbance=None, **ode_kwargs):
    initial_state = np.array(initial_state)
    u = fb(t_eval[0], disturb_variable(initial_state, state_disturbance))
    t_shape = t_eval.shape

    def rhs(t, x):
        return sys(t, x, u)

    integrator = ode(rhs)
    integrator.set_initial_value(initial_state, t_eval[0])
    integrator.set_integrator('dopri5', **ode_kwargs)

    solx = np.zeros(t_shape + initial_state.shape, float)
    solx[0] = initial_state
    solu = np.zeros(t_shape + np.shape(u), float)
    solu[0] = u

    if hasattr(fb, 'state'):
        solfb = [copy(fb.state)]
    else:
        solfb = None

    for i in range(1, t_shape[0]):
        if not integrator.successful():
            print('[warn] integrator doesn\'t feel good')
            break
        # step of integration
        integrator.integrate(t_eval[i])
        t = integrator.t
        solx[i] = integrator.y
        # call controller
        u = fb(t, disturb_variable(solx[i], state_disturbance))
        if u is None:
            break
        solu[i] = u
        # save fb state
        if hasattr(fb, 'state'):
            solfb.append(copy(fb.state))

    return solx, solu, solfb
