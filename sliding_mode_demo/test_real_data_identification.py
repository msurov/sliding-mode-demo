from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
from common import load_logs_csv


def soft_sign(x, eps=1e-3):
    return x / (np.abs(x) + eps)

def resample(x1, y1, x2):
    sp = make_interp_spline(x1, y1, k=1)
    return sp(x2)

def test(csv_filename):
    data = load_logs_csv(csv_filename)

    time = data['time']
    time -= time[0]
    speed = data['speed']
    throttle = data['torque']
    brake = data['brake']    

    mask = (time > 15) & (time < 1000)
    time = time[mask]
    speed = speed[mask]
    throttle = throttle[mask]
    brake = brake[mask]

    n = len(time)
    speed_conv_rate = 2.0
    b_conv_rate = 1.7
    c_conv_rate = 1.7

    b_est = 3.5
    c_est = 0.
    y_est = speed[0]
    a_approx = -0.00
    c0_approx = -0.0

    b_values = [b_est]
    c_values = [c_est]
    y_values = [y_est]
    delay_steps = 0

    for i in range(1, n):
        y = speed[i]
        u = throttle[np.clip(i - delay_steps, 0, n - 1)]

        if brake[i] != 0:
            b_values.append(b_est)
            c_values.append(c_est)
            y_est = y
            y_values.append(y_est)
            continue

        if np.abs(y) < 0.01:
            b_values.append(b_est)
            c_values.append(c_est)
            y_est = y
            y_values.append(y_est)
            continue

        dt = time[i] - time[i-1]

        v = soft_sign(y_est - y, 0.1)
        db_est = -b_conv_rate * np.sign(u) * v
        dc_est = -c_conv_rate * v
        dy_est = a_approx * y + c0_approx * np.sign(y) + b_est * u + c_est - speed_conv_rate * v
        b_est = np.clip(b_est + db_est * dt, 1., 8.)
        c_est = np.clip(c_est + dc_est * dt, -1., 1.)
        y_est = y_est + dy_est * dt

        b_values.append(b_est)
        c_values.append(c_est)
        y_values.append(y_est)

    b_values = np.array(b_values)
    c_values = np.array(c_values)
    y_values = np.array(y_values)

    ax = plt.subplot(321)
    plt.grid(True)
    plt.plot(time, throttle, '.-', label='throttle')
    plt.plot(time, brake, '.-', label='brake')
    plt.legend()
    plt.ylabel('throttle')
    ax = plt.subplot(323, sharex=ax)
    plt.grid(True)
    plt.plot(time, speed, '.-', label='speed')
    plt.plot(time, y_values, '.-', label='speed est')
    plt.legend()
    plt.ylabel('speed')
    ax = plt.subplot(322, sharex=ax)
    plt.grid(True)
    plt.plot(time, b_values, '.-', label='b est')
    plt.legend()
    plt.ylabel('b')
    ax = plt.subplot(324, sharex=ax)
    plt.grid(True)
    plt.plot(time, c_values, '.-', label='c est')
    plt.ylabel('c')
    plt.legend()
    ax = plt.subplot(325, sharex=ax)
    plt.grid(True)
    plt.plot(time, -c_values / b_values, '.-', label=R'$u_0 = -\frac{c_{est}}{b_{est}}$')
    plt.ylabel('-c/b')
    plt.legend()

    plt.tight_layout()
    plt.show()

# test('data/vehicle_data.csv')
# test('data/kalibr-5_n1-008_2023-09-29-07-49-04Z_manual_data.evo1h_record_default.part_0.csv')
# test('data/kalibr-5_n1-017_2023-09-19-12-27-10Z_hill_test_reverse.evo1h_record_default.csv')
# test('data/kalibr-5_n1-017_2023-09-19-12-30-31Z_hill_test_reverse_1.evo1h_record_default.csv')
# test('data/kalibr-5_n1-017_2023-09-19-12-35-20Z_hill_test_reverse_2.evo1h_record_default.csv')
# test('data/kalibr-5_n1-008_2023-10-17-10-17-22Z_new_speed_controller_3.evo1h_record_default.size_20318059461.csv')
test('data/kalibr-5_n1-008_2023-10-18-10-26-56Z_gorka_test_pac.evo1h_record_default.size_9798566189.csv')
