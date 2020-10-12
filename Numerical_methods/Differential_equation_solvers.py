import numpy as np
from matplotlib import pyplot as plt


def euler_method(t0, y0, func, t_max, h=1):
    y = y0
    t = t0
    y_values = [y0]
    t_values = [t0]
    while t < t_max:
        y += func(t, y) * h
        t += h
        y_values.append(y)
        t_values.append(t)
    return t_values, y_values


def runge_kutta_method(t0, y0, func, t_max, h):
    y = y0
    t = t0
    y_values = [y0]
    t_values = [t0]
    while t < t_max:
        k1 = h * func(t, y)
        k2 = h * func(t + h/2, y + k1/2)
        k3 = h * func(t + h/2, y + k2/2)
        k4 = h * func(t + h, y + k3)

        y += k1/6 + k2/3 + k3/3 + k4/6
        t += h
        print(f"K1: {k1}, K2: {k2}, K3: {k3}, K4: {k4}")
        y_values.append(y)
        t_values.append(t)
    return t_values, y_values


def get_values(t0, y0, func, t_max, h):
    euler_t, euler_y = euler_method(t0, y0, func, t_max, h)
    runge_kutta_t, runge_kutta_y = runge_kutta_method(t0, y0, func, t_max, h)
    return euler_y, euler_t, runge_kutta_y, runge_kutta_t


def get_errors(true_f, estimate_y, estimate_t):
    errors = []
    for i in range(len(estimate_t)):
        error = np.abs(true_f(estimate_t[i]) - estimate_y[i])
        errors.append(error)
    return errors


def main():

    '''
    h = 0.1
    t0 = 0
    t_max = 10

    f1 = lambda t, y: 2 * t - y
    f1_true = lambda t: 2 * t + np.exp(-t) - 2
    f2 = lambda t, y: 2 * y * (1 - y)
    f2_true = lambda t: np.exp(2 * t) / (np.exp(2 * t) + 1)

    functions = [f1, f2]
    true_functions = [f1_true, f2_true]
    y0 = [-1, 0.5]
    text = ["y'(t) = 2t - y(t), y(0) = -1", "y'(t) = 2y(t)(1 - y(t)), y(0) = 0.5"]

    x = np.linspace(t0, t_max)
    for i, j, k, m in zip(functions, true_functions, y0, text):
        y_euler, t_euler, y_runge, t_runge = get_values(t0, k, i, t_max, h)
        fig, axs = plt.subplots(1, 2, figsize=(9, 5.5))
        axs[0].plot(t_euler, y_euler, '-r')
        axs[0].plot(t_runge, y_runge, '-b')
        axs[0].plot(x, j(x), '-y')
        axs[0].set_ylabel("Y")
        axs[0].set_xlabel("t")
        axs[0].set_title("Estimation")
        axs[1].plot(t_euler, get_errors(j, y_euler, t_euler.copy()), '-r')
        axs[1].plot(t_runge, get_errors(j, y_runge, t_runge.copy()), '-b')
        axs[1].set_ylabel("Error")
        axs[1].set_xlabel("t")
        axs[1].set_title("Errors")
        fig.suptitle(f"Estimating function: {m} with step size: {h}", y=0.98)
        #plt.savefig(f"Plot: {m}: {h}.png")
        plt.show()
    '''

if __name__ == "__main__":
    main()
