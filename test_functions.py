import numpy as np


def f_sin(t):
    return np.sin(t)


def f_sin_3t(t):
    return np.sin(3 * t)


def f_log(t):
    return np.log(t + 1)


def f_log_sin(t):
    return np.log(np.sin(t) + 3)


TEST_FUNCTIONS = {
    "sin": (f_sin, r"$\sin(t)$"),
    "sin3t": (f_sin_3t, r"$\sin(3t)$"),
    "log": (f_log, r"$\log(t+1)$"),
    "log_sin": (f_log_sin, r"$\log(\sin(t)+3)$"),
}
