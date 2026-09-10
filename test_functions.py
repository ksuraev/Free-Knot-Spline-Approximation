import numpy as np


def f_sin(t):
    return np.sin(t)


def f_sin_3t(t):
    return np.sin(3 * t)


def f_log(t):
    return np.log(t + 1)


def f_log_sin(t):
    return np.log(np.sin(t) + 3)


def f_sin_if_else(t):
    return np.where(t <= 2 * np.pi, np.sin(t), np.sin(2 * t))


def f_cos_if_else(t):
    return np.where(t <= 2 * np.pi, np.cos(t), np.cos(3 * t))


def f_cos_weird(t):
    return np.where(t <= 3 * np.pi, np.cos(t) + 0.1 * (t - 3 * np.pi), np.cos(3 * t))


def f_sin_weird(t):
    return np.where(t <= 2 * np.pi, np.sin(t) + 0.1 * (t - 2 * np.pi), np.sin(t))


TEST_FUNCTIONS = {
    "sin": (f_sin, r"$\sin(t)$"),
    "sin3t": (f_sin_3t, r"$\sin(3t)$"),
    "log": (f_log, r"$\log(t+1)$"),
    "log_sin": (f_log_sin, r"$\log(\sin(t)+3)$"),
    "sin_if_else": (f_sin_if_else, r"$\sin(t)$ if $t \leq 2\pi$, else $\sin(2t)$"),
    "cos_if_else": (f_cos_if_else, r"$\cos(t)$ if $t \leq 2\pi$, else $\cos(3t)$"),
    "cos_weird": (
        f_cos_weird,
        r"$\cos(t) + 0.1(t - 3\pi)$ if $t \leq 3\pi$, else $\cos(3t)$",
    ),
    "sin_weird": (
        f_sin_weird,
        r"$\sin(t) + 0.1(t - 2\pi)$ if $t \leq 2\pi$, else $\sin(t)$",
    ),
}
