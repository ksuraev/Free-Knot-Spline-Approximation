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


def f_exp_sin_cos(t):
    return np.exp(t) * (
        np.sin(3 * np.pi * t) + 2 * np.cos(2 * np.pi * t) + 0.5 * np.sin(np.pi * t)
    )


def p(t):
    return 4 * np.minimum(t - np.floor(t), np.ceil(t) - t) - 1


def q(t):
    return p(t - 0.25)


def g(t):
    return np.exp(t) * (q(1.5 * t) + 2 * p(t) + 0.5 * q(0.5 * t))


def f_g(t):
    return 0.5 * (f_exp_sin_cos(t) - g(t))


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
    "exp_sin_cos": (
        f_exp_sin_cos,
        r"$e^t(\sin(3\pi t) + 2\cos(2\pi t) + 0.5\sin(\pi t))$",
    ),
    "g": (g, r"$e^t(q(1.5t) + 2p(t) + 0.5q(0.5t))$"),
    "f_g": (f_g, r"$0.5(f(t) - g(t))$"),
}
