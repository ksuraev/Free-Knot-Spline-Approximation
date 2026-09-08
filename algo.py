from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nadia
import nadia_mod
import nurnberger_mod
import remez
import test_functions

date = datetime.now().strftime("%Y-%m-%d")

plot_dir = Path("plots") / date
plot_dir.mkdir(parents=True, exist_ok=True)


def fixed_left_tail(f, a, theta, b, m, n):
    P_left, d_left, alt_left = remez.remez(f, a, theta, m)
    fixed_left = nadia_mod.gra(f, [theta, b], m, 1, fixed_left_value=P_left(theta))

    def S(i, t):
        if i == 0:
            return P_left(t)
        return fixed_left["S"](0, t)

    return {
        "d_max": abs(d_left),
        "gra_d_max": fixed_left["d_max"],
        "case": 2,
        "S": S,
        "basis": [np.asarray(alt_left), fixed_left["basis"][0]],
        "knots": [a, theta, b],
        "optimal": fixed_left["exit_type"] == 1,
    }


def fixed_right_tail(f, a, theta, b, m, n):
    P_right, d_right, alt_right = remez.remez(f, theta, b, m)
    fixed_right = nadia_mod.gra(f, [a, theta], m, 1, fixed_right_value=P_right(theta))

    def S(i, t):
        if i == 0:
            return fixed_right["S"](0, t)
        return P_right(t)

    return {
        "d_max": abs(d_right),
        "gra_d_max": fixed_right["d_max"],
        "case": 3,
        "S": S,
        "basis": [np.asarray(fixed_right["basis"][0]), alt_right],
        "knots": [a, theta, b],
        "optimal": fixed_right["exit_type"] == 1,
    }


def psi(f, a, b, theta, m, n):
    two_int_chain = nadia_mod.gra(f, [a, theta, b], m, n)

    # Case 1: found optimal spline across two intervals
    if two_int_chain["exit_type"] == 1:
        return {
            "d_max": two_int_chain["d_max"],
            "case": 1,
            "S": two_int_chain["S"],
            "basis": two_int_chain["basis"],
            "knots": [a, theta, b],
            "optimal": True,
        }

    exit_interval = two_int_chain["exit_i_star"]

    # Case 2: tried to replace basis point in 1st interval
    # Start with [a, theta] as minimal chain
    if exit_interval == 0:
        fixed = fixed_left_tail(f, a, theta, b, m, n)

        # [a, theta] is wrong choice for minimal chain, swap to [theta, b]
        if abs(fixed["gra_d_max"]) > abs(fixed["d_max"]) + 1e-4:
            print(
                f"case 2: d_max={fixed['d_max']:.10f} < gra_d_max={fixed['gra_d_max']:.10f} at theta={theta:.10f}"
            )
            return fixed_right_tail(f, a, theta, b, m, n)

        return fixed

    # Case 3: tried to replace basis point in 2nd interval
    # Start with [theta, b] as minimal chain
    fixed = fixed_right_tail(f, a, theta, b, m, n)

    # [theta, b] is wrong choice for minimal chain, swap to [a, theta]
    if abs(fixed["gra_d_max"]) > abs(fixed["d_max"]) + 1e-4:
        print(
            f"case 3: d_max={fixed['d_max']:.10f} < gra_d_max={fixed['gra_d_max']:.10f} at theta={theta:.10f}"
        )
        return fixed_left_tail(f, a, theta, b, m, n)

    return fixed


def directional_derivative(f, a, b, theta, psi_theta, m, n, h):
    psi_theta_h = psi(f, a, b, theta + h, m, n)["d_max"]

    return (psi_theta_h - psi_theta) / abs(h)


def armijo(f, a, b, theta, psi_theta, m, n, g, d, rho=0.5, c=0.1):
    if d < 0:
        alpha = a - theta / d
    else:
        alpha = (b - theta) / d

    while alpha > 1e-8:
        theta_next = theta + alpha * d

        if theta_next <= a or theta_next >= b:
            alpha *= rho
            continue

        psi_next = psi(f, a, b, theta_next, m, n)["d_max"]

        print(
            f"alpha={alpha:.10f}, theta_next={theta_next:.10f}, psi_next={psi_next:.10f}, psi_theta={psi_theta:.10f}, g={g:.10f}, d={d:.10f}"
        )

        if psi_next <= psi_theta + c * alpha * g:
            return theta_next

        alpha *= rho

    return theta


def opt(
    f, a, b, m, n, x_min, x_max, h=0.1, rho=0.5, c=0.1, tolerance=1e-5, max_iter=100
):
    theta = x_min

    for k in range(max_iter):
        psi_theta = psi(f, a, b, theta, m, n)["d_max"]

        g_plus = directional_derivative(f, a, b, theta, psi_theta, m, n, h)

        g_minus = directional_derivative(f, a, b, theta, psi_theta, m, n, -h)

        print(
            f"k={k}, theta={theta:.10f}, psi={psi_theta:.10f}, g_plus={g_plus:.10f}, g_minus={g_minus:.10f}"
        )

        if g_plus <= g_minus:
            d = h
            g = g_plus
        else:
            d = -h
            g = g_minus

        if g >= 0 or abs(g) < tolerance:
            break

        theta_next = armijo(f, a, b, theta, psi_theta, m, n, g, d, rho, c)

        if abs(theta_next - theta) < tolerance:
            break

        theta = theta_next

    psi_opt = psi(f, a, b, theta, m, n)

    return theta, psi_opt


def plot_psi(f, f_label, a, b, m, n, theta_start, theta_end, file_name, step=0.02):
    thetas = np.arange(theta_start, theta_end, step)

    psi_values = []

    for theta in thetas:
        result = psi(f, a, b, theta, m, n)

        psi_values.append(result["d_max"])

    plt.figure(figsize=(8, 6))

    plt.plot(thetas, psi_values)

    psi_min = min(psi_values)
    theta_min = thetas[np.argmin(psi_values)]

    plt.axvline(
        x=theta_min,
        color="red",
        linestyle="--",
        lw=0.5,
        label=rf"$\theta_{{\min}}={theta_min:.4f}$ with $\psi(\theta_{{\min}})={psi_min:.4f}$",
    )

    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\psi(\theta)$")
    plt.title(r"$\psi(\theta)$ for " + f"{f_label} with m={m}, k={k}")

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(file_name)
    plt.show()


if __name__ == "__main__":
    function_name = "sin3t"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a = 0
    b = 6

    k = 1
    n = k + 1
    m = 1

    knots, x_min, x_max, d_n = nurnberger_mod.discontinuous_spline(
        f, function_name, a, b, k, m
    )

    if x_min is None:
        x_min = knots[1]

    print("Knots:", knots)
    print("x_min:", x_min)
    print("d_n:", d_n)

    start_theta = a + 0.1
    end_theta = b - 0.1

    psi_plot_file = (
        plot_dir
        / f"psi_{function_name}_a{a}_b{b}_s{start_theta}_e{end_theta}_k{k}_m{m}.png"
    )

    plot_psi(f, f_label, a, b, m, n, start_theta, end_theta, psi_plot_file, 0.04)

    # theta_opt, psi_result = opt(f, a, b, m, n, 3.68, x_max)

    # print("Optimal theta:", theta_opt)
    # print("psi(theta):", psi_result["d_max"])
    # print("psi case:", psi_result["case"])

    # case = (
    #     "fixed left"
    #     if psi_result["case"] == 2
    #     else "fixed right" if psi_result["case"] == 3 else "two intervals"
    # )

    # nadia.plot(
    #     f,
    #     f_label,
    #     a,
    #     b,
    #     n,
    #     psi_result["S"],
    #     psi_result["knots"],
    #     psi_result["basis"],
    #     m,
    #     k,
    #     (f"optimal, {case}" if psi_result["optimal"] else f"not optimal, {case}"),
    #     plot_dir
    #     / f"{function_name}_a{a}_b{b}_knot{theta_opt:.5f}_k{k}_m{m}_psi(t){psi_result["d_max"]:.5f}.png",
    # )
