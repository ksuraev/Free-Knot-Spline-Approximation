import matplotlib.pyplot as plt
import numpy as np

import nadia_mod
import nurnberger_mod
import plotting
import remez
import test_functions


def fixed_left_tail(f, a, theta, b, m, n):
    # Run Remez on the left interval [a, theta] to get the polynomial
    P_left, d_left, alt_left = remez.remez(f, a, theta, m)

    # Run GRA on the right interval [theta, b] with fixed left value P_left(theta)
    fixed_left = nadia_mod.gra(f, [theta, b], m, 1, fixed_left_value=P_left(theta))

    # Spline function combining the left polynomial and the right spline
    def S(i, t):
        if i == 0:
            return P_left(t)
        return fixed_left["S"](0, t)

    a0 = P_left(a)
    a1 = P_left.coef[1:]
    a2 = np.array([fixed_left["coefficients"][0] - a1[0]])

    return {
        "a0": a0,
        "coeffs": [a1, a2],
        "d_max": abs(d_left),
        "gra_d_max": fixed_left["d_max"],
        "case": 2,
        "S": S,
        "basis": [np.asarray(alt_left), fixed_left["basis"][0]],
        "knots": [a, theta, b],
        "optimal": fixed_left["exit_type"] == 1,
        "signs": fixed_left["signs"],
        "alternance_points": fixed_left["alternance_points"],
    }


def fixed_right_tail(f, a, theta, b, m, n):
    # Run Remez on the right interval [theta, b] to get the polynomial
    P_right, d_right, alt_right = remez.remez(f, theta, b, m)

    # Run GRA on the left interval [a, theta] with fixed right value P_right(theta)
    fixed_right = nadia_mod.gra(f, [a, theta], m, 1, fixed_right_value=P_right(theta))

    # Spline function combining the left spline and the right polynomial
    def S(i, t):
        if i == 0:
            return fixed_right["S"](0, t)
        return P_right(t)

    return {
        "a0": fixed_right["a0"],
        "coeffs": fixed_right["coefficients"],
        "d_max": abs(d_right),
        "gra_d_max": fixed_right["d_max"],
        "case": 3,
        "S": S,
        "basis": [np.asarray(fixed_right["basis"][0]), alt_right],
        "knots": [a, theta, b],
        "optimal": fixed_right["exit_type"] == 1,
        "signs": fixed_right["signs"],
    }


def psi_with_swap(f, a, b, theta, m, n):
    two_int_chain = nadia_mod.gra(f, [a, theta, b], m, n)

    # Case 1: found optimal spline across two intervals
    if two_int_chain["exit_type"] == 1:
        return {
            "alternance_points": two_int_chain["alternance_points"],
            "signs": two_int_chain["signs"],
            "coeffs": two_int_chain["coefficients"],
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
        fixed_left = fixed_left_tail(f, a, theta, b, m, n)

        # [a, theta] is wrong choice for minimal chain, swap to [theta, b]
        if abs(fixed_left["gra_d_max"]) > abs(fixed_left["d_max"]) + 1e-1:
            print(
                f"case 2: d_max={fixed_left['d_max']:.10f} < gra_d_max={fixed_left['gra_d_max']:.10f} at theta={theta:.10f}"
            )
            fixed_right = fixed_right_tail(f, a, theta, b, m, n)

            # Compare the differences between gra_d_max and d_max for both fixed_left and fixed_right
            fixed_right_diff = abs(fixed_right["gra_d_max"] - fixed_right["d_max"])
            fixed_left_diff = abs(fixed_left["gra_d_max"] - fixed_left["d_max"])

            # Return whichever fixed tail has the smaller difference between gra_d_max and d_max
            if fixed_right_diff < fixed_left_diff:
                print(
                    f"case 2a: fixed_right_diff={fixed_right_diff:.10f} < fixed_left_diff={fixed_left_diff:.10f} at theta={theta:.10f}"
                )
                return fixed_right

        return fixed_left

    # Case 3: tried to replace basis point in 2nd interval
    # Start with [theta, b] as minimal chain
    fixed_right = fixed_right_tail(f, a, theta, b, m, n)

    # [theta, b] is wrong choice for minimal chain, swap to [a, theta]
    if abs(fixed_right["gra_d_max"]) > abs(fixed_right["d_max"]) + 1e-1:
        print(
            f"case 3: d_max={fixed_right['d_max']:.10f} < gra_d_max={fixed_right['gra_d_max']:.10f} at theta={theta:.10f}"
        )
        fixed_left = fixed_left_tail(f, a, theta, b, m, n)

        # Compare the differences between gra_d_max and d_max for both fixed_left and fixed_right
        fixed_left_diff = abs(fixed_left["gra_d_max"] - fixed_left["d_max"])
        fixed_right_diff = abs(fixed_right["gra_d_max"] - fixed_right["d_max"])

        # Return whichever fixed tail has the smaller difference between gra_d_max and d_max
        if fixed_left_diff < fixed_right_diff:
            print(
                f"case 3a: fixed_left_diff={fixed_left_diff:.10f} < fixed_right_diff={fixed_right_diff:.10f} at theta={theta:.10f}"
            )
            return fixed_left

    return fixed_right


# def psi_without_swap(f, a, b, theta, m, n):
#     two_int_chain = nadia_mod.gra(f, [a, theta, b], m, n)

#     # Case 1: found optimal spline across two intervals
#     if two_int_chain["exit_type"] == 1:
#         return {
#             "d_max": two_int_chain["d_max"],
#             "case": 1,
#             "S": two_int_chain["S"],
#             "basis": two_int_chain["basis"],
#             "knots": [a, theta, b],
#             "optimal": True,
#         }

#     exit_interval = two_int_chain["exit_i_star"]

#     # Case 2: tried to replace basis point in 1st interval
#     # [a, theta] as minimal chain
#     if exit_interval == 0:
#         return fixed_left_tail(f, a, theta, b, m, n)

#     # Case 3: tried to replace basis point in 2nd interval
#     # [theta, b] as minimal chain
#     return fixed_right_tail(f, a, theta, b, m, n)


def directional_derivative(f, a, b, theta, psi_theta, m, n, h, psi=psi_with_swap):
    psi_theta_h = psi(f, a, b, theta + h, m, n)["d_max"]

    return (psi_theta_h - psi_theta) / abs(h)


def armijo(f, a, b, theta, psi_theta, m, n, g, d, rho=0.5, c=0.1, psi=psi_with_swap):
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
    f,
    a,
    b,
    m,
    n,
    x_min,
    x_max,
    h=0.1,
    rho=0.5,
    c=0.1,
    tolerance=1e-5,
    max_iter=30,
    psi=psi_with_swap,
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
    else:
        print(f"Maximum iterations ({max_iter}) reached in opt()")

    psi_opt = psi(f, a, b, theta, m, n)

    return theta, psi_opt


def plot_psi(
    f,
    f_label,
    a,
    b,
    m,
    n,
    theta_start,
    theta_end,
    file_name,
    step=0.05,
    psi=psi_with_swap,
):
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

    if "with_swap" in str(file_name):
        plt.title(r"$\psi(\theta)$ for " + f"{f_label} with m={m}, k={k} (with swap)")
    else:
        plt.title(
            r"$\psi(\theta)$ for " + f"{f_label} with m={m}, k={k} (without swap)"
        )

    plt.grid(alpha=0.3)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.tight_layout()

    plt.savefig(file_name, dpi=300)
    plt.show()


if __name__ == "__main__":
    function_name = "sin_weird"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a = 0
    b = 12

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
    print("x_max:", x_max)
    print("d_n:", d_n)

    # Find optimal theta
    theta_opt, psi_result = opt(f, a, b, m, n, 7.8, x_max)

    print(
        f"optimal theta: {theta_opt:.10f}, psi(theta_opt): {psi_result['d_max']:.10f}"
    )

    def case(case_num):
        if case_num == 1:
            return "two intervals"
        elif case_num == 2:
            return "fixed left"
        elif case_num == 3:
            return "fixed right"

    status = (
        f"optimal, {case(psi_result['case'])}"
        if psi_result["optimal"]
        else f"not optimal, {case(psi_result['case'])}"
    )

    plotting.plot_detailed(
        f,
        psi_result["S"],
        a,
        b,
        knots=psi_result["knots"],
        points=psi_result["basis"],
        f_label=f_label,
        approximation_label="Spline approximation",
        points_label="Basis points",
        title=(
            f"Degree-{m} spline approximation of {f_label}. "
            f"{k} internal knots ({status}). "
            f"Max abs deviation: {psi_result['d_max']:.5f}."
        ),
        file_name=f"{function_name}_a{a}_b{b}_knot{theta_opt:.5f}_k{k}_m{m}_psi(t){psi_result['d_max']:.5f}.png",
    )

    # # gradient test shit
    # result = psi_with_swap(f, a, b, 7, m, n)
    # print(f"result: {result}")
    # # G = nadia.build_gradients(
    # #     result["alternance_points"], result["knots"], m, a, result["signs"]
    # # )
    # d2 = nadia.find_descent_direction(
    #     result["alternance_points"],
    #     result["knots"],
    #     m,
    #     result["coeffs"],
    #     result["signs"],
    # )

    # print(f"descent direction: {d2}")

    # status = (
    #     f"optimal, {case(result['case'])}"
    #     if result["optimal"]
    #     else f"not optimal, {case(result['case'])}"
    # )

    # nadia.plot(
    #     f,
    #     f_label,
    #     a,
    #     b,
    #     n,
    #     result["S"],
    #     result["knots"],
    #     result["basis"],
    #     m,
    #     k,
    #     result["d_max"],
    #     status,
    #     plot_dir
    #     / f"{function_name}_a{a}_b{b}_knot{9:.5f}_k{k}_m{m}_psi(t){result["d_max"]:.5f}_with_swap.png",
    #     f"Degree-{m} spline approximation of {f_label}. {k} internal knots ({status}). Max abs deviation: {result["d_max"]:.5f} (with swap)",
    # )

    # r = psi_with_swap(f, a, b, 7, m, n)
    # d = directional_derivative(f, a, b, 7, r["d_max"], m, n, 0.01)
    # print(d)

    # # psi plot with swap
    # start_theta = a + 0.1
    # end_theta = b - 0.1

    # plot_psi(
    #     f,
    #     f_label,
    #     a,
    #     b,
    #     m,
    #     n,
    #     start_theta,
    #     end_theta,
    #     plot_dir
    #     / f"PSI_{function_name}_a{a}_b{b}_s{start_theta}_e{end_theta}_k{k}_m{m}_with_swap.png",
    # )
