import numpy as np

import GRA_mod
import nurnberger_mod
import plotting
import remez
import Spline
import test_functions


def fixed_left_tail(f, a, theta, b, m):
    """Compute the best spline approximation with a fixed left tail at theta."""
    # Best polynomial approximation on [a, theta]
    left_approx = remez.run(f, a, theta, m)

    # GRA on [theta, b], fixed to the left polynomial value at theta
    right_result = GRA_mod.run(
        f, [theta, b], m, 1, fixed_left_value=left_approx.g(theta)
    )

    right_approx = right_result["approximation"]

    # Combine the left polynomial and right spline
    S_left = Spline.Spline([a, theta], [left_approx.g])
    S = S_left.concatenate(right_approx.g)

    # Combine their bases
    basis = [left_approx.basis, right_approx.basis[0]]

    # Create the final approximation object
    approx = Spline.Approximation(f, S, [a, b], basis=basis)

    _, _, d_left = left_approx.maxdeviation()
    _, _, d_gra = right_approx.maxdeviation()

    return {
        "approximation": approx,
        "d_max": abs(d_left),
        "gra_d_max": abs(d_gra),
        "case": 2,
        "optimal": right_result["exit_type"] == 1,
    }


def fixed_right_tail(f, a, theta, b, m):
    """Compute the best spline approximation with a fixed right tail at theta."""
    # Best polynomial approximation on [theta, b]
    right_approx = remez.run(f, theta, b, m)

    # GRA on [a, theta], fixed to the right polynomial value at theta
    left_result = GRA_mod.run(
        f, [a, theta], m, 1, fixed_right_value=right_approx.g(theta)
    )

    left_approx = left_result["approximation"]

    # Combine the left spline and right polynomial
    S_right = Spline.Spline([theta, b], [right_approx.g])
    S = left_approx.g.concatenate(S_right)

    # Combine bases
    basis = [left_approx.basis[0], right_approx.basis]

    approx = Spline.Approximation(f, S, [a, b], basis=basis)

    # Get the maximum deviations for both the left and right approximations
    _, _, d_right = right_approx.maxdeviation()
    _, _, d_gra = left_approx.maxdeviation()

    return {
        "approximation": approx,
        "d_max": abs(d_right),
        "gra_d_max": abs(d_gra),
        "case": 3,
        "optimal": left_result["exit_type"] == 1,
    }


def Psi_bar(f, a, b, theta, m, n, verbose=False):
    """Compute the maximum deviation of the best spline approximation with a knot at theta."""
    two_int_chain = GRA_mod.run(f, [a, theta, b], m, n)

    # Case 1: found optimal spline across two intervals
    if two_int_chain["exit_type"] == 1:
        approx = two_int_chain["approximation"]
        if verbose:
            print(
                f"case 1: optimal spline found across two intervals at theta={theta:.10f}"
            )

        _, _, d_star = approx.maxdeviation()

        return {
            "approximation": approx,
            "d_max": abs(d_star),
            "case": 1,
            "optimal": True,
        }

    exit_interval = two_int_chain["exit_i_star"]

    # Case 2: tried to replace basis point in 1st interval
    # Start with [a, theta] as minimal chain
    if exit_interval == 0:
        fixed_left = fixed_left_tail(f, a, theta, b, m)

        # [a, theta] is wrong choice for minimal chain, try swap to [theta, b]
        if fixed_left["gra_d_max"] > fixed_left["d_max"] + 1e-1:
            if verbose:
                print(
                    f"case 2: d_max={fixed_left['d_max']:.10f} < gra_d_max={fixed_left['gra_d_max']:.10f} at theta={theta:.10f}"
                )

            fixed_right = fixed_right_tail(f, a, theta, b, m)

            # Compare the differences between gra_d_max and d_max for both fixed_left and fixed_right
            fixed_right_diff = abs(fixed_right["gra_d_max"] - fixed_right["d_max"])
            fixed_left_diff = abs(fixed_left["gra_d_max"] - fixed_left["d_max"])

            # Return whichever fixed tail has the smaller difference between gra_d_max and d_max
            if fixed_right_diff < fixed_left_diff:
                if verbose:
                    print(
                        f"case 2a: fixed_right_diff={fixed_right_diff:.10f} < fixed_left_diff={fixed_left_diff:.10f} at theta={theta:.10f}"
                    )
                return fixed_right

        return fixed_left

    # Case 3: tried to replace basis point in 2nd interval
    # Start with [theta, b] as minimal chain
    fixed_right = fixed_right_tail(f, a, theta, b, m)

    # [theta, b] is wrong choice for minimal chain, try [a, theta]
    if fixed_right["gra_d_max"] > fixed_right["d_max"] + 1e-1:
        if verbose:
            print(
                f"case 3: d_max={fixed_right['d_max']:.10f} < gra_d_max={fixed_right['gra_d_max']:.10f} at theta={theta:.10f}"
            )

        fixed_left = fixed_left_tail(f, a, theta, b, m)

        # Compare the differences between gra_d_max and d_max for both fixed_left and fixed_right
        fixed_left_diff = abs(fixed_left["gra_d_max"] - fixed_left["d_max"])
        fixed_right_diff = abs(fixed_right["gra_d_max"] - fixed_right["d_max"])

        # Return whichever fixed tail has the smaller difference between gra_d_max and d_max
        if fixed_left_diff < fixed_right_diff:
            if verbose:
                print(
                    f"case 3a: fixed_left_diff={fixed_left_diff:.10f} < fixed_right_diff={fixed_right_diff:.10f} at theta={theta:.10f}"
                )
            return fixed_left

    return fixed_right


def directional_derivative(f, a, b, theta, psi_theta, m, n, h):
    """Approximate the directional derivative of psi at theta in the direction of h using finite differences."""
    psi_theta_h = Psi_bar(f, a, b, theta + h, m, n)["d_max"]

    return (psi_theta_h - psi_theta) / abs(h)


def armijo_line_search(
    f, a, b, theta, psi_theta, m, n, g, d, rho=0.5, c=0.1, verbose=False
):
    """Armijo line search to find the next theta in the direction of d."""
    if d < 0:
        alpha = a - theta / d
    else:
        alpha = (b - theta) / d

    while alpha > 1e-8:
        # Compute the next theta using the step size alpha
        theta_next = theta + alpha * d

        # Check if theta_next is within the interval [a, b]
        if theta_next <= a or theta_next >= b:
            alpha *= rho
            continue

        # Compute psi at the new theta
        psi_next = Psi_bar(f, a, b, theta_next, m, n, verbose=verbose)["d_max"]

        if verbose:
            print(
                f"alpha={alpha:.10f}, theta_next={theta_next:.10f}, psi_next={psi_next:.10f}, psi_theta={psi_theta:.10f}, g={g:.10f}, d={d:.10f}\n"
            )

        # Check the Armijo condition
        if psi_next <= psi_theta + c * alpha * abs(d) * g:
            return theta_next

        # If the Armijo condition is not satisfied, reduce alpha and try again
        alpha *= rho

    return theta


def descent_algortihm(
    f,
    a,
    b,
    m,
    n,
    theta_min,
    h=0.1,
    rho=0.5,
    c=0.1,
    tolerance=1e-5,
    max_iter=30,
    verbose=False,
):
    """Find the optimal theta that minimises Psi_bar(theta) using directional derivatives and Armijo line search."""
    theta = theta_min
    iterates = [theta]

    for i in range(max_iter):
        # Compute the approximate directional derivatives at the current theta
        psi_theta = Psi_bar(f, a, b, theta, m, n, verbose=verbose)["d_max"]
        g_plus = directional_derivative(f, a, b, theta, psi_theta, m, n, h)
        g_minus = directional_derivative(f, a, b, theta, psi_theta, m, n, -h)

        if verbose:
            print(
                f"i={i}, theta={theta:.10f}, psi={psi_theta:.10f} g_plus={g_plus:.10f}, g_minus={g_minus:.10f}"
            )

        # Determine the search direction based on the directional derivatives
        if g_plus <= g_minus:
            d = h
            g = g_plus
        else:
            d = -h
            g = g_minus

        if g >= 0 or abs(g) < tolerance:
            break

        # Use Armijo line search to find the next theta
        theta_next = armijo_line_search(
            f, a, b, theta, psi_theta, m, n, g, d, rho, c, verbose=verbose
        )
        iterates.append(theta_next)

        # Check for convergence based on the change in theta
        if abs(theta_next - theta) < tolerance:
            break

        theta = theta_next

    else:
        print(f"Maximum iterations ({i + 1}) reached in descent_algortihm()")

    psi_theta = Psi_bar(f, a, b, theta, m, n, verbose=verbose)

    if verbose:
        print(
            f"Optimal theta found: {theta:.10f} with Psi_bar(theta)={psi_theta['d_max']:.10f}"
        )

    return theta, psi_theta, i + 1, iterates


if __name__ == "__main__":
    # Example usage
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]
    a, b = test_functions.INTERVALS[function_name]

    k = 1
    n = k + 1
    m = 1

    # Get starting knots from Nurnberger's modified algorithm
    approx, theta_min = nurnberger_mod.discontinuous_spline(f, a, b, k, m)

    # If theta_min is None, use the first internal knot from the approximation as the starting point
    if theta_min is None:
        theta_min = approx.g.knots[1]

    # Find optimal theta
    theta_opt, psi_result, iterations, theta_path = descent_algortihm(
        f, a, b, m, n, theta_min, verbose=True
    )

    # Load precomputed psi(theta) values from the .npz file
    data = np.load(f"psi_surfaces/psi_surface_{function_name}_k{k}_m{m}.npz")
    thetas = data["theta_values"]
    psi_values = data["psi_values"]

    plotting.plot_objective_psi_bar(
        thetas,
        psi_values,
        theta_found=theta_opt,
        theta_path=theta_path,
        psi_found=psi_result["d_max"],
        title=r"Descent path on $\overline{{\Psi}}(\theta)$",
        file_name=f"psi_opt_path_{function_name}_k{k}_m{m}",
    )

    plotting.plot_single(
        psi_result["approximation"],
        points=psi_result["approximation"].basis,
        f_label=function_label,
        approximation_label=rf"$S_{{{m},{k}}}(t)$",
        file_name=f"{function_name}_k{k}_m{m}",
    )
