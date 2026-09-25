import numpy as np

import GRA
import plotting
import test_functions

TOL = 1e-5


def exchange(i, t_star, d_star, f, approx, basis, knots, n, verbose=False):
    """Modified generalised VP basis exchange"""
    t_star_sign = np.sign(d_star)

    # Check if t* is an internal knot
    knot_index = None
    for j in range(1, n):
        if abs(t_star - knots[j]) <= TOL:
            knot_index = j
            t_star = knots[j]
            break

    # t* cannot be a basis point
    if any(np.any(np.isclose(b, t_star)) for b in basis):
        if verbose:
            print(f"EXIT 2: t*={t_star} is already a basis point.")
        return None

    # Case 1: t* is an internal knot - look in both adjacent intervals
    if knot_index is not None:
        left_i = knot_index - 1
        right_i = knot_index

        left_pt = basis[left_i][-1]
        right_pt = basis[right_i][0]

        left_sign = np.sign(approx.deviation(left_pt))
        right_sign = np.sign(approx.deviation(right_pt))

        # update i to the interval of the basis point that has the same sign as t*
        if left_sign == t_star_sign:
            i = left_i
            t_tilde = left_pt
        elif right_sign == t_star_sign:
            i = right_i
            t_tilde = right_pt
        else:
            if verbose:
                print(
                    f"EXIT 2 (internal knot): No basis point with same sign as t*={t_star} in adjacent intervals {left_i} and {right_i}"
                )
            return None

    # Case 2: t* is a normal point - look only in current interval
    else:
        basis_points = basis[i]

        left = basis_points[basis_points < t_star]
        right = basis_points[basis_points > t_star]

        left_pt = left[-1] if len(left) else None
        right_pt = right[0] if len(right) else None

        t_tilde = None

        if left_pt is not None and np.sign(approx.deviation(left_pt)) == t_star_sign:
            t_tilde = left_pt
        elif (
            right_pt is not None and np.sign(approx.deviation(right_pt)) == t_star_sign
        ):
            t_tilde = right_pt

        # For a single interval problem, allow exchange with basis point at opposite end of interval if it has same sign as t*
        if t_tilde is None and n == 1:
            if t_star < basis_points[0]:
                end_pt = basis_points[-1]
            elif t_star > basis_points[-1]:
                end_pt = basis_points[0]
            else:
                end_pt = None

            if end_pt is not None and np.sign(approx.deviation(end_pt)) == t_star_sign:
                t_tilde = end_pt

        if t_tilde is None:
            if verbose:
                print(
                    f"EXIT 2: No basis point with same sign as t*={t_star} in interval {i}"
                )
            return None

    # Get the basis points in the current interval
    basis_points = basis[i]

    # Get max absolute deviation at basis points in interval i
    basis_deviations = np.array([approx.deviation(t) for t in basis_points])
    max_basis_deviation = np.max(np.abs(basis_deviations))

    # Absolute deviation at t* must be greater than the absolute deviation at any of the basis points in that interval
    if abs(d_star) <= np.max(np.abs(basis_deviations)) + 1e-4:
        if verbose:
            print(
                f"EXIT 2: Absolute deviation at t*, {abs(d_star)} is <= max absolute deviation at basis points in interval {i}, {max_basis_deviation + 1e-4}"
            )
        return None

    # Replace basis point t_tilde with t_star in interval i
    new_basis = [b.copy() for b in basis]
    new_basis[i] = np.sort(
        np.append(basis_points[~np.isclose(basis_points, t_tilde)], t_star)
    )

    return new_basis


def run(
    f,
    knots,
    m,
    n,
    exchange_function=exchange,
    fixed_left_value=None,
    fixed_right_value=None,
    verbose=False,
):
    """Run the modified GRA algorithm with the provided exchange function."""
    return GRA.run(
        f,
        knots,
        m,
        n,
        exchange_function=exchange_function,
        fixed_left_value=fixed_left_value,
        fixed_right_value=fixed_right_value,
        verbose=verbose,
    )


if __name__ == "__main__":
    # evaluating Psi bar example, section 3.2 of report
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]
    function_label = test_functions.FUNCTION_LABELS[function_name]
    a, b = test_functions.INTERVALS[function_name]

    k = 3  # number of internal fixed knots
    m = 1  # degree of polynomial to fit in each subinterval
    n = k + 1  # number of subintervals

    result = GRA.run(
        f, np.linspace(a, b, k + 2), m, n, exchange_function=exchange, verbose=True
    )

    plotting.plot_duo(
        result["approximation"],
        points=result["approximation"].basis,
        f_label=function_label,
        deviation_label=rf"$f_{1}(t)-s^*_{{{m}}}(t)$",
        approximation_label=rf"$s^*_{{{m}}}(t)$",
        approximation_title=rf"Best fixed-knot spline $s^*_{{{m}}}$",
        deviation_title=rf"Corresponding deviation $f_1-s^*_{{{m}}}$",
        highlight_knots=[-0.5, 0.5],
        file_name=f"duo_gra_mod_{function_name}_k{k}_m{m}",
    )
