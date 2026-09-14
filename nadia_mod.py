# Nadia's algorithm with basis exchange modifications ONLY
# Rename file eventually
# Modified exchange() to allow basis point to be replaced by internal knot and fixed tails
# and allow cyclic exchange for single interval problems
import numpy as np

import nadia_original
import plotting
import test_functions

TOL = 1e-5


def exchange(i, t_star, d_star, f, S, basis, knots, n, verbose=False):

    t_star_sign = np.sign(d_star)

    # Check if t* is an internal knot
    knot_index = None

    for j in range(1, n):
        if abs(t_star - knots[j]) <= TOL:
            knot_index = j
            t_star = knots[j]
            # print(f"t* is an internal knot at {t_star} with sign {t_star_sign}")
            break

    # t* cannot be a basis point
    if any(np.any(np.isclose(b, t_star)) for b in basis):
        print(f"EXIT 2: t*={t_star} is already a basis point in interval {i}")
        return None

    # Case 1: t* is an internal knot - look in both adjacent intervals
    if knot_index is not None:

        left_i = knot_index - 1
        right_i = knot_index

        left_pt = basis[left_i][-1]
        right_pt = basis[right_i][0]

        left_sign = np.sign(S.deviation(left_pt))
        right_sign = np.sign(S.deviation(right_pt))

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

        if left_pt is not None and np.sign(S.deviation(left_pt)) == t_star_sign:
            t_tilde = left_pt
        elif right_pt is not None and np.sign(S.deviation(right_pt)) == t_star_sign:
            t_tilde = right_pt

        # For a single interval problem, allow exchange with basis point at opposite end of interval if it has same sign as t*
        if t_tilde is None and n == 1:
            if t_star < basis_points[0]:
                end_pt = basis_points[-1]
            elif t_star > basis_points[-1]:
                end_pt = basis_points[0]
            else:
                end_pt = None

            if end_pt is not None and np.sign(S.deviation(end_pt)) == t_star_sign:
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
    basis_deviations = np.array([S.deviation(t) for t in basis_points])
    max_basis_deviation = np.max(np.abs(basis_deviations))

    # Absolute deviation at t* must be greater than the absolute deviation at any of the basis points in that interval
    if abs(d_star) <= np.max(np.abs(basis_deviations)) + TOL:
        if verbose:
            print(
                f"EXIT 2: Absolute deviation at t*, {abs(d_star)} is <= max absolute deviation at basis points in interval {i}, {max_basis_deviation + TOL}"
            )
        return None

    # Replace basis point t_tilde with t_star in interval i
    new_basis = [b.copy() for b in basis]
    new_basis[i] = np.sort(
        np.append(basis_points[~np.isclose(basis_points, t_tilde)], t_star)
    )

    return new_basis


def gra(
    f,
    knots,
    m,
    n,
    exchange_function=exchange,
    fixed_left_value=None,
    fixed_right_value=None,
):
    return nadia_original.gra(
        f,
        knots,
        m,
        n,
        exchange_function=exchange_function,
        fixed_left_value=fixed_left_value,
        fixed_right_value=fixed_right_value,
    )


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 1
    m = 1
    n = k + 1

    # Choose initial knots
    knots = [a, 0.38, b]

    print(f"Function: {function_name}")
    print(f"Knots: {knots}")

    # Pass the modified exchange function to gra
    result = nadia_original.gra(f, knots, m, n, exchange_function=exchange)

    if result["exit_type"] == 1:
        print("EXIT 1 (spline is optimal). Chain: ", result["chain"])

    print("basis:            ", result["approximation"].basis)
    print("alternance points:", result["approximation"].alternancesequence()[0])

    d_max = result["approximation"].maxdeviation()[2]
    print(
        f"Max abs deviation: {d_max:.5f} at t = {result['approximation'].maxdeviation()[1]:.5f}"
    )

    status = "Optimal" if result["exit_type"] == 1 else "Non-optimal"

    plotting.plot_duo(
        result["approximation"],
        points=result["approximation"].basis,
        f_label=f_label,
        approximation_label=rf"$S_{{{m}}}(t)$",
        title=(
            f"Degree-{m} spline approximation of {f_label}. "
            f"{k} internal knots ({status}). "
            f"Max abs deviation: {d_max:.5f}."
        ),
        file_name=f"duo_mod_{function_name}_k{k}_m{m}.png",
    )

    # plotting.plot_report(
    #     f,
    #     result["S"],
    #     a,
    #     b,
    #     knots=knots,
    #     points=result["alternance_points"],
    #     f_label=f_label,
    #     approximation_label=rf"$S_{{{m}}}(t)$",
    #     points_label="Alternance points",
    #     file_name=f"mod_report_{function_name}_k{k}_m{m}.png",
    # )
