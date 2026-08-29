import numpy as np

import nadia
import test_functions

TOL = 1e-5


# allow basis point to be replaced by internal knot
def exchange(i, t_star, d_star, f, S, basis, knots, n):

    t_star_sign = np.sign(d_star)

    # Check if t* is an internal knot
    knot_index = None

    for j in range(1, n):
        if abs(t_star - knots[j]) <= TOL:
            knot_index = j
            t_star = knots[j]
            print(f"t* is an internal knot at {t_star} with sign {t_star_sign}")
            break

    # t* cannot be a basis point
    if any(np.any(np.isclose(b, t_star)) for b in basis):
        print(f"t*={t_star} is already a basis point in interval {i}")
        return None

    # Internal knot: look in both adjacent intervals
    if knot_index is not None:

        left_i = knot_index - 1
        right_i = knot_index

        left_pt = basis[left_i][-1]
        right_pt = basis[right_i][0]

        left_sign = np.sign(nadia.deviation(f, S, left_i, left_pt))
        right_sign = np.sign(nadia.deviation(f, S, right_i, right_pt))

        # update i to the interval of the basis point that has the same sign as t*
        if left_sign == t_star_sign:
            i = left_i
            t_tilde = left_pt

        elif right_sign == t_star_sign:
            i = right_i
            t_tilde = right_pt

        else:
            print("No valid basis point to replace")
            return None

    # Normal point: look only in current interval
    else:

        basis_points = basis[i]

        left = basis_points[basis_points < t_star]
        right = basis_points[basis_points > t_star]

        left_pt = left[-1] if len(left) else None
        right_pt = right[0] if len(right) else None

        t_tilde = None

        if (
            left_pt is not None
            and np.sign(nadia.deviation(f, S, i, left_pt)) == t_star_sign
        ):
            t_tilde = left_pt

        elif (
            right_pt is not None
            and np.sign(nadia.deviation(f, S, i, right_pt)) == t_star_sign
        ):
            t_tilde = right_pt

        if t_tilde is None:
            print("no valid basis point to replace")
            return None

    # replace basis point t_tilde with t_star in interval i
    basis_points = basis[i]

    basis_deviations = np.array([nadia.deviation(f, S, i, t) for t in basis_points])

    max_basis_deviation = np.max(np.abs(basis_deviations))

    if abs(d_star) <= max_basis_deviation + TOL:
        print(
            f"Absolute deviation at t*, {d_star} is <= max absolute deviation at basis points in interval {i}, {np.max(np.abs(basis_deviations))}"
        )
        return None

    new_basis = [b.copy() for b in basis]

    new_basis[i] = np.sort(
        np.append(basis_points[~np.isclose(basis_points, t_tilde)], t_star)
    )

    return new_basis


# generalised Remez algorithm
def gra(f, knots, m, n):
    basis = nadia.step_zero(knots, m, n)
    S, delta, a0, a = nadia.step_one(knots, basis, m, n, f)

    optimal = False
    exit_type = None

    for _ in range(100):
        (max_i, t_max, d_max), (min_i, t_min, d_min), (i_star, t_star, d_star) = (
            nadia.find_extrema_overall(f, S, knots, n)
        )

        optimal, pts, signs, chain = nadia.check_exit_1(f, S, knots, n, m, abs(d_star))

        if optimal:
            exit_type = 1
            break

        # modified exchange
        new_basis = exchange(i_star, t_star, d_star, f, S, basis, knots, n)

        if new_basis is None:
            exit_type = 2
            break

        basis = new_basis
        S, delta, a0, a = nadia.step_one(knots, basis, m, n, f)

    # Final maximum absolute deviation
    _, _, (i_star, t_star, d_star) = nadia.find_extrema_overall(f, S, knots, n)

    return {
        "a0": a0,
        "coefficients": a,
        "knots": knots,
        "basis": basis,
        "S": S,
        "delta": delta,
        "d_max": abs(d_star),
        "t_star": t_star,
        "optimal": optimal,
        "exit_type": exit_type,
        "alternance_points": pts,
    }


if __name__ == "__main__":
    function_name = "sin3t"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = 2, 6
    k = 1  # number of internal fixed knots
    m = 2  # degree of polynomial to fit in each subinterval
    n = k + 1  # number of subintervals

    # Choose intial knots
    knots = [2, 3.67555847, 6]
    print(f"Function: {function_name}")
    print(f"Knots: {knots}")

    result = gra(f, knots, m, n)

    if result["exit_type"] == 1:
        print("EXIT 1 (spline is optimal).")
    elif result["exit_type"] == 2:
        print("EXIT 2 (no valid exchange).")

    print("basis:            ", result["basis"])
    print("alternance points:", result["alternance_points"])
    print(f"Max abs deviation: {result["d_max"]:.5f} at t = {result["t_star"]:.5f}")

    status = "Optimal" if result["optimal"] else "Not optimal"
    nadia.plot(
        f,
        f_label,
        a,
        b,
        n,
        result["S"],
        knots,
        result["basis"],
        m,
        k,
        status,
        f"mod_{function_name}_k{k}_m{m}.png",
    )
