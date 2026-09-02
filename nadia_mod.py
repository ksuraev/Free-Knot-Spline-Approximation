# Modified exchange() to allow basis point to be replaced by internal knot and fixed tails
import numpy as np

import nadia
import test_functions

TOL = 1e-5


# Form intial basis - m per internal subinterval, m+1 per endpoint subinterval. Internal spline knots are excluded from the basis
def step_zero(knots, m, n, fixed_left_tail=False, fixed_right_tail=False):
    basis = []

    for i in range(n):
        start = knots[i]
        end = knots[i + 1]

        # default
        s = 0 if i == 0 else 1
        e = -1 if i == n - 1 else -2

        # fixed left tail - exclude the left endpoint from basis
        if i == 0 and fixed_left_tail:
            s = 1

        # fixed right tail - exclude the right endpoint from basis
        if i == n - 1 and fixed_right_tail:
            e = -2

        local_basis = np.linspace(start, end, m + 3)[s:e]
        basis.append(local_basis)

    # print([len(x) for x in basis])
    # print(basis)
    return basis


# step 1 : Construct full matrix (𝝲+2 rows) and solve for polynomial spline S and delta
def step_one(knots, basis, m, n, f, fixed_left_value=None, fixed_right_value=None):
    P_matrices = [nadia.build_P_matrix(i, basis, knots, m) for i in range(n)]
    Q_rows = [nadia.build_Q_row(i, knots, m) for i in range(n)]

    num_constraints = (1 if fixed_left_value is not None else 0) + (
        1 if fixed_right_value is not None else 0
    )

    basis_counts = [len(b) for b in basis]
    total_rows = sum(basis_counts) + num_constraints

    A = np.zeros((total_rows, total_rows))
    b = np.zeros(total_rows)

    row = 0
    sign = -1

    for interval in range(n):
        for r in range(basis_counts[interval]):
            # First element in each row is 1
            A[row, 0] = 1.0

            # Fill Q rows Q_1 to Q_{i-1} for the current interval
            for col_start in range(interval):
                len_block = 1 + col_start * m
                A[row, len_block : len_block + m] = Q_rows[col_start]

            # Fill this intervals P_i matrix
            len_block = 1 + interval * m
            A[row, len_block : len_block + m] = P_matrices[interval][r]

            # Fill delta column with alternating sign
            A[row, -1] = sign

            # Fill b with function values at basis points
            b[row] = f(basis[interval][r])

            sign *= -1
            row += 1

    # A[row] = [1, 0, 0, ..., 0] for fixed left value
    if fixed_left_value is not None:
        A[row, 0] = 1.0
        b[row] = fixed_left_value
        row += 1

    # A[row] = [1, Q_1, Q_2, ..., Q_n] for fixed right value
    if fixed_right_value is not None:
        A[row, 0] = 1.0

        for interval in range(n):
            len_block = 1 + interval * m
            A[row, len_block : len_block + m] = Q_rows[interval]

        b[row] = fixed_right_value
        row += 1

    solution = np.linalg.solve(A, b)
    a0 = solution[0]
    a = solution[1:-1].reshape(n, m)

    delta = solution[-1]

    def S(i, t):
        # don't have a better name for 'last_term' yet
        last_term = a0 if i == 0 else S(i - 1, knots[i])
        return sum(a[i, j] * (t - knots[i]) ** (j + 1) for j in range(m)) + last_term

    return S, delta, a0, a


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


# Tarashnin's necessary and sufficient optimality conditions
def check_exit_1(
    f, S, knots, n, m, global_max, fixed_left_tail=False, fixed_right_tail=False
):
    pts, signs = nadia.find_alternance_points(f, S, knots, n, global_max)

    pts_and_signs = list(zip(pts, signs))

    def points_alternate(points):
        # true if the signs of the points alternate
        point_signs = [s for _, s in points]

        return len(point_signs) >= 2 and all(
            point_signs[k] != point_signs[k + 1] for k in range(len(point_signs) - 1)
        )

    # condition (i): in one subinterval, there is at least m+2 alternance points
    for i in range(n):

        points = [(t, s) for t, s in pts_and_signs if (knots[i] <= t <= knots[i + 1])]

        required = m + 2

        if fixed_left_tail and i == 0:
            required -= 1
        if fixed_right_tail and i == n - 1:
            required -= 1

        if len(points) >= required and points_alternate(points):
            print(
                f"Condition (i) satisfied in interval {i} with {len(points)} alternance points."
            )
            return True, pts, signs, (i, i)

    # condition (ii)
    for i in range(n):
        for j in range(i + 1, n):

            chain_intervals = []

            for k in range(i, j + 1):

                if k == i:
                    # First interval: [θ_i, θ_{i+1}]
                    points = [
                        (t, s)
                        for t, s in pts_and_signs
                        if (knots[k] <= t <= knots[k + 1])
                    ]

                else:
                    # Subsequent intervals: (θ_k, θ_{k+1}]
                    points = [
                        (t, s)
                        for t, s in pts_and_signs
                        if (knots[k] < t <= knots[k + 1])
                    ]

                chain_intervals.append(points)

            # Check distribution
            # first interval requires m+1 points, unless its the left tail and fixed, then it requires m points
            first_required = m + 1
            if fixed_left_tail and i == 0:
                first_required -= 1

            if len(chain_intervals[0]) < first_required:
                continue

            # Last interval requires m+1 points, unless its the right tail and fixed, then it requires m points
            last_required = m + 1

            if fixed_right_tail and j == n - 1:
                last_required -= 1

            if len(chain_intervals[-1]) < last_required:
                continue

            # Intermediate intervals: at least m alternance points
            if not all(
                len(chain_intervals[k]) >= m for k in range(1, len(chain_intervals) - 1)
            ):
                continue

            combined = [point for interval in chain_intervals for point in interval]

            total_required = m * (j - i + 1) + 2
            if fixed_left_tail and i == 0:
                total_required -= 1
            if fixed_right_tail and j == n - 1:
                total_required -= 1
            if len(combined) >= total_required and points_alternate(combined):
                print(
                    f"Condition (ii) satisfied in intervals {i}-{j} with {len(combined)} alternance points"
                )
                return (True, pts, signs, (i, j))

    return False, pts, signs, None


# generalised Remez algorithm
def gra(f, knots, m, n, fixed_left_value=None, fixed_right_value=None):
    fixed_left_tail = fixed_left_value is not None
    fixed_right_tail = fixed_right_value is not None

    basis = step_zero(knots, m, n, fixed_left_tail, fixed_right_tail)
    S, delta, a0, a = step_one(
        knots, basis, m, n, f, fixed_left_value, fixed_right_value
    )

    optimal = False
    exit_type = None

    for _ in range(100):
        (max_i, t_max, d_max), (min_i, t_min, d_min), (i_star, t_star, d_star) = (
            nadia.find_extrema_overall(f, S, knots, n)
        )

        optimal, pts, signs, chain = check_exit_1(
            f, S, knots, n, m, abs(d_star), fixed_left_tail, fixed_right_tail
        )

        if optimal:
            exit_type = 1
            break

        # modified exchange
        new_basis = exchange(i_star, t_star, d_star, f, S, basis, knots, n)

        if new_basis is None:
            exit_type = 2
            break

        basis = new_basis
        S, delta, a0, a = step_one(
            knots, basis, m, n, f, fixed_left_value, fixed_right_value
        )

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
        "chain": chain,
    }


if __name__ == "__main__":
    function_name = "sin3t"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = 0, 4
    k = 1
    m = 2
    n = k + 1

    # Choose intial knots
    knots = [a, 2, b]

    print(f"Function: {function_name}")
    print(f"Knots: {knots}")

    result = gra(f, knots, m, n)

    if result["exit_type"] == 1:
        print("EXIT 1 (spline is optimal). Minimal chain: ", result["chain"])
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

# 2nd numerical experiment from Poussin paper
# if __name__ == "__main__":
#     function_name = "sin"
#     f, f_label = test_functions.TEST_FUNCTIONS[function_name]

#     # construct SP1 on [2, 6]
#     knots_1 = [2, 3.43177734, 6]
#     m = 2
#     n_1 = 2

#     result_1 = gra(f, knots_1, m, n_1)
#     print("Optimal:", result_1["optimal"])
#     print("S(1):", result_1["S"](0, 2))
#     print("Max abs deviation:", result_1["d_max"])
#     print("Basis:", result_1["basis"])

#     SP1 = result_1["S"]
#     fixed_value = SP1(0, 2)
#     print("SP1(2) =", fixed_value)

#     # GRAFT on [0, 2] with fixed right value SP1(2)
#     knots_2 = [0, 2]
#     n_2 = 1

#     result_2 = gra(f, knots_2, m, n_2, fixed_right_value=fixed_value)

#     print("Optimal:", result_2["optimal"])
#     print("Fixed value:", fixed_value)
#     print("S(2):", result_2["S"](0, 2))
#     print("Max abs deviation:", result_2["d_max"])
#     print("Basis:", result_2["basis"])

#     def combined_S(i, t):
#         if i == 0:
#             return result_2["S"](0, t)

#         return result_1["S"](i - 1, t)

#     combined_knots = result_2["knots"] + result_1["knots"][1:]
#     combined_basis = result_2["basis"] + result_1["basis"]

#     n = 3
#     k = 2

#     status = "Optimal" if result_1["optimal"] and result_2["optimal"] else "Not optimal"
#     nadia.plot(
#         f,
#         f_label,
#         0,
#         6,
#         n,
#         combined_S,
#         combined_knots,
#         combined_basis,
#         m,
#         k,
#         status,
#         f"GRAFT_{function_name}_k{k}_m{m}.png",
#     )
