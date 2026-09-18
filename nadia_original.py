# rename file
# GRA and GRAFT (unmodified)
import numpy as np

import plotting
import Spline
import test_functions

TOL = 1e-5
ALTERNANCE_TOL = 1e-5


# TODO: check about fixed tails - we need m points in fixed tails?
def step_zero(knots, m, n, fixed_left_tail=False, fixed_right_tail=False):
    """Form initial basis for the spline approximation.
    m points per internal subinterval and m+1 points per endpoint subinterval.
    Internal spline knots are excluded from the basis."""
    basis = []

    for i in range(n):
        start = knots[i]
        end = knots[i + 1]

        # pts = m + 1 if (i == 0 or i == n - 1) else m
        if i == 0:
            pts = m if fixed_left_tail else m + 1
        elif i == n - 1:
            pts = m if fixed_right_tail else m + 1
        else:
            pts = m

        local_basis = np.linspace(start, end, pts + 2)[1:-1]
        basis.append(local_basis)

    return basis


def build_P(basis, knots, m):
    return np.array(
        [
            [
                np.maximum(0, t - knot) ** d
                for knot in knots[0:-1]
                for d in range(1, m + 1)
            ]
            for t in basis
        ]
    )


def step_one(knots, basis, m, n, f, fixed_left_value=None, fixed_right_value=None):
    """Solve the linear system for the spline coefficients."""
    temp_basis = basis.copy()
    if fixed_left_value is not None:
        temp_basis.append(np.array([knots[0]]))
    if fixed_right_value is not None:
        temp_basis.append(np.array([knots[-1]]))

    basis_counts = [len(b) for b in temp_basis]
    total_rows = sum(basis_counts)

    A = np.zeros((total_rows, total_rows))
    b = np.zeros(total_rows)

    # Build the matrix A
    A = np.concatenate(
        [
            np.ones((total_rows, 1)),
            build_P(np.concatenate(temp_basis), knots, m),
            np.zeros((total_rows, 1)),
        ],
        axis=1,
    )

    # Add the alternating signs for the last column of A
    sign = -1
    for r in range(A.shape[0]):
        A[r, -1] = sign
        sign *= -1

    # Build the vector b
    b = np.array([f(t) for b in temp_basis for t in b])

    # Fixed right value
    row = A.shape[0] - 1
    if fixed_right_value is not None:
        A[row, -1] = 0
        b[row] = fixed_right_value
        row -= 1

    # Fixed left value
    if fixed_left_value is not None:
        A[row, -1] = 0
        b[row] = fixed_left_value

    # Solve the linear system
    solution = np.linalg.solve(A, b)

    # Extract coefficients and delta
    a0 = solution[0]
    a = solution[1:-1].reshape(n, m)

    # Construct spline S from the coefficients
    coeffs = [np.concatenate([[0], c]) for c in a]
    coeffs[0][0] = a0
    polynomials = [Spline.Polynomial(c, offset=x) for c, x in zip(coeffs, knots[:-1])]
    S = Spline.SUSpline(knots, polynomials)

    return S


def exchange(i, t_star, d_star, f, approx, basis, knots, n, verbose=False):
    """VP basis exchange procedure. Returns new basis if exchange is possible, otherwise returns None."""
    # t* cannot be an internal knot
    for j in range(1, n):
        if abs(t_star - knots[j]) <= TOL:
            if verbose:
                print(f"EXIT 2: t*={t_star} is internal knot {knots[j]}")
            return None

    basis_points = basis[i]

    # t* cannot be a basis point
    if np.any(np.isclose(basis_points, t_star)):
        if verbose:
            print(f"EXIT 2: t*={t_star} is already a basis point in interval {i}")
        return None

    basis_deviations = np.array([approx.deviation(t) for t in basis_points])

    # Absolute deviation at t* must be greater than the absolute deviation at any of the basis points in that interval
    if abs(d_star) <= np.max(np.abs(basis_deviations)) + TOL:
        if verbose:
            print(
                f"EXIT 2: Absolute deviation at t*, {d_star} is <= max absolute deviation at basis points in interval {i}, {np.max(np.abs(basis_deviations))}"
            )
        return None

    # Get the basis points to the left and right of t_star
    left = basis_points[basis_points < t_star]
    right = basis_points[basis_points > t_star]

    left_pt = left[-1] if len(left) else None
    right_pt = right[0] if len(right) else None

    t_star_sign = np.sign(d_star)
    t_tilde = None

    # Check if the deviation at the left or right basis point has the same sign as the deviation at t*
    if left_pt is not None and np.sign(approx.deviation(left_pt)) == t_star_sign:
        t_tilde = left_pt
    elif right_pt is not None and np.sign(approx.deviation(right_pt)) == t_star_sign:
        t_tilde = right_pt

    # If no such t~ exists, then no exchange is possible
    if t_tilde is None:
        return None

    # Replace t_tilde with t_star in basis points
    new_basis = [b.copy() for b in basis]
    new_basis[i] = np.sort(
        np.append(basis_points[~np.isclose(basis_points, t_tilde)], t_star)
    )

    return new_basis


def check_exit_1(
    approx,
    knots,
    n,
    m,
    global_max,
    fixed_left_tail=False,
    fixed_right_tail=False,
    verbose=False,
):
    """Check if the spline approximation satisfies Tarashnin's necessary and sufficient optimality conditions (EXIT 1)."""
    # Get points of alternance and their signs
    pts, signs = approx.alternancesequence()
    pts_and_signs = list(zip(pts, signs))

    def points_alternate(points):
        """Return True if the signs of the points alternate, False otherwise."""
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
            if verbose:
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

            # Intermediate intervals: at least m alternance points required
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
                if verbose:
                    print(
                        f"Condition (ii) satisfied in intervals {i}-{j} with {len(combined)} alternance points"
                    )
                return True, pts, signs, (i, j)

    return False, pts, signs, None


# generalised Remez algorithm
def gra(
    f,
    knots,
    m,
    n,
    exchange_function,
    fixed_left_value=None,
    fixed_right_value=None,
    verbose=False,
):
    """Run the generalised Remez algorithm (GRA) to find the optimal spline approximation of f.
    Optionally, fixed values can be specified for the left and right tails to run the GRAFT algorithm.
    """

    fixed_left_tail = fixed_left_value is not None
    fixed_right_tail = fixed_right_value is not None

    # Form initial basis and compute initial spline approx
    basis = step_zero(knots, m, n, fixed_left_tail, fixed_right_tail)
    S = step_one(knots, basis, m, n, f, fixed_left_value, fixed_right_value)
    approx = Spline.Approximation(f, S, (knots[0], knots[-1]), basis=basis)

    optimal = False
    exit_type = None
    exit_i_star, exit_t_star, exit_d_star = None, None, None

    for _ in range(100):
        i_star, t_star, d_star = approx.maxdeviation()

        optimal, pts, signs, chain = check_exit_1(
            approx,
            knots,
            n,
            m,
            abs(d_star),
            fixed_left_tail,
            fixed_right_tail,
            verbose=verbose,
        )

        # Tarashnin's necessary and sufficient optimality conditions satisfied (EXIT 1)
        if optimal:
            exit_type = 1
            break

        new_basis = exchange_function(
            i_star, t_star, d_star, f, approx, basis, knots, n, verbose=verbose
        )

        # No valid exchange found (EXIT 2)
        if new_basis is None:
            exit_type = 2
            exit_i_star, exit_t_star, exit_d_star = i_star, t_star, d_star
            break

        # Update basis and recompute spline approximation
        basis = new_basis
        S = step_one(knots, basis, m, n, f, fixed_left_value, fixed_right_value)
        approx = Spline.Approximation(f, S, (knots[0], knots[-1]), basis=basis)

    # Final maximum absolute deviation
    i_star, t_star, d_star = approx.maxdeviation()
    approx = Spline.Approximation(f, S, [knots[0], knots[-1]], basis)

    if verbose:
        if exit_type == 1:
            print(f"EXIT 1: Spline is optimal. Chain: {chain}")
        print(f"Final max abs deviation: {abs(d_star):.5f} at t*={t_star:.5f}")
        print(f"Final basis: {approx.basis}")

    return {
        "approximation": approx,
        "exit_type": exit_type,
        "chain": chain,
        "exit_i_star": exit_i_star,
        "exit_t_star": exit_t_star,
        "exit_d_star": exit_d_star,
    }


if __name__ == "__main__":
    function_name = "g"
    f, f_label = test_functions.TEST_FUNCTIONS[function_name]

    a, b = -1, 1
    k = 3  # number of internal fixed knots
    m = 1  # degree of polynomial to fit in each subinterval
    n = k + 1  # number of subintervals

    # Choose initial knots
    # knots = np.array([-1, -5 / 6, -1 / 2, -1 / 6, 0, 1 / 6, 1 / 2, 5 / 6, 1])
    knots = np.linspace(a, b, k + 2)

    result = gra(f, knots, m, n, exchange_function=exchange, verbose=True)

    status = "Optimal" if result["exit_type"] == 1 else "Not optimal"

    plotting.plot_duo(
        result["approximation"],
        points=result["approximation"].basis,
        f_label=f_label,
        approximation_label=rf"$S_{{{m}}}(t)$",
        title=(
            f"Degree-{m} spline approximation of {f_label}. "
            f"{k} internal knots ({status}). "
        ),
        file_name=f"duo_orig_{function_name}_k{k}_m{m}.png",
    )
