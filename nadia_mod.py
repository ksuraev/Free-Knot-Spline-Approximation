import matplotlib.pyplot as plt
import numpy as np

TOL = 1e-5
ALTERNANCE_TOL = 1e-5


# Form intial basis - m per internal subinterval, m+1 per endpoint subinterval. Internal spline knots are excluded from the basis
def step_zero(knots, m, n):
    basis = []

    for i in range(n):
        start = knots[i] if i == 0 else knots[i]
        end = knots[i + 1] if i == n - 1 else knots[i + 1]

        # count = m + 1 if i == 0 or i == n - 1 else m
        # basis.append(np.linspace(start, end, count))

        s = 0 if i == 0 else 1
        e = -1 if i == n - 1 else -2
        local_basis = np.linspace(start, end, m + 3)[s:e]
        basis.append(local_basis)

    # print([len(x) for x in basis])
    # print(basis)
    return basis


# Construct P_i matrix for given subinterval i
# p^i_𝛼β = {(t_i𝛼-θ_{i-1})^β}, 𝛼=1,...,k_i, β=1,...,m
def build_P_matrix(i, basis, knots, m):
    prev_knot = knots[i]
    return np.array(
        [[(t - prev_knot) ** beta for beta in range(1, m + 1)] for t in basis[i]]
    )


# Construct Q_i matrix row for given subinterval i
# q^i_𝛼β = {(θ_i-θ_{i-1})^β}, β=1,...,m
def build_Q_row(i, knots, m):
    return np.array([(knots[i + 1] - knots[i]) ** beta for beta in range(1, m + 1)])


# step 1 : Construct full matrix (𝝲+2 rows) and solve for polynomial spline S and delta
def step_one(knots, basis, m, n, f):
    P_matrices = [build_P_matrix(i, basis, knots, m) for i in range(n)]
    Q_rows = [build_Q_row(i, knots, m) for i in range(n - 1)]

    basis_counts = [len(b) for b in basis]
    total_rows = sum(basis_counts)

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

    solution = np.linalg.solve(A, b)
    a0 = solution[0]
    a = solution[1:-1].reshape(n, m)

    delta = solution[-1]

    def S(i, t):
        # don't have a better name for 'last_term' yet
        last_term = a0 if i == 0 else S(i - 1, knots[i])
        return sum(a[i, j] * (t - knots[i]) ** (j + 1) for j in range(m)) + last_term

    return S, delta


# Compute the deviation between f and spline S at point t
def deviation(f, S, i, t):
    return f(t) - S(i, t)


# Find local maxima of the absolute deviation in interval i
def find_local_abs_deviation_maxima(f, S, i, knots, n_samples=10000):
    start = knots[i]
    end = knots[i + 1]

    t_samples = np.linspace(start, end, n_samples)
    d_samples = np.array([deviation(f, S, i, t) for t in t_samples])

    abs_d_samples = np.abs(d_samples)

    indices = []

    # Left endpoint: one-sided local maximum
    if abs_d_samples[0] >= abs_d_samples[1]:
        indices.append(0)

    # Interior local maxima
    for j in range(1, len(t_samples) - 1):
        if (
            abs_d_samples[j] >= abs_d_samples[j - 1]
            and abs_d_samples[j] >= abs_d_samples[j + 1]
        ):
            indices.append(j)

    # Right endpoint: one sided local maximum
    if abs_d_samples[-1] >= abs_d_samples[-2]:
        indices.append(len(t_samples) - 1)

    return [(t_samples[j], d_samples[j]) for j in indices]


# find the overall maximum and minimum deviation across all intervals
def find_extrema_overall(f, S, knots, n, n_samples=10000):
    t_max = None
    d_max = -np.inf
    i_max = None

    t_min = None
    d_min = np.inf
    i_min = None

    for i in range(n):
        if i == 0:
            t_samples = np.linspace(knots[i], knots[i + 1], n_samples)
        else:
            t_samples = np.linspace(knots[i], knots[i + 1], n_samples)[1:]

        d_samples = np.array([deviation(f, S, i, t) for t in t_samples])

        idx_max = np.argmax(d_samples)
        idx_min = np.argmin(d_samples)

        # max signed deviation
        if d_samples[idx_max] > d_max:
            i_max, t_max, d_max = i, t_samples[idx_max], d_samples[idx_max]

        # min signed deviation
        if d_samples[idx_min] < d_min:
            i_min, t_min, d_min = i, t_samples[idx_min], d_samples[idx_min]

    # max absolute deviation
    if abs(d_max) >= abs(d_min):
        i_star, t_star, d_star = i_max, t_max, d_max
    else:
        i_star, t_star, d_star = i_min, t_min, d_min

    return (
        (i_max, t_max, d_max),
        (i_min, t_min, d_min),
        (i_star, t_star, d_star),
    )


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

        left_sign = np.sign(deviation(f, S, left_i, left_pt))

        right_sign = np.sign(deviation(f, S, right_i, right_pt))

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

        if left_pt is not None and np.sign(deviation(f, S, i, left_pt)) == t_star_sign:
            t_tilde = left_pt

        elif (
            right_pt is not None
            and np.sign(deviation(f, S, i, right_pt)) == t_star_sign
        ):
            t_tilde = right_pt

        if t_tilde is None:
            print("no valid basis point to replace")
            return None

    # replace basis point t_tilde with t_star in interval i
    basis_points = basis[i]

    basis_deviations = np.array([deviation(f, S, i, t) for t in basis_points])

    max_basis_deviation = np.max(np.abs(basis_deviations))

    if abs(d_star) <= max_basis_deviation + TOL:
        print(
            f"Absolute deviation at t*, {d_star} is <= max absolute deviation at basis points in interval {i}, {np.max(np.abs(basis_deviations))}"
        )
        return None

    new_basis = [b.copy() for b in basis]

    new_basis[i] = np.sort(
        np.append(
            basis_points[~np.isclose(basis_points, t_tilde)],
            t_star,
        )
    )
    # print(f"new basis: {new_basis}\n")

    return new_basis


# find sampled alternance points across all intervals, filtered by global maximum deviation
def find_alternance_points(f, S, knots, n, global_max):
    all_alt_points = []

    for i in range(n):
        all_alt_points.extend(find_local_abs_deviation_maxima(f, S, i, knots))

    if not all_alt_points:
        return np.array([]), np.array([]), 0.0

    # Filter points that are within ALTERNANCE_TOL of the global maximum deviation
    filtered = [
        (t, d) for t, d in all_alt_points if abs(abs(d) - global_max) <= ALTERNANCE_TOL
    ]

    # Handle duplicates
    unique = []
    for t, d in filtered:
        if not unique or not np.isclose(t, unique[-1][0]):
            unique.append((t, d))

    pts = np.array([t for t, _ in unique])
    signs = np.array([np.sign(d) for _, d in unique])

    return pts, signs


# Tarashnin's necessary and sufficient optimality conditions
def check_exit_1(f, S, knots, n, m, global_max):
    pts, signs = find_alternance_points(f, S, knots, n, global_max)

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

        if len(points) >= m + 2 and points_alternate(points):
            print(
                f"Condition (i) satisfied in interval {i} with {len(points)} alternance points."
            )
            return True, pts, signs, (i, i)

    # Condition (ii)
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
            # First and last interval: at least m+1 alternance points
            if len(chain_intervals[0]) < m + 1:
                continue

            if len(chain_intervals[-1]) < m + 1:
                continue

            # Intermediate intervals: at least m alternance points
            if not all(
                len(chain_intervals[k]) >= m for k in range(1, len(chain_intervals) - 1)
            ):
                continue

            combined = [point for interval in chain_intervals for point in interval]

            required = m * (j - i + 1) + 2

            if len(combined) >= required and points_alternate(combined):
                print(
                    f"Condition (ii) satisfied in intervals {i}-{j} with {len(combined)} alternance points"
                )
                return (True, pts, signs, (i, j))

    return False, pts, signs, None


def plot_approximation(f, f_label, a, b, n, S, knots, basis, m, status):
    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.linspace(a, b, 1000)
    ax.plot(t, f(t), color="slategrey", label=f_label)

    for i in range(n):
        t_interval = np.linspace(knots[i], knots[i + 1])
        ax.plot(
            t_interval,
            S(i, t_interval),
            color="dodgerblue",
            label="Spline approximation" if i == 0 else None,
        )

    basis_label = ", ".join(f"{point:g}" for interval in basis for point in interval)
    for i, interval_basis in enumerate(basis):
        for j, point in enumerate(interval_basis):
            ax.axvline(
                point,
                linestyle=":",
                color="black",
                label=f"Basis points: {basis_label}" if i == 0 and j == 0 else None,
            )

    knot_label = ", ".join(f"{knot:g}" for knot in knots)

    for j, knot in enumerate(knots):
        ax.axvline(
            knot,
            color="red",
            lw=0.5,
            linestyle="-",
            label=f"Knots: {knot_label}" if j == 0 else None,
        )

    ax.set_xlabel("t")
    ax.set_title(
        f"Degree-{m} spline approximation of {f_label} with {k} internal knots ({status})"
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    fig.tight_layout()
    fig.savefig(f"{function_name}_k{k}_m{m}_approximation.png")
    plt.show()


def plot_deviation(f, f_label, n, S, knots, basis, m, k, status):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(n):
        t_interval = np.linspace(knots[i], knots[i + 1], 1000)

        d = np.array([deviation(f, S, i, t) for t in t_interval])

        ax.plot(
            t_interval, d, color="slategray", label=r"$f(t)-S(t)$" if i == 0 else None
        )

    ax.axhline(0)

    basis_label = ", ".join(f"{point:g}" for interval in basis for point in interval)
    for i, interval_basis in enumerate(basis):
        for j, point in enumerate(interval_basis):
            ax.axvline(
                point,
                linestyle=":",
                color="black",
                label=f"Basis points: {basis_label}" if i == 0 and j == 0 else None,
            )
    knot_label = ", ".join(f"{knot:g}" for knot in knots)
    for j, knot in enumerate(knots):
        ax.axvline(
            knot,
            color="red",
            lw=0.5,
            linestyle="-",
            label=f"Knots: {knot_label}" if j == 0 else None,
        )

    ax.set_xlabel("t")
    ax.set_ylabel(r"$f(t)-S(t)$")
    ax.set_title(
        f"Deviation for degree-{m} spline approximation of {f_label} with {k} internal knots ({status})"
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    fig.tight_layout()
    fig.savefig(f"{function_name}_k{k}_m{m}_deviation.png")
    plt.show()


def f_sin(t):
    return np.sin(t)


def f_log(t):
    return np.log(t + 1)


def f_log_sin(t):
    return np.log(np.sin(t) + 3)


TEST_FUNCTIONS = {
    "sin": (f_sin, r"$\sin(t)$"),
    "log": (f_log, r"$\log(t+1)$"),
    "log_sin": (f_log_sin, r"$\log(\sin(t)+3)$"),
}


if __name__ == "__main__":
    function_name = "sin"
    f, f_label = TEST_FUNCTIONS[function_name]

    a, b = 2, 6
    k = 1  # number of internal fixed knots
    m = 2  # degree of polynomial to fit in each subinterval
    n = k + 1  # number of subintervals

    # Choose intial knots
    knots = [2, 5.9, 6]  # differentiable at 3.43177734

    # Choose initial basis
    basis = step_zero(knots, m, n)

    # Construct polynomial spline S
    S, delta = step_one(knots, basis, m, n, f)

    max_iterations = 100

    for _ in range(max_iterations):

        # Find extrema of current spline
        (max_i, t_max, d_max), (min_i, t_min, d_min), (i_star, t_star, d_star) = (
            find_extrema_overall(f, S, knots, n)
        )

        optimal, pts, signs, chain = check_exit_1(f, S, knots, n, m, abs(d_star))
        # if chain is not None:
        #     print(f"chain: {chain}")

        if optimal:
            print("EXIT 1 (spline is optimal).")
            print("basis:            ", basis)
            print("alternance points:", pts)
            break

        new_basis = exchange(i_star, t_star, d_star, f, S, basis, knots, n)

        if new_basis is None:
            print("EXIT 2 (no valid exchange).")
            print("current basis:    ", basis)
            print("alternance points:", pts)
            break

        # Construct new spline
        basis = new_basis
        S, delta = step_one(knots, basis, m, n, f)

    # final results
    (max_i, t_max, d_max), (min_i, t_min, d_min), (i_star, t_star, d_star) = (
        find_extrema_overall(f, S, knots, n)
    )

    print(f"Absolute deviation at basis points: {abs(delta):.5f}")
    print(f"Maximum deviation: {d_max:.5f} at t = {t_max:.5f}")
    print(f"Minimum deviation: {d_min:.5f} at t = {t_min:.5f}")
    print(f"Maximum absolute deviation: " f"{d_star:.5f} at t = {t_star:.5f}")

    status = "Optimal" if optimal else "Not optimal"
    plot_approximation(f, f_label, a, b, n, S, knots, basis, m, status)
    plot_deviation(f, f_label, n, S, knots, basis, m, k, status)
