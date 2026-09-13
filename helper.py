import numpy as np


def find_local_abs_deviation_maxima(
    deviation_function,
    start,
    end,
    n_samples=10000,
):
    t_samples = np.linspace(start, end, n_samples)

    d_samples = np.array([deviation_function(t) for t in t_samples])

    abs_d_samples = np.abs(d_samples)

    indices = []

    if abs_d_samples[0] >= abs_d_samples[1]:
        indices.append(0)

    for j in range(1, len(t_samples) - 1):
        if (
            abs_d_samples[j] >= abs_d_samples[j - 1]
            and abs_d_samples[j] >= abs_d_samples[j + 1]
        ):
            indices.append(j)

    if abs_d_samples[-1] >= abs_d_samples[-2]:
        indices.append(len(t_samples) - 1)

    return [(t_samples[j], d_samples[j]) for j in indices]


# find the overall maximum, minimum and absolute maximum deviations across all intervals
def find_extrema_overall(
    deviation_function,
    knots,
    n_samples=10000,
):
    n = len(knots) - 1

    t_max = None
    d_max = -np.inf
    i_max = None

    t_min = None
    d_min = np.inf
    i_min = None

    for i in range(n):
        if i == 0:
            t_samples = np.linspace(
                knots[i],
                knots[i + 1],
                n_samples,
            )
        else:
            t_samples = np.linspace(
                knots[i],
                knots[i + 1],
                n_samples,
            )[1:]

        d_samples = np.array([deviation_function(i, t) for t in t_samples])

        idx_max = np.argmax(d_samples)
        idx_min = np.argmin(d_samples)

        if d_samples[idx_max] > d_max:
            i_max = i
            t_max = t_samples[idx_max]
            d_max = d_samples[idx_max]

        if d_samples[idx_min] < d_min:
            i_min = i
            t_min = t_samples[idx_min]
            d_min = d_samples[idx_min]

    if abs(d_max) >= abs(d_min):
        i_star, t_star, d_star = i_max, t_max, d_max
    else:
        i_star, t_star, d_star = i_min, t_min, d_min

    return (
        (i_max, t_max, d_max),
        (i_min, t_min, d_min),
        (i_star, t_star, d_star),
    )


def find_alternance_points(
    deviation_function,
    knots,
    global_max=None,
    tol=1e-5,
    n_samples=10000,
):
    all_extrema = []

    for i in range(len(knots) - 1):

        def local_deviation(t, i=i):
            return deviation_function(i, t)

        extrema = find_local_abs_deviation_maxima(
            local_deviation, knots[i], knots[i + 1], n_samples=n_samples
        )

        all_extrema.extend(extrema)

    if not all_extrema:
        return np.array([]), np.array([]), 0.0

    if global_max is None:
        global_max = max(abs(d) for _, d in all_extrema)

    filtered = [(t, d) for t, d in all_extrema if abs(abs(d) - global_max) <= tol]

    unique = []

    for t, d in filtered:
        if not unique or not np.isclose(t, unique[-1][0]):
            unique.append((t, d))

    pts = np.array([t for t, _ in unique])
    signs = np.array([np.sign(d) for _, d in unique])

    return pts, signs, global_max
