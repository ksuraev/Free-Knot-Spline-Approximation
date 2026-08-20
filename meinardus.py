import matplotlib.pyplot as plt
import numpy as np

import remez


def d(a, b):
    _, e_max, ref_points = remez.remez(f, a, b, degree)
    return abs(e_max), ref_points


def f(x):
    return np.sin(x)


a, b = 0, 6
k = 3  # number of free knots (not including a and b)
degree = 3  # degree of polynomial to fit
tolerance = 1e-6
max_iter = 100

# n = 0
knots = np.linspace(a, b, k + 2)
deviations = []
for i in range(len(knots) - 1):
    e_max, _ = d(knots[i], knots[i + 1])
    deviations.append(e_max)

d_min = min(deviations)
d_max = max(deviations)

print(f"Equidistant knots max and min deviations: {d_max:.8f}, {d_min:.8f}")

# n >= 1
for iteration in range(max_iter):
    if d_max - d_min <= tolerance:
        break

    # target deviation for this iteration
    d_n = (d_min * d_max) ** 0.5

    new_knots = [a]
    x_i = a
    j = 0

    # Subroutine: solve d(x_i, x_bar) = d_n while knots can be placed
    for _ in range(k):
        d_i, _ = d(x_i, b)
        if d_i <= d_n:
            break

        x_l = x_i
        x_u = b

        for _ in range(max_iter):
            x_bar = (x_l + x_u) / 2
            e, ref_points = d(x_i, x_bar)

            if abs(e - d_n) < tolerance * d_n:
                break
            elif e < d_n:
                x_l = x_bar
            else:
                x_u = x_bar

        # Previous - setting knot to x_bar
        # new_knots.append(x_bar)
        # x_i = x_bar

        # New - setting knot to last reference point instead of x_bar
        new_knot = ref_points[-1]
        new_knots.append(new_knot)
        x_i = new_knot

        j += 1

    # Collapse any unplaced knots to the right endpoint b
    remaining = k - j
    new_knots.extend([b] * remaining)
    new_knots.append(b)
    knots = np.array(new_knots)

    # c_n = deviation of the last real interval
    c_n, _ = d(knots[j], knots[j + 1])

    d_min = max(d_min, min(c_n, d_n))
    d_max = min(d_max, max(c_n, d_n))

# Print knots
print("final knots:", knots)
print(f"final knots max and min deviations: {d_max:.8f}, {d_min:.8f}")

# Plot original function, polynomial approximation and final knots
piecewise_polynomial = []
for i in range(len(knots) - 1):
    P, _, _ = remez.remez(f, knots[i], knots[i + 1], degree)
    piecewise_polynomial.append((knots[i], knots[i + 1], P))

fig, ax = plt.subplots(figsize=(10, 6))

# original function f(x)
x = np.arange(a, b, 0.01)
ax.plot(x, f(x), color="slategray", label="f(x)")

# approximation polynomial P(x) over each interval
for start, end, P in piecewise_polynomial:
    x = np.arange(start, end, 0.01)
    ax.plot(x, P(x), color="dodgerblue", label="P(x)" if start == knots[0] else None)

# knots as vertical lines
for x in knots:
    ax.axvline(x, color="red", ls=":", label="knots" if x == knots[0] else None)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"Degree-{degree} approximation with {k} free knots")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("plot.png")
plt.show()
