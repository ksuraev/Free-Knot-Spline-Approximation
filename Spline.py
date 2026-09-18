import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial


class Polynomial(np.polynomial.Polynomial):
    """A polynomial with an offset, represented as

    .. math::
      \\sum_{i=0}^n a_i(t-θ)^i

    This class derives from the numpy polynomial class.
    """

    def __init__(self, coefficients, domain=None, window=None, symbol="t", offset=0):
        super().__init__(coefficients, domain, window, symbol)
        self.offset = offset

    def addoffset(self, theta):
        """Add an offset to the polynomial, keeping the coefficients."""
        return Polynomial(
            self.coef,
            domain=self.domain,
            window=self.window,
            symbol=self.symbol,
            offset=theta,
        )

    def rebase(self, theta=0):
        """Return the same polynomial, but with a different basis of `t-θ`."""
        coefs = [
            1 / factorial(i) * self.deriv(i)(theta) for i in range(self.degree() + 1)
        ]
        return Polynomial(coefs, self.domain, self.window, self.symbol, offset=theta)

    def deriv(self, m=1):
        return super().deriv(m=m).addoffset(self.offset)

    def __str__(self):
        string = super().__str__()
        if self.offset > 0:
            string = string.replace(self.symbol, f"({self.symbol} - {self.offset})")
        if self.offset < 0:
            string = string.replace(self.symbol, f"({self.symbol} + {-self.offset})")
        return string

    @classmethod
    def interpolate(f, points):
        """Interpolate the function f through the points"""
        assert False, "TODO"

    def __call__(self, t):
        return super().__call__(t - self.offset)


class Spline:
    """A Spline function"""

    def __init__(self, knots, polynomials):
        """A spline is made of knots, so that between any pair of subsequent
        knots it is equal to a polynomial."""

        assert (
            len(knots) == len(polynomials) + 1
        ), "The number of polynomials must equal the number of subintervals."

        self.knots = knots
        self.polynomials = np.array(
            [v if isinstance(v, Polynomial) else Polynomial(v) for v in polynomials]
        )

        # We work out the degree of the spline from the degrees
        # of the polynomial pieces:
        degrees = [v.degree() for v in self.polynomials]
        self.degree = np.max(degrees)

        assert all(
            degrees == self.degree
        ), "Spline polynomials must all have the same degree."

    def nintervals(self):
        return len(self.polynomials)

    def __call__(self, t):
        """Evaluate the spline at point t"""

        assert np.all(t >= self.knots[0]) and np.all(
            t <= self.knots[-1]
        ), "Value outside of Spline interval."

        if isinstance(t, np.ndarray):
            results = np.zeros(t.shape)
            for i, p in enumerate(self.polynomials):
                mask = (t >= self.knots[i]) & (t < self.knots[i + 1])
                results[mask] += p(t[mask])
            return results

        # Find the subinterval containing `t`.
        k = np.where(self.knots[:-1] <= t)[0][-1]
        return self.polynomials[k](t)

    def is_continuous(self):
        """returns true if the spline is continuous at the knots."""
        return all(
            self.polynomials[k](theta) == self.polynomials[k + 1](theta)
            for k, theta in enumerate(self.knots[1:-1])
        )

    def concatenate(this, S):
        """Concatenate two splines"""

        assert this.knots[-1] == S.knots[0], "The splines' intervals are not adjacent."

        return Spline(
            np.concatenate([this.knots, S.knots[1:]]),
            np.concatenate([this.polynomials, S.polynomials]),
        )

    def __str__(self):
        return (
            "\n".join(
                [
                    f"[{self.knots[i]}, {self.knots[i+1]}): \
                         {p}"
                    for i, p in enumerate(self.polynomials)
                ]
            )
            + "\n"
        )

    # This should ideally return an Approximation object.
    @classmethod
    def interpolate(f, points):
        """Interpolate the function `f` through the given points."""
        assert False, "TODO"

    def to_SSpline(self):
        """Convert a spline to the format used in Sukhorukova, 2010."""
        polynomials = [
            p.rebase(theta) for p, theta in zip(self.polynomials, self.knots)
        ]
        return Spline(self.knots, polynomials)

    def to_SUSpline(self):
        """Convert a Spline to a SUSpline"""
        cumulative = [self.polynomials[0]] + list(
            self.polynomials[1:] - self.polynomials[:-1]
        )
        polynomials = [p.rebase(theta) for p, theta in zip(cumulative, self.knots)]
        return SUSpline(self.knots, polynomials)

    def rebase(self, theta=0):
        """Rewrite the spline so all polynomial pieces use the same basis."""
        polynomials = [p.rebase(theta) for p in self.polynomials]
        return Spline(self.knots, polynomials)


class SUSpline(Spline):
    """A spline whose formula is as in Sukhorukova & Ugon, 2017"""

    def __call__(self, t):
        """Evaluate the spline at point `t`."""

        assert np.all(t >= self.knots[0]) and np.all(
            t <= self.knots[-1]
        ), "Value outside of Spline interval."

        if isinstance(t, np.ndarray):
            results = np.zeros(t.shape)
            for p in self.polynomials:
                mask = t >= p.offset
                results[mask] += p(t[mask])
            return results

        return sum(p(t) for p in self.polynomials if t >= p.offset)


# We can get our algorithms to return an object of this type
# with appropriately picked parameters.
class Approximation:
    """An approximation of f by g."""

    def __init__(self, f, g, interval, basis=None):
        self.f = f
        self.g = g
        self.interval = interval
        self.basis = basis

    def deviation(self, t):
        """Return the signed deviation f(t) - g(t)."""
        return self.f(t) - self.g(t)

    def _extrema(self, n_samples=1000):
        """Return local maxima of absolute deviation as (i, t, d)."""
        knots = self.g.knots if hasattr(self.g, "knots") else self.interval
        extrema = []

        for i in range(len(knots) - 1):
            t = np.linspace(knots[i], knots[i + 1], n_samples)

            if i > 0:
                t = t[1:]

            d = self.deviation(t)
            abs_d = np.abs(d)

            indices = np.where((abs_d[1:-1] >= abs_d[0:-2]) &
                               (abs_d[1:-1] >= abs_d[2:]))[0]

            if abs_d[0] >= abs_d[1]:
                indices = np.concat([[0], indices])

            if abs_d[-1] >= abs_d[-2]:
                indices = np.concat([indices, [len(t) - 1]])

            extrema.extend(zip(i*np.ones(len(t), dtype=int), t[indices], d[indices]))

        return extrema

    def maxdeviation(self, n_samples=10000):
        a = self.interval[0]
        b = self.interval[1]
        sample = np.concat([np.linspace(a, b, n_samples)] + self.basis)
        d_abs = np.abs(self.deviation(sample))
        d_max = max(d_abs)
        j_max = np.where(d_abs == d_max)[0][0]
        t_max = sample[j_max]
        # while b-a > 1e-10:
        #     sample = np.linspace(a, b, n_samples)
        #     d_abs = np.abs(self.deviation(sample))
        #     d_max = max(d_abs)
        #     j_max = np.where(d_abs == d_max)[0][0]
        #     t_max = sample[j_max]
        #     if j_max > 1:
        #         a = sample[j_max-1]
        #     if j_max < n_samples-1:
        #         b = sample[j_max+1]
        i_max = np.where(self.g.knots[:-1] <= t_max)[0][-1]
        return (i_max, t_max, self.deviation(t_max))

    def maxdeviationpoints(self, tol=1e-5, n_samples=10000):
        """Return all points attaining the maximum absolute deviation."""
        extrema = self._extrema(n_samples)

        if not extrema:
            return []

        global_max = max(abs(d) for _, _, d in extrema)

        return [(t, d) for _, t, d in extrema if abs(abs(d) - global_max) <= tol]

    def alternancesequence(self, tol=1e-5, n_samples=10000):
        """Return largest sequence of alternance points, with their signs."""
        points = self.maxdeviationpoints(tol=tol, n_samples=n_samples)

        unique = []

        for t, d in points:
            if not unique or not np.isclose(t, unique[-1][0]):
                unique.append((t, d))

        pts = np.array([t for t, _ in unique])
        signs = np.array([np.sign(d) for _, d in unique])

        return pts, signs

    def plot_functions(self, plot_title, plot_name):
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.linspace(self.interval[0], self.interval[1], 1000)

        # Original function
        ax.plot(x, self.f(x), color="slategray", label="f(x)")

        # Approximation
        y = [self.g(t) for t in x]
        ax.plot(x, y, color="dodgerblue", label="Approximation")

        ax.set_title(plot_title)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_name)

        plt.show()

    def plot_deviation(self, plot_title, plot_name):
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.linspace(self.interval[0], self.interval[1], 1000)

        # Signed deviation
        d = [self.deviation(t) for t in x]
        ax.plot(x, d, color="slategray", label="f(x) - g(x)")

        # Basis points
        if self.basis is not None:
            basis = np.concatenate(
                [np.atleast_1d(points) for points in np.atleast_1d(self.basis)]
            )

            for point in basis:
                ax.axvline(x=point, color="lightgray", linestyle="--", alpha=0.5)

        # Alternance points
        alt_points, _ = self.alternancesequence()

        for point in alt_points:
            ax.axvline(x=point, color="red", linestyle=":", alpha=0.5)

        ax.set_title(plot_title)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(plot_name)

        plt.show()
