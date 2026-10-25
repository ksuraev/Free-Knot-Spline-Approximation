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
        """Return the m-th derivative of the polynomial, keeping the offset."""
        return super().deriv(m=m).addoffset(self.offset)

    def __str__(self):
        """Return a string representation of the polynomial, with the offset."""
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

        assert len(knots) == len(polynomials) + 1, (
            "The number of polynomials must equal the number of subintervals."
        )

        self.knots = knots
        self.polynomials = np.array(
            [v if isinstance(v, Polynomial) else Polynomial(v) for v in polynomials]
        )

        # Work out the degree of the spline from the degrees of the polynomial pieces
        degrees = [v.degree() for v in self.polynomials]
        self.degree = np.max(degrees)

        assert all(degrees == self.degree), (
            "Spline polynomials must all have the same degree."
        )

    def nintervals(self):
        """Return the number of intervals in the spline."""
        return len(self.polynomials)

    def __call__(self, t):
        """Evaluate the spline at point t"""

        assert np.all(t >= self.knots[0]) and np.all(t <= self.knots[-1]), (
            "Value outside of Spline interval."
        )

        if isinstance(t, np.ndarray):
            results = np.zeros(t.shape)
            for i, p in enumerate(self.polynomials):
                mask = (t >= self.knots[i]) & (t < self.knots[i + 1])
                results[mask] += p(t[mask])
            return results

        # Find the subinterval containing `t`
        k = np.where(self.knots[:-1] <= t)[0][-1]

        return self.polynomials[k](t)

    def is_continuous(self):
        """Returns true if the spline is continuous at the knots."""
        return all(
            self.polynomials[k](theta) == self.polynomials[k + 1](theta)
            for k, theta in enumerate(self.knots[1:-1])
        )

    def is_differentiable_at(self, j, tol=1e-6):
        """Check if the spline is differentiable at the specified internal knot."""
        assert 1 <= j <= len(self.knots) - 2, "knot index j must be an internal knot"

        knot = self.knots[j]
        p_right = self.polynomials[j - 1]
        p_left = self.polynomials[j]

        # check continuity
        if not np.isclose(p_left(knot), p_right(knot), atol=tol, rtol=0):
            return False

        return np.isclose(p_left.deriv()(knot), p_right.deriv()(knot), atol=tol, rtol=0)

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
                    f"[{self.knots[i]}, {self.knots[i + 1]}): \
                         {p}"
                    for i, p in enumerate(self.polynomials)
                ]
            )
            + "\n"
        )

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

    @classmethod
    def from_coefficients(cls, knots, m, coefs):
        """Construct a SUSpline from a flat coefficient vector."""

        a0 = coefs[0]
        a = coefs[1:].reshape(len(knots) - 1, m)

        polynomial_coefs = [np.concatenate([[0], c]) for c in a]

        polynomial_coefs[0][0] = a0

        polynomials = [
            Polynomial(c, offset=knot) for c, knot in zip(polynomial_coefs, knots[:-1])
        ]

        return cls(knots, polynomials)

    def __call__(self, t):
        """Evaluate the spline at point `t`."""

        assert np.all(t >= self.knots[0]) and np.all(t <= self.knots[-1]), (
            "Value outside of Spline interval."
        )

        if isinstance(t, np.ndarray):
            results = np.zeros(t.shape)
            for p in self.polynomials:
                mask = t >= p.offset
                results[mask] += p(t[mask])
            return results

        return sum(p(t) for p in self.polynomials if t >= p.offset)

    def is_differentiable_at(self, j, tol=1e-6):
        assert 1 <= j <= len(self.knots) - 2, "knot index j must be an internal knot"

        p = self.polynomials[j]

        a_j0 = p.coef[0]
        a_j1 = p.coef[1] if len(p.coef) > 1 else 0.0

        continuous = np.isclose(a_j0, 0.0, atol=tol, rtol=0)
        derivative_continuous = np.isclose(a_j1, 0.0, atol=tol, rtol=0)

        return bool(continuous and derivative_continuous)


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

            if i < len(knots) - 2:
                t = t[:-1]

            d = self.deviation(t)
            abs_d = np.abs(d)

            indices = []

            if abs_d[0] >= abs_d[1]:
                indices.append(0)

            indices.extend(
                j
                for j in range(1, len(t) - 1)
                if abs_d[j] >= abs_d[j - 1] and abs_d[j] >= abs_d[j + 1]
            )

            if abs_d[-1] >= abs_d[-2]:
                indices.append(len(t) - 1)

            extrema.extend((i, t[j], d[j]) for j in indices)

        return extrema

    def maxdeviation(self, n_samples=1000):
        """Return the maximum deviation"""
        extrema = self._extrema(n_samples)
        d_max = max(extrema, key=lambda x: abs(x[2]))
        return d_max

    def maxdeviationpoints(self, tol=1e-4, n_samples=1000):
        """Return all points attaining the maximum absolute deviation."""
        extrema = self._extrema(n_samples)

        if not extrema:
            return []

        global_max = max(abs(d) for _, _, d in extrema)

        return [
            (t, d)
            for _, t, d in extrema
            if np.isclose(abs(d), global_max, atol=tol, rtol=1e-2)
        ]

    def alternancesequence(self, tol=1e-4, n_samples=1000):
        """Return largest sequence of alternance points, with their signs."""
        points = self.maxdeviationpoints(tol=tol, n_samples=n_samples)

        unique = []

        for t, d in points:
            if not unique or not np.isclose(t, unique[-1][0], rtol=1e-2, atol=tol):
                unique.append((t, d))

        pts = np.array([t for t, _ in unique])
        signs = np.array([np.sign(d) for _, d in unique])

        return pts, signs

    def alternance_per_interval(self, alt_pts, signs):
        """Return the number of alternance points in each closed interval."""
        alt_pts = np.asarray(alt_pts)
        signs = np.asarray(signs)

        counts = []
        knots = self.g.knots

        for i in range(len(knots) - 1):
            left = knots[i]
            right = knots[i + 1]

            mask = (alt_pts >= left) & (alt_pts <= right)
            interval_signs = signs[mask]

            if len(interval_signs) == 0:
                count = 0
            else:
                count = np.sum(interval_signs[1:] != interval_signs[:-1]) + 1

            counts.append(int(count))

        return counts
