import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


class Polynomial(np.polynomial.Polynomial):
    """A polynomial with an offset, represented as

    .. math::
      \\sum_{i=0}^n a_i(t-θ)^i

    This class derives from the numpy polynomial class.
    """

    def __init__(self,
                 coefficients,
                 domain=None,
                 window=None,
                 symbol="t",
                 offset=0):
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
            1 / factorial(i) * self.deriv(i)(theta)
            for i in range(self.degree() + 1)
        ]
        return Polynomial(coefs,
                          self.domain,
                          self.window,
                          self.symbol,
                          offset=theta)

    def deriv(self, m=1):
        return super().deriv(m=m).addoffset(self.offset)

    def __str__(self):
        string = super().__str__()
        if self.offset > 0:
            string = string.replace(self.symbol,
                                    f"({self.symbol} - {self.offset})")
        if self.offset < 0:
            string = string.replace(self.symbol,
                                    f"({self.symbol} + {-self.offset})")
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
            [v if isinstance(v, Polynomial) else Polynomial(v)
             for v in polynomials]
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

        assert (t >= self.knots[0]) and (
            t <= self.knots[-1]
        ), "Value outside of Spline interval."

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

        assert this.knots[-1] == S.knots[0], (
                "The splines' intervals are not adjacent.")

        return Spline(
            np.concatenate([this.knots, S.knots[1:]]),
            np.concatenate([this.polynomials, S.polynomials]),
        )

    def __str__(self):
        return "\n".join([f"[{self.knots[i]}, {self.knots[i+1]}): \
                         {p}" for i, p in enumerate(self.polynomials)]) + '\n'

    # This should ideally return an Approximation object.
    @classmethod
    def interpolate(f, points):
        """Interpolate the function `f` through the given points."""
        assert False, "TODO"

    def to_SSpline(self):
        """Convert a spline to the format used in Sukhorukova, 2010."""
        polynomials = [p.rebase(theta)
                       for p, theta in zip(self.polynomials, self.knots)]
        return Spline(self.knots, polynomials)

    def to_SUSpline(self):
        """Convert a Spline to a SUSpline"""
        cumulative = [self.polynomials[0]] + list(
            self.polynomials[1:] - self.polynomials[:-1]
        )
        polynomials = [p.rebase(theta)
                       for p, theta in zip(cumulative, self.knots)]
        return SUSpline(self.knots, polynomials)

    def rebase(self, theta=0):
        """Rewrite the spline so all polynomial pieces use the same basis."""
        polynomials = [p.rebase(theta) for p in self.polynomials]
        return Spline(self.knots, polynomials)


class SUSpline(Spline):
    """A spline whose formula is as in Sukhorukova & Ugon, 2017"""

    def __call__(self, t):
        """Evaluate the spline at point `t`."""

        assert (t >= self.knots[0]) and (
            t <= self.knots[-1]
        ), "Value outside of Spline interval."

        return sum(p(t) for p in self.polynomials if t >= p.offset)


# We can get our algorithms to return an object of this type
# with appropriately picked parameters.
class Approximation:
    """An approximation function."""

    def __init__(self, f, g, interval, basis=None):
        self.f = f
        self.g = g
        self.interval = interval
        self.basis = basis

    def deviation(self, t):
        """The absolute deviation achieved at point `t`"""
        return abs(self.f(t) - self.g(t))

    def maxdeviation(self):
        """The maximum absolute deviation achieved across the interval."""
        sample = np.linspace(self.interval[0], self.interval[1], 10000)
        return max([self.deviation(t) for t in sample])

    # This should be improved. Right now this likely has subsequent samples,
    # We should build a mechanism to only report one point if, several
    # subsequent points achieve the max deviation.
    def maxdeviationpoints(self, tol=1e-6):
        """The points where the maximum deviation is achieved"""
        maxdev = self.maxdev()
        sample = np.linspace(self.interval[0], self.interval[1], 10000)
        return [t for t in sample if self.deviation(t) >= maxdev - tol]

    def alternancesequence(self, tol=1e-6):
        """Give the largest alternance sequence in the interval."""
        assert False, "Not Implemented."

    def plot_functions(self, plot_title, plot_name):

        fig, ax = plt.subplots(figsize=(10, 6))

        # original function f(x)
        x = np.linspace(self.interval[0], self.interval[1], 1000)
        ax.plot(x, self.f(x), color="slategray", label="f(x)")

        # approximation polynomial P(x)
        ax.plot(x, self.g(x), color="dodgerblue", label="P(x)")

        # alternance points
        for x in self.basis:
            ax.axvline(x=x, color="lightgray", linestyle="--", alpha=0.5)

        ax.set_title(plot_title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_name)
        plt.show()

    def plot_deviation(self, plot_title, plot_name):

        fig, ax = plt.subplots(figsize=(10, 6))

        # original function f(x)
        x = np.linspace(self.interval[0], self.interval[1], 1000)
        ax.plot(x, self.deviation(x), color="slategray", label="|f(x)-P(x)|")

        # alternance points
        for x in self.basis:
            ax.axvline(x=x, color="lightgray", linestyle="--", alpha=0.5)

        for x in self.alternancesequence():
            ax.axvline(x=x, color="red", linestyle=":", alpha=0.5)

        ax.set_title(plot_title)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(plot_name)
        plt.show()
