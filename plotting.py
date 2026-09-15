from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

date = datetime.now().strftime("%Y-%m-%d")

plot_dir = Path("plots") / date
plot_dir.mkdir(parents=True, exist_ok=True)


FUNCTION_COLOR = "slategray"
APPROXIMATION_COLOR = "cornflowerblue"
DEVIATION_COLOR = "darkslategray"
KNOT_COLOR = "crimson"
POINT_COLOR = "black"


def _flatten_points(points):
    """Flatten a list of points into a single array."""
    if points is None or len(points) == 0:
        return []

    if np.isscalar(points[0]):
        return np.asarray(points)

    return np.concatenate([np.asarray(p) for p in points])


def _plot_function(ax, approx, label, n_samples=1000):
    """Plot the original function f."""
    a, b = approx.interval
    t = np.linspace(a, b, n_samples)

    ax.plot(t, approx.f(t), color=FUNCTION_COLOR, label=label)


def _plot_approximation(ax, approx, label, n_samples=1000):
    """Plot the approximating polynomial or spline."""

    # Spline: plot each piece separately
    if hasattr(approx.g, "knots"):
        knots = approx.g.knots

        for i in range(approx.g.nintervals()):
            t = np.linspace(knots[i], knots[i + 1], n_samples)

            ax.plot(
                t,
                [approx.g(x) for x in t],
                color=APPROXIMATION_COLOR,
                label=label if i == 0 else None,
            )

    # Polynomial
    else:
        a, b = approx.interval
        t = np.linspace(a, b, n_samples)

        ax.plot(t, approx.g(t), color=APPROXIMATION_COLOR, label=label)


def _plot_deviation_curve(ax, approx, n_samples=1000):
    """Plot the signed deviation f(t) - g(t)."""

    if hasattr(approx.g, "knots"):
        knots = approx.g.knots

        for i in range(approx.g.nintervals()):
            t = np.linspace(knots[i], knots[i + 1], n_samples)

            d = np.array([approx.deviation(x) for x in t])

            ax.plot(
                t,
                d,
                color=DEVIATION_COLOR,
                label=r"$f(t)-S(t)$" if i == 0 else None,
            )

    else:
        a, b = approx.interval
        t = np.linspace(a, b, n_samples)

        d = np.array([approx.deviation(x) for x in t])

        ax.plot(t, d, color=DEVIATION_COLOR, label=r"$f(t)-P(t)$")

    ax.axhline(0, color="black", lw=0.5)


def _plot_knots(ax, approx, label="Knots"):
    """Plot spline knots, if the approximation is a spline."""

    if not hasattr(approx.g, "knots"):
        return

    for j, knot in enumerate(approx.g.knots):
        ax.axvline(
            knot,
            color=KNOT_COLOR,
            lw=0.8,
            linestyle="-",
            label=label if j == 0 else None,
        )


def _plot_basis_lines(ax, points, label="Basis points"):
    points = _flatten_points(points)

    for j, point in enumerate(points):
        ax.axvline(
            point,
            linestyle=":",
            color=POINT_COLOR,
            label=label if j == 0 else None,
        )


def _plot_deviation_markers(ax, approx, points, label="Alternance points"):
    """Plot approximation points and their deviations from f."""

    points = _flatten_points(points)

    for j, point in enumerate(points):
        y_approx = approx.g(point)
        y_f = approx.f(point)

        # Point on approximation.
        ax.scatter(
            point,
            y_approx,
            color=POINT_COLOR,
            s=20,
            zorder=5,
            label=label if j == 0 else None,
        )

        # Vertical deviation.
        ax.plot(
            [point, point],
            [y_approx, y_f],
            color=POINT_COLOR,
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )


def plot_duo(
    approx,
    points=None,
    f_label=r"$f(t)$",
    approximation_label=r"$S(t)$",
    points_label="Basis points",
    title=None,
    file_name=None,
):
    """Plot two subplots: the function and its approximation, and the deviation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Functions
    _plot_function(ax1, approx, f_label)
    _plot_approximation(ax1, approx, approximation_label)

    # Deviation
    _plot_deviation_curve(ax2, approx)

    # Knots
    _plot_knots(ax1, approx)
    _plot_knots(ax2, approx)

    # Basis / alternance points
    if points is not None:
        _plot_deviation_markers(ax1, approx, points, label=points_label)
        _plot_basis_lines(ax2, points, label=points_label)

    ax1.set_xlabel("t")
    ax1.set_title("Approximation")

    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$f(t)-S(t)$")
    ax2.set_title("Deviation")

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if file_name is not None:
        fig.savefig(plot_dir / file_name, dpi=300, bbox_inches="tight")

    plt.show()


def plot_report(
    approx,
    points=None,
    f_label=r"$f(t)$",
    approximation_label=r"$S(t)$",
    points_label=None,
    title=None,
    file_name=None,
    figsize=(7, 5),
):
    """Plot a single figure with the function, approximation, and deviation. Report-style plot."""
    fig, ax = plt.subplots(figsize=figsize)

    _plot_function(ax, approx, f_label)

    _plot_approximation(ax, approx, approximation_label)

    _plot_knots(ax, approx)

    if points is not None:
        _plot_deviation_markers(ax, approx, points, label=points_label)

    ax.set_xlabel(r"$t$")

    if title is not None:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.tick_params(axis="both", which="major", labelsize=14)

    ax.legend(fontsize=15, frameon=False, loc="best")

    fig.tight_layout()

    if file_name is not None:
        fig.savefig(plot_dir / file_name, dpi=300, bbox_inches="tight")

    plt.show()
