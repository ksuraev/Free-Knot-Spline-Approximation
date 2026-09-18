from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("pgf")
import matplotlib.pyplot as plt
import numpy as np

import Spline

pgf_preamble_string = "\n".join([r"\usepackage{amsmath}"])


mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": pgf_preamble_string,
    }
)


date = datetime.now().strftime("%Y-%m-%d")

plot_dir = Path("plots") / date
plot_dir.mkdir(parents=True, exist_ok=True)


FUNCTION_COLOUR = "slategray"
APPROX_COLOUR = "cornflowerblue"
DEVIATION_COLOUR = "slategray"
KNOT_COLOUR = "lightsteelblue"
POINT_COLOUR = "black"


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

    ax.plot(t, approx.f(t), color=FUNCTION_COLOUR, label=label)


def _plot_approximation(ax, approx, label):
    g = approx.g

    if isinstance(g, Spline.SUSpline):
        # Continuous spline: evaluate the full spline
        t = np.linspace(*approx.interval, 1000)
        y = np.array([g(x) for x in t])

        ax.plot(t, y, color=APPROX_COLOUR, label=label, zorder=2)

    elif isinstance(g, Spline.Spline):
        # Discontinuous piecewise spline: plot each piece separately
        for i in range(g.nintervals()):
            t = np.linspace(g.knots[i], g.knots[i + 1], 300)
            y = g.polynomials[i](t)

            ax.plot(
                t, y, color=APPROX_COLOUR, label=label if i == 0 else None, zorder=2
            )
    else:
        # Polynomial
        t = np.linspace(*approx.interval, 1000)
        ax.plot(t, g(t), color=APPROX_COLOUR, label=label, zorder=2)


def _plot_deviation_curve(ax, approx, n_samples=1000):
    """Plot the signed deviation f(t) - g(t)."""

    if hasattr(approx.g, "knots"):
        knots = approx.g.knots

        for i in range(approx.g.nintervals()):
            t = np.linspace(knots[i], knots[i + 1], n_samples)
            d = np.array([approx.deviation(x) for x in t])

            ax.plot(
                t, d, color=DEVIATION_COLOUR, label=r"$f(t)-S(t)$" if i == 0 else None
            )

    else:
        a, b = approx.interval
        t = np.linspace(a, b, n_samples)
        d = np.array([approx.deviation(x) for x in t])
        ax.plot(t, d, color=DEVIATION_COLOUR, label=r"$f(t)-P(t)$")

    ax.axhline(0, color="black", lw=0.5)


def _plot_knots(ax, approx):
    """Plot spline knots, if the approximation is a spline."""
    if not hasattr(approx.g, "knots"):
        return
    for knot in approx.g.knots[1:-1]:
        ax.axvline(knot, linewidth=0.7, zorder=0, color=KNOT_COLOUR)


def _plot_basis_lines(ax, approx, label="Basis points"):
    basis_points = _flatten_points(approx.basis)

    for point in basis_points:
        y_dev = approx.deviation(point)

        # point on the x-axis
        ax.scatter(point, 0, color="black", s=15, zorder=5)

        # dashed connector
        ax.plot(
            [point, point],
            [0, y_dev],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )


def _plot_deviation_markers(ax, approx, points, label="Alternance points"):
    """Plot function points and their deviations from the approximation."""

    j = 0

    # Discontinuous spline
    if isinstance(approx.g, Spline.Spline) and not isinstance(
        approx.g, Spline.SUSpline
    ):
        for i, interval_points in enumerate(points):
            P = approx.g.polynomials[i]

            for point in interval_points:
                y_approx = P(point)
                y_f = approx.f(point)

                # Point on the function
                ax.scatter(
                    point,
                    y_f,
                    color=POINT_COLOUR,
                    s=15,
                    zorder=5,
                    label=label if j == 0 else None,
                )

                # Vertical deviation
                ax.plot(
                    [point, point],
                    [y_approx, y_f],
                    color=POINT_COLOUR,
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                )

                j += 1

    # Continuous spline (GRA) or polynomial (Remez)
    else:
        points = _flatten_points(points)

        for j, point in enumerate(points):
            y_approx = approx.g(point)
            y_f = approx.f(point)

            # Point on the function.
            ax.scatter(
                point,
                y_f,
                color=POINT_COLOUR,
                s=15,
                zorder=5,
                label=label if j == 0 else None,
            )

            # Vertical deviation.
            ax.plot(
                [point, point],
                [y_approx, y_f],
                color=POINT_COLOUR,
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
        _plot_basis_lines(ax2, approx, label=points_label)

    ax1.set_xlabel("t")
    ax1.set_title("Approximation")

    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$f(t)-S(t)$")
    ax2.set_title("Deviation")

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3))

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for spine in ax1.spines.values():
        spine.set_linewidth(0.6)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.6)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if file_name is not None:
        fig.savefig((plot_dir / file_name).with_suffix(".pgf"), bbox_inches="tight")
        fig.savefig(
            (plot_dir / file_name).with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )


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
    _plot_knots(ax, approx)
    _plot_approximation(ax, approx, approximation_label)

    if points is not None:
        _plot_deviation_markers(ax, approx, points, label=points_label)

    ax.set_xlabel(r"$t$", fontsize=15)

    if title is not None:
        ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.legend(
        fontsize=16,
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    fig.tight_layout()

    if file_name is not None:
        fig.savefig((plot_dir / file_name).with_suffix(".pgf"), bbox_inches="tight")
        fig.savefig(
            (plot_dir / file_name).with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )


def plot_objective_psi(
    f,
    a,
    b,
    m,
    n,
    theta_start,
    theta_end,
    psi_function,
    file_name=None,
    step=0.05,
    figsize=(7, 5),
    verbose=False,
):
    """Plot the objective function psi(theta) in same style as the report plots."""

    thetas = np.arange(theta_start, theta_end, step)
    psi_values = np.array(
        [
            psi_function(f, a, b, theta, m, n, verbose=verbose)["d_max"]
            for theta in thetas
        ]
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        thetas, psi_values, color=APPROX_COLOUR, linewidth=1.5, label=r"$\psi(\theta)$"
    )

    min_index = np.argmin(psi_values)
    theta_min = thetas[min_index]
    psi_min = psi_values[min_index]

    ax.scatter(
        theta_min,
        psi_min,
        color=POINT_COLOUR,
        s=20,
        zorder=5,
        label=rf"$\theta^*={theta_min:.3f}$",
    )

    ax.axvline(theta_min, color=KNOT_COLOUR, linewidth=0.7, zorder=0)

    ax.set_xlabel(r"$\theta$", fontsize=15)
    ax.set_ylabel(r"$\psi(\theta)$", fontsize=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.tick_params(axis="both", which="major", labelsize=15)

    ax.legend(
        fontsize=16,
        loc="best",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
    )

    fig.tight_layout()

    if file_name is not None:
        fig.savefig((plot_dir / file_name).with_suffix(".pgf"), bbox_inches="tight")
        fig.savefig(
            (plot_dir / file_name).with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )
