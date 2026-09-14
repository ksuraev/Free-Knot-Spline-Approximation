import matplotlib.pyplot as plt
import numpy as np

FUNCTION_COLOR = "slategray"
APPROXIMATION_COLOR = "cornflowerblue"
DEVIATION_COLOR = "darkslategray"
KNOT_COLOR = "crimson"
POINT_COLOR = "black"


def _flatten_points(points):
    if points is None or len(points) == 0:
        return []

    if np.isscalar(points[0]):
        return np.asarray(points)

    return np.concatenate([np.asarray(p) for p in points])


def _plot_function(ax, f, a, b, label, n_samples=1000):
    t = np.linspace(a, b, n_samples)
    ax.plot(t, f(t), color=FUNCTION_COLOR, label=label)


def _plot_knots(ax, knots, label="Knots"):
    for j, knot in enumerate(knots):
        ax.axvline(
            knot,
            color=KNOT_COLOR,
            lw=0.8,
            linestyle="-",
            label=label if j == 0 else None,
        )


def _plot_basis_lines(ax, points, label="Basis points"):
    points = _flatten_points(points)

    if len(points) == 0:
        return

    for j, point in enumerate(points):
        ax.axvline(
            point,
            linestyle=":",
            color=POINT_COLOR,
            label=label if j == 0 else None,
        )


def _plot_approximation(
    ax, approximation, knots, label, a=None, b=None, n_samples=1000
):
    if knots is None:
        # Single polynomial
        t = np.linspace(a, b, n_samples)
        ax.plot(t, approximation(t), color=APPROXIMATION_COLOR, label=label)

    else:
        # Piecewise approximation
        for i in range(len(knots) - 1):
            t_interval = np.linspace(knots[i], knots[i + 1], n_samples)

            ax.plot(
                t_interval,
                approximation(i, t_interval),
                color=APPROXIMATION_COLOR,
                label=label if i == 0 else None,
            )


def _plot_deviation_curve(ax, f, approximation, knots, a=None, b=None, n_samples=1000):
    if knots is None:
        t = np.linspace(a, b, n_samples)
        d = f(t) - approximation(t)

        ax.plot(t, d, color=DEVIATION_COLOR, label=r"$f(t)-P(t)$")

    else:
        for i in range(len(knots) - 1):
            t_interval = np.linspace(knots[i], knots[i + 1], n_samples)

            d = f(t_interval) - approximation(i, t_interval)

            ax.plot(
                t_interval,
                d,
                color=DEVIATION_COLOR,
                label=r"$f(t)-S(t)$" if i == 0 else None,
            )

    ax.axhline(0, color="black", lw=0.5)


def _plot_deviation_markers(
    ax, f, approximation, points, knots=None, label="Alternance points"
):
    points = _flatten_points(points)

    if len(points) == 0:
        return

    for j, point in enumerate(points):
        if knots is None:
            # Single polynomial
            y_approx = approximation(point)
        else:
            # Piecewise approximation
            # doesn't quite work for discontinuous spline - assigns to the right so duplicates end up in same interval
            i = np.searchsorted(knots, point, side="right") - 1
            i = min(max(i, 0), len(knots) - 2)

            y_approx = approximation(i, point)

        y_f = f(point)

        # Point on approximation
        ax.scatter(
            point,
            y_approx,
            color="black",
            s=20,
            zorder=5,
            label=label if j == 0 else None,
        )

        # Vertical deviation line
        ax.plot(
            [point, point],
            [y_approx, y_f],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
        )


def plot_detailed(
    f,
    approximation,
    a,
    b,
    knots=None,
    points=None,
    f_label=r"$f(t)$",
    approximation_label=r"$S(t)$",
    points_label="Basis points",
    title=None,
    file_name=None,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Target function
    t = np.linspace(a, b, 1000)
    ax1.plot(t, f(t), color=FUNCTION_COLOR, label=f_label)

    # Approximation
    _plot_approximation(ax1, approximation, knots, approximation_label, a, b)

    # Deviation
    _plot_deviation_curve(ax2, f, approximation, knots, a, b)

    # Knots
    if knots is not None:
        _plot_knots(ax1, knots)
        _plot_knots(ax2, knots)

    # Basis / alternance points
    if points is not None:
        _plot_deviation_markers(
            ax1,
            f,
            approximation,
            points,
            knots=knots,
            label=points_label,
        )

        _plot_basis_lines(
            ax2,
            points,
            label=points_label,
        )

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
        fig.savefig(file_name)

    plt.show()


def plot_report(
    f,
    approximation,
    a,
    b,
    knots=None,
    points=None,
    f_label=r"$f(t)$",
    approximation_label=r"$S(t)$",
    points_label=None,
    title=None,
    file_name=None,
    figsize=(7, 5),
):
    fig, ax = plt.subplots(figsize=figsize)

    # Target function
    t = np.linspace(a, b, 1000)

    ax.plot(t, f(t), color=FUNCTION_COLOR, label=f_label)

    # Approximation
    _plot_approximation(ax, approximation, knots, approximation_label, a, b)

    # Knots
    if knots is not None:
        _plot_knots(ax, knots)

    # Alternance / basis points and their deviations
    if points is not None:
        _plot_deviation_markers(
            ax, f, approximation, points, knots=knots, label=points_label
        )

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
        fig.savefig(file_name, dpi=300, bbox_inches="tight")

    plt.show()
