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
    """Plot the approximation g."""
    g = approx.g

    if isinstance(g, Spline.SUSpline):
        # Continuous spline: evaluate the full spline
        t = np.linspace(*approx.interval, 1000)
        y = np.array([g(x) for x in t])

        ax.plot(
            t,
            y,
            color=APPROX_COLOUR,
            label=label,
            zorder=2,
            linewidth=2,
        )

    elif isinstance(g, Spline.Spline):
        # Discontinuous piecewise spline: plot each piece separately
        for i in range(g.nintervals()):
            t = np.linspace(g.knots[i], g.knots[i + 1], 300)
            y = g.polynomials[i](t)

            ax.plot(
                t,
                y,
                color=APPROX_COLOUR,
                label=label if i == 0 else None,
                zorder=2,
                linewidth=2,
            )
    else:
        # Polynomial
        t = np.linspace(*approx.interval, 1000)
        ax.plot(
            t,
            g(t),
            color=APPROX_COLOUR,
            label=label,
            zorder=2,
            linewidth=2,
        )


def _plot_deviation_curve(ax, approx, n_samples=1000, deviation_label=r"$f(t)-S(t)$"):
    """Plot the signed deviation f(t) - g(t)."""

    if hasattr(approx.g, "knots"):
        knots = approx.g.knots

        for i in range(approx.g.nintervals()):
            t = np.linspace(knots[i], knots[i + 1], n_samples)
            d = np.array([approx.deviation(x) for x in t])

            ax.plot(
                t, d, color=DEVIATION_COLOUR, label=deviation_label if i == 0 else None
            )

    else:
        a, b = approx.interval
        t = np.linspace(a, b, n_samples)
        d = np.array([approx.deviation(x) for x in t])
        ax.plot(t, d, color=DEVIATION_COLOUR, label=deviation_label)

    ax.axhline(0, color="black", lw=0.5)


def _plot_knots(ax, approx):
    """Plot spline knots, if the approximation is a spline."""
    if not hasattr(approx.g, "knots"):
        return
    for knot in approx.g.knots[1:-1]:
        ax.axvline(knot, linewidth=1.5, zorder=0, color=KNOT_COLOUR)


def _plot_basis_lines(ax, approx):
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


def _plot_deviation_markers(ax, approx, points):
    """Plot function points and their deviations from the approximation."""

    j = 0

    # Discontinuous spline
    if isinstance(approx.g, Spline.Spline) and not isinstance(
        approx.g, Spline.SUSpline
    ):
        points = np.asarray(points, dtype=object)
        if points.ndim == 1 and all(np.isscalar(point) for point in points):
            interval_points = [[] for _ in range(approx.g.nintervals())]
            for point in points:
                interval = np.searchsorted(approx.g.knots, point, side="right") - 1
                interval = min(interval, approx.g.nintervals() - 1)
                interval_points[interval].append(point)
        else:
            interval_points = points

        for i, interval_points in enumerate(interval_points):
            P = approx.g.polynomials[i]

            for point in interval_points:
                y_approx = P(point)
                y_f = approx.f(point)

                # Point on the function
                ax.scatter(point, y_f, color=POINT_COLOUR, s=15, zorder=5)

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
            ax.scatter(point, y_f, color=POINT_COLOUR, s=15, zorder=5)

            # Vertical deviation.
            ax.plot(
                [point, point],
                [y_approx, y_f],
                color=POINT_COLOUR,
                linestyle="--",
                linewidth=1,
                alpha=0.8,
            )


def _style_axes(*axes, labelsize=None):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        if labelsize is not None:
            ax.tick_params(axis="both", which="major", labelsize=labelsize)


def _save_figure(fig, file_name):
    if file_name is not None:
        fig.savefig(
            (plot_dir / file_name).with_suffix(".pgf"),
            bbox_inches="tight",
        )
        fig.savefig(
            (plot_dir / file_name).with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            format="pdf",
        )

    plt.close(fig)


def plot_duo(
    approx,
    points=None,
    f_label=r"$f(t)$",
    deviation_label=r"$f(t)-S(t)$",
    approximation_label=r"$S(t)$",
    approximation_title=None,
    deviation_title=None,
    title=None,
    file_name=None,
):
    """Plot two subplots: the function and its approximation, and the deviation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Functions
    _plot_function(ax1, approx, f_label)
    _plot_approximation(ax1, approx, approximation_label)

    # Deviation
    _plot_deviation_curve(ax2, approx, deviation_label=deviation_label)

    # Knots
    _plot_knots(ax1, approx)
    _plot_knots(ax2, approx)

    # Basis / alternance points
    if points is not None:
        _plot_deviation_markers(ax1, approx, points)
        _plot_basis_lines(ax2, approx)

    ax1.set_xlabel(r"$t$", fontsize=15)

    ax2.set_xlabel(r"$t$", fontsize=15)

    ax1.set_title(approximation_title, fontsize=15)
    ax2.set_title(deviation_title, fontsize=15)

    for ax in (ax1, ax2):
        ax.legend(
            fontsize=20,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=2,
            frameon=False,
        )

    _style_axes(ax1, ax2, labelsize=15)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    _save_figure(fig, file_name)
    plt.close(fig)


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
        _plot_deviation_markers(ax, approx, points)

    ax.set_xlabel(r"$t$", fontsize=15)

    _style_axes(ax, labelsize=15)

    ax.legend(
        fontsize=20,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )

    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    _save_figure(fig, file_name)


def _psi_at(theta, thetas, psi_values):
    return np.interp(theta, thetas, psi_values)


def _psi_at_3d(theta, theta_1_values, theta_2_values, psi_values):
    theta = np.asarray(theta)

    if theta.ndim == 1:
        theta = theta[None, :]

    values = []

    for theta_1, theta_2 in theta:
        i = np.argmin(np.abs(theta_1_values - theta_1))
        j = np.argmin(np.abs(theta_2_values - theta_2))

        value = psi_values[i, j]

        if not np.isfinite(value):
            print(f"No finite psi value near " f"({theta_1:.3f}, {theta_2:.3f})")

        values.append(value)

    return np.asarray(values)


def plot_objective_psi(
    thetas,
    psi_values,
    theta_found=None,
    psi_found=None,
    theta_path=None,
    nurnbergers_orig_point=None,
    nurnbergers_mod_point=None,
    file_name=None,
    figsize=(7, 5),
):
    """Plot precomputed psi(theta) values and optionally the algorithm result."""
    thetas = np.asarray(thetas)
    psi_values = np.asarray(psi_values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        thetas,
        psi_values,
        color=APPROX_COLOUR,
        linewidth=1.5,
        label=r"$\psi(\theta_1)$",
    )

    # Find the minimum psi value and its corresponding theta from samples
    min_index = np.argmin(psi_values)
    theta_min = thetas[min_index]
    psi_min = psi_values[min_index]

    # Plot the minimum sampled point as a vertical dashed line and a marker
    ax.plot(
        theta_min,
        0,
        marker="|",
        color="black",
        markersize=10,
        markeredgewidth=1.0,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    y_range = psi_values.max() - psi_values.min()
    ax.set_ylim(psi_values.min() - 0.08 * y_range, psi_values.max() + 0.03 * y_range)
    y_bottom = ax.get_ylim()[0]

    theta_min_label = (
        rf"$\theta_{{\min}}={theta_min:.3f}$"
        if nurnbergers_mod_point is not None or nurnbergers_orig_point is not None
        else rf"$\theta_{1}^*={theta_min:.3f}$" rf"$,\ \psi^*={psi_min:.3f}$"
    )

    ax.vlines(
        theta_min,
        ymin=y_bottom,
        ymax=psi_min,
        color=POINT_COLOUR,
        linestyle="--",
        linewidth=1.3,
        zorder=2,
        label=theta_min_label,
    )

    if theta_path is not None and theta_found is None:
        theta_path = np.asarray(theta_path)
        theta_found = theta_path[-1]
        psi_found = _psi_at(theta_found, thetas, psi_values)

    # Theta returned by the algorithm
    if theta_found is not None:
        if psi_found is None:
            raise ValueError("psi_found must be provided if theta_found is provided.")

        # outline
        ax.scatter(
            theta_found,
            psi_found,
            marker="x",
            color="white",
            s=80,
            linewidths=3,
            zorder=7,
        )
        # fill
        ax.scatter(
            theta_found,
            psi_found,
            marker="x",
            color="crimson",
            s=60,
            linewidths=2,
            zorder=8,
            label=(
                rf"$\hat{{\theta}}={theta_found:.3f}$"
                rf"$,\ \hat{{\psi}}={psi_found:.3f}$"
            ),
        )

    # Plot Nurnberger's point if provided
    if nurnbergers_orig_point is not None:
        ax.scatter(
            nurnbergers_orig_point,
            _psi_at(nurnbergers_orig_point, thetas, psi_values),
            marker="o",
            color="black",
            s=40,
            linewidths=2,
            zorder=6,
            label=rf"$\theta_1^{{\mathrm{{orig}}}}={nurnbergers_orig_point:.3f}$",
        )

    if nurnbergers_mod_point is not None:
        ax.scatter(
            nurnbergers_mod_point,
            _psi_at(nurnbergers_mod_point, thetas, psi_values),
            marker="o",
            color="crimson",
            s=40,
            linewidths=2,
            zorder=6,
            label=rf"$\theta_{{\min}}={nurnbergers_mod_point:.3f}$",
        )
    # Path taken by the algorithm
    if theta_path is not None:
        theta_path = np.asarray(theta_path)
        theta_path = np.asarray(theta_path)
        path_psi = _psi_at(theta_path, thetas, psi_values)
        ax.plot(
            theta_path,
            path_psi,
            color="crimson",
            linestyle=":",
            linewidth=1.8,
            zorder=4,
        )
        ax.plot(
            theta_path[:-1],
            path_psi[:-1],
            color="crimson",
            linestyle="None",
            marker="o",
            markersize=6,
            zorder=4,
        )

    ax.set_xlabel(r"$\theta_1$", fontsize=15)
    ax.set_ylabel(r"$\psi(\theta_1)$", fontsize=15)

    _style_axes(ax, labelsize=14)

    ax.legend(
        fontsize=16,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=(
            4
            if nurnbergers_orig_point is not None and nurnbergers_mod_point is not None
            else 3
        ),
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
        handlelength=1.5,
    )

    fig.tight_layout()
    _save_figure(fig, file_name)

    return theta_min, psi_min


def plot_objective_psi_contour(
    theta_1_values,
    theta_2_values,
    psi_values,
    theta_found=None,
    theta_path=None,
    nurnbergers_orig_point=None,
    nurnbergers_mod_point=None,
    file_name=None,
    figsize=(7, 5),
):
    """Plot the two-knot objective as a 2D contour plot."""

    theta_1_values = np.asarray(theta_1_values)
    theta_2_values = np.asarray(theta_2_values)
    psi_values = np.asarray(psi_values)

    theta_1_grid, theta_2_grid = np.meshgrid(
        theta_1_values,
        theta_2_values,
        indexing="ij",
    )

    if psi_values.shape != theta_1_grid.shape:
        raise ValueError(
            "psi_values must have shape "
            f"{theta_1_grid.shape}, got {psi_values.shape}"
        )

    valid = (theta_1_grid < theta_2_grid) & np.isfinite(psi_values)

    surface_values = np.ma.masked_where(~valid, psi_values)

    if np.all(~valid):
        raise ValueError("No valid surface values.")

    min_index = np.nanargmin(np.where(valid, psi_values, np.nan))
    theta_1_min = theta_1_grid.flat[min_index]
    theta_2_min = theta_2_grid.flat[min_index]
    psi_min = psi_values.flat[min_index]

    fig, ax = plt.subplots(figsize=figsize)

    levels = np.linspace(
        np.nanmin(psi_values[valid]),
        np.nanmax(psi_values[valid]),
        12,
    )

    contourf = ax.contourf(
        theta_1_grid,
        theta_2_grid,
        surface_values,
        levels=levels,
        cmap="coolwarm",
    )

    ax.contour(
        theta_1_grid,
        theta_2_grid,
        surface_values,
        levels=levels,
        colors="0",
        linewidths=0.6,
    )

    cbar = fig.colorbar(contourf, ax=ax)
    cbar.set_label(r"$\psi(\theta_1,\theta_2)$", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    # Optional descent path
    if theta_path is not None:
        theta_path = np.asarray(theta_path)

        if theta_path.ndim != 2 or theta_path.shape[1] != 2:
            raise ValueError("theta_path must have shape (n_points, 2)")

        ax.plot(
            theta_path[:, 0],
            theta_path[:, 1],
            color="black",
            linestyle=":",
            linewidth=1.4,
            zorder=6,
        )

        if len(theta_path) > 1:
            ax.scatter(
                theta_path[:-1, 0],
                theta_path[:-1, 1],
                color="black",
                s=18,
                zorder=7,
            )

        if theta_found is None:
            theta_found = theta_path[-1]

    # Optional final algorithm result
    if theta_found is not None:
        theta_found = np.asarray(theta_found)

        if theta_found.shape != (2,):
            raise ValueError("theta_found must contain two values")

        ax.scatter(
            theta_found[0],
            theta_found[1],
            marker="x",
            s=65,
            facecolor="black",
            linewidths=1.8,
            zorder=8,
            label=(
                rf"$(\hat{{\theta}}_1,\hat{{\theta}}_2)="
                rf"({theta_found[0]:.3f},{theta_found[1]:.3f})$"
            ),
        )

    if nurnbergers_orig_point is not None:
        ax.scatter(
            nurnbergers_orig_point[0],
            nurnbergers_orig_point[1],
            marker="+",
            s=50,
            facecolor="black",
            linewidths=1.5,
            zorder=9,
            label=(
                rf"$(\theta_1^{{\mathrm{{orig}}}},\theta_2^{{\mathrm{{orig}}}})="
                rf"({nurnbergers_orig_point[0]:.3f},{nurnbergers_orig_point[1]:.3f})$"
            ),
        )
    if nurnbergers_mod_point is not None:
        ax.scatter(
            nurnbergers_mod_point[0],
            nurnbergers_mod_point[1],
            marker="x",
            s=50,
            facecolor="black",
            linewidths=1.5,
            zorder=9,
            label=(
                rf"$(\theta_1^{{\mathrm{{mod}}}},\theta_2^{{\mathrm{{mod}}}})="
                rf"({nurnbergers_mod_point[0]:.3f},{nurnbergers_mod_point[1]:.3f})$"
            ),
        )

    ax.set_xlabel(r"$\theta_1$", fontsize=14)
    ax.set_ylabel(r"$\theta_2$", fontsize=14)

    _style_axes(ax, labelsize=14)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    fig.tight_layout()

    if file_name is not None:
        _save_figure(fig, file_name)

    plt.close(fig)

    return (theta_1_min, theta_2_min), psi_min


def plot_objective_psi_3d(
    theta_1_values,
    theta_2_values,
    psi_values,
    theta_found=None,
    theta_path=None,
    file_name=None,
    figsize=(7, 5),
):
    """Plot a precomputed two-knot objective surface"""

    theta_1_values = np.asarray(theta_1_values)
    theta_2_values = np.asarray(theta_2_values)
    psi_values = np.asarray(psi_values)

    theta_1_grid, theta_2_grid = np.meshgrid(
        theta_1_values,
        theta_2_values,
        indexing="ij",
    )

    surface_values = np.ma.masked_where(
        ~((theta_1_grid < theta_2_grid) & np.isfinite(psi_values)),
        psi_values,
    )

    if np.all(np.isnan(surface_values)):
        raise ValueError("No valid surface values.")

    min_index = np.nanargmin(surface_values)

    theta_1_min = theta_1_grid.flat[min_index]
    theta_2_min = theta_2_grid.flat[min_index]
    psi_min = surface_values.flat[min_index]

    x_min, x_max = theta_1_values.min(), theta_1_values.max()
    y_min, y_max = theta_2_values.min(), theta_2_values.max()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.set_proj_type("ortho")
    ax.view_init(elev=28, azim=-25)
    ax.set_box_aspect((1, 1, 0.55), zoom=1.05)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Surface
    ax.plot_surface(
        theta_1_grid,
        theta_2_grid,
        surface_values,
        cmap="coolwarm",
        antialiased=True,
        alpha=0.8,
    )
    # Path taken by the algorithm
    if theta_path is not None:
        theta_path = np.asarray(theta_path)

        path_psi = _psi_at_3d(
            theta_path,
            theta_1_values,
            theta_2_values,
            psi_values,
        )
        ax.plot(
            theta_path[:, 0],
            theta_path[:, 1],
            path_psi,
            color="black",
            linestyle=":",
            linewidth=1,
            zorder=10,
            label="Iterates",
        )

        ax.scatter(
            theta_path[:-1, 0],
            theta_path[:-1, 1],
            path_psi[:-1],
            color="black",
            marker="o",
            s=2,
            depthshade=False,
            zorder=11,
        )
    if theta_path is not None and theta_found is None:
        theta_found = theta_path[-1]

        psi_found = _psi_at_3d(
            theta_found,
            theta_1_values,
            theta_2_values,
            psi_values,
        )[0]
    if theta_found is not None:
        psi_found = _psi_at_3d(
            theta_found,
            theta_1_values,
            theta_2_values,
            psi_values,
        )[0]

        ax.scatter(
            theta_found[0],
            theta_found[1],
            psi_found,
            marker="x",
            color="black",
            s=10,
            linewidths=1,
            depthshade=False,
            zorder=12,
            label=(
                rf"$\hat{{\theta}}=" rf"({theta_found[0]:.3f},{theta_found[1]:.3f})$"
            ),
        )

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("0.85")
        axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 1.0)
        axis._axinfo["grid"]["linewidth"] = 0.6

    ax.set_xlabel(r"$\theta_1$", fontsize=14, labelpad=8)
    ax.set_ylabel(r"$\theta_2$", fontsize=14, labelpad=8)
    ax.set_zlabel(r"$\psi(\theta_1,\theta_2)$", fontsize=14, labelpad=10)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.tick_params(axis="z", labelsize=11)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            fontsize=16,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )

    fig.tight_layout()

    _save_figure(fig, file_name)

    return (theta_1_min, theta_2_min), psi_min
