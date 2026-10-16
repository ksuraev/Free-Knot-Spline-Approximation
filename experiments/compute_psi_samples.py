import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(ROOT))

import parallel_psi_samples

import multi_knot_approx
import one_knot_approx
import plotting
import test_functions


def compute_psi_samples_1d(f, a, b, m, n, theta_start, theta_end, nsamples, step=0.05):
    """Compute Psi(theta) for a range of theta values between theta_start and theta_end."""
    if nsamples is None:
        thetas = np.arange(theta_start, theta_end, step)
    else:
        thetas = np.linspace(theta_start, theta_end, nsamples)

    psi_values = np.array(
        [one_knot_approx.Psi_bar(f, a, b, theta, m, n)["d_max"] for theta in thetas]
    )

    return thetas, psi_values


def compute_psi_samples_1d_simplex(
    f, a, b, m, n, theta_start, theta_end, nsamples, step=0.05
):
    """Compute Psi(theta) for a range of theta values between theta_start and theta_end."""
    if nsamples is None:
        thetas = np.arange(theta_start, theta_end, step)
    else:
        thetas = np.linspace(theta_start, theta_end, nsamples)
    psi_values = []
    for theta in thetas:
        if not (a < theta < b):
            raise ValueError(f"Theta value {theta} is out of bounds [{a}, {b}]")
        (
            z,
            A,
            _,
        ) = multi_knot_approx.Psi_bar(f, np.array([a, theta, b]), m, nsamples=1000)
        psi_values.append(z)

    min_index = np.nanargmin(psi_values)
    optimal_theta = thetas[min_index]
    optimal_psi = psi_values[min_index]
    plotting.plot_objective_psi_bar(
        thetas,
        psi_values,
        optimal_theta,
        optimal_psi,
        file_name=f"psi_surface_{function}_k{1}_m{m}_10000",
    )

    return thetas, psi_values


def compute_psi_samples_2d(
    f,
    a,
    b,
    m,
    theta_1_start,
    theta_1_end,
    theta_2_start,
    theta_2_end,
    nsamples,
    step=0.05,
):
    """Compute Psi(theta_1, theta_2) for a range of theta_1 and theta_2 values."""
    if nsamples is not None:
        theta_1_values = np.linspace(theta_1_start, theta_1_end, nsamples)
        theta_2_values = np.linspace(theta_2_start, theta_2_end, nsamples)
    else:
        theta_1_values = np.arange(theta_1_start, theta_1_end, step)
        theta_2_values = np.arange(theta_2_start, theta_2_end, step)

    # Initialise a 2D array to store psi values
    psi_values = np.full((len(theta_1_values), len(theta_2_values)), np.nan)

    for i, theta_1 in enumerate(tqdm(theta_1_values, desc=f"m={m}", leave=False)):
        # Find the index in theta_2_values where theta_2 > theta_1
        j_start = np.searchsorted(theta_2_values, theta_1, side="right")

        for j in range(j_start, len(theta_2_values)):
            theta_2 = theta_2_values[j]

            knots = np.array([a, theta_1, theta_2, b])
            _, _, _, psi_value = multi_knot_approx.evaluate_and_get_direction(
                f, knots, m, nsamples=nsamples
            )

            if psi_value is not None:
                psi_values[i, j] = psi_value

    return theta_1_values, theta_2_values, psi_values


def compute_psi_samples_2d_simplex(
    f,
    a,
    b,
    m,
    theta_1_start,
    theta_1_end,
    theta_2_start,
    theta_2_end,
    nsamples_theta=10000,
    nsamples_t=1000,
):
    """Compute Psi_bar at 10,000 sampled knot pairs."""

    rng = np.random.default_rng(0)

    theta_1_values = []
    theta_2_values = []
    psi_values = []

    while len(theta_1_values) < nsamples_theta:
        theta_1 = rng.uniform(theta_1_start, theta_1_end)
        theta_2 = rng.uniform(theta_2_start, theta_2_end)

        if theta_1 >= theta_2:
            continue

        knots = np.array(
            [
                a,
                theta_1,
                theta_2,
                b,
            ]
        )

        z, _, _ = multi_knot_approx.Psi_bar(
            f,
            knots,
            m,
            nsamples=nsamples_t,
        )

        theta_1_values.append(theta_1)
        theta_2_values.append(theta_2)
        psi_values.append(abs(z))

    return (
        np.asarray(theta_1_values),
        np.asarray(theta_2_values),
        np.asarray(psi_values),
    )


if __name__ == "__main__":
    experiments = [
        (function, k, m)
        for function in test_functions.TEST_FUNCTIONS
        for k in [2]
        for m in range(1, 4)
    ]

    for function, k, m in experiments:
        print(f"Computing {function}, k={k}, m={m}")
        f, _ = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]
        if k == 1:
            if not Path(f"psi_surface_{function}_k{k}_m{m}_10000.npz").exists():
                thetas, psi_values = compute_psi_samples_1d_simplex(
                    f, a, b, m, k + 1, a + 0.1, b - 0.1, nsamples=10000
                )
                np.savez(
                    f"psi_surface_{function}_k{k}_m{m}_10000.npz",
                    theta_values=thetas,
                    psi_values=psi_values,
                )
        else:
            if not Path(f"psi_surface_{function}_k{k}_m{m}_10000.npz").exists():
                theta_1_values, theta_2_values, psi_values = (
                    parallel_psi_samples.compute_psi_samples_2d_simplex_parallel(
                        function=function,
                        a=a,
                        b=b,
                        m=m,
                        theta_1_start=a + 0.1,
                        theta_1_end=b - 0.1,
                        theta_2_start=a + 0.1,
                        theta_2_end=b - 0.1,
                        nsamples_theta=10000,
                        nsamples_t=1000,
                    )
                )
                min_index = np.nanargmin(psi_values)

                optimal_theta_1 = theta_1_values[min_index]
                optimal_theta_2 = theta_2_values[min_index]
                optimal_psi = psi_values[min_index]

                np.savez(
                    f"psi_surface_{function}_k2_m{m}_10000.npz",
                    theta_1_values=theta_1_values,
                    theta_2_values=theta_2_values,
                    psi_values=psi_values,
                )
                print(
                    "Optimal sampled knots:",
                    optimal_theta_1,
                    optimal_theta_2,
                )
                print("Optimal sampled Psi:", optimal_psi)
                plotting.plot_objective_psi_bar_contour_scattered(
                    theta_1_values,
                    theta_2_values,
                    psi_values,
                    theta_found=(
                        optimal_theta_1,
                        optimal_theta_2,
                    ),
                    title=r"Contour plot of $\overline{\Psi}(\theta_1,\theta_2)$",
                    file_name=f"psi_surface_{function}_k{k}_m{m}_10000",
                )

            # plotting.plot_objective_psi_bar_contour(
            #     theta_1_values,
            #     theta_2_values,
            #     psi_values,
            #     theta_found=(optimal_theta_1, optimal_theta_2),
            #     title=r"Contour plot of $\overline{\Psi}(\theta_1, \theta_2)$",
            #     file_name=f"psi_surface_{function}_k{k}_m{m}_10000",
            # )
            # theta_1_values, theta_2_values, psi_values = compute_psi_samples_2d(
            #     f, a, b, m, a + 0.1, b - 0.1, a + 0.1, b - 0.1, step=0.1
            # )
            # np.savez(
            #     f"psi_surface_{function}_k{k}_m{m}.npz",
            #     theta_1_values=theta_1_values,
            #     theta_2_values=theta_2_values,
            #     psi_values=psi_values,
            # )
