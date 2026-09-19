import numpy as np
from tqdm import tqdm

import extras
import one_knot_approx
import test_functions


def compute_psi_samples_1d(f, a, b, m, n, theta_start, theta_end, step=0.05):
    """Compute Psi(theta) for a range of theta values between theta_start and theta_end."""
    thetas = np.arange(theta_start, theta_end, step)

    psi_values = np.array(
        [one_knot_approx.psi(f, a, b, theta, m, n)["d_max"] for theta in thetas]
    )

    return thetas, psi_values


def compute_psi_samples_2d(
    f, a, b, m, theta_1_start, theta_1_end, theta_2_start, theta_2_end, step=0.05
):
    """Compute Psi(theta_1, theta_2) for a range of theta_1 and theta_2 values."""
    theta_1_values = np.arange(theta_1_start, theta_1_end, step)
    theta_2_values = np.arange(theta_2_start, theta_2_end, step)

    psi_values = np.full(
        (len(theta_1_values), len(theta_2_values)),
        np.nan,
    )

    for i, theta_1 in enumerate(tqdm(theta_1_values, desc=f"m={m}", leave=False)):
        j_start = np.searchsorted(
            theta_2_values,
            theta_1,
            side="right",
        )

        for j in range(j_start, len(theta_2_values)):
            theta_2 = theta_2_values[j]

            knots = np.array([a, theta_1, theta_2, b])
            _, psi_values[i, j] = extras.solve_simplex(f, knots, m)

    return theta_1_values, theta_2_values, psi_values


if __name__ == "__main__":
    for function in test_functions.TEST_FUNCTIONS:
        f, f_label = test_functions.TEST_FUNCTIONS[function]
        a, b = test_functions.INTERVALS[function]

        for k in [1, 2]:
            for m in range(1, 4):
                print(f"Computing {function}, k={k}, m={m}")
                if k == 1:
                    theta_l = a + 0.1
                    theta_r = b - 0.1
                    thetas, psi_values = compute_psi_samples_1d(
                        f, a, b, m, k + 1, theta_l, theta_r, step=0.01
                    )
                    np.savez(
                        f"psi_surface_{function}_k{k}_m{m}.npz",
                        theta_values=thetas,
                        psi_values=psi_values,
                    )
                else:
                    theta_1_values, theta_2_values, psi_values = compute_psi_samples_2d(
                        f,
                        a,
                        b,
                        m,
                        a + 0.1,
                        b - 0.1,
                        a + 0.1,
                        b - 0.1,
                        step=0.05,
                    )
                    np.savez(
                        f"psi_surface_{function}_k{k}_m{m}.npz",
                        theta_1_values=theta_1_values,
                        theta_2_values=theta_2_values,
                        psi_values=psi_values,
                    )
