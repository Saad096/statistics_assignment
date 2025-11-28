# src/regression_kernels.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path

from .logging_config import get_logger
from .utils import rbf_kernel

logger = get_logger("regression_kernels")


def run_regression_comparison():
    np.random.seed(42)
    M = 50
    x = np.linspace(-3, 3, M)
    y_true = 2 * np.sin(2 * x) + 0.5 * x**3
    y_obs = y_true + np.random.normal(0, 0.5, M)
    y = y_obs.reshape(-1, 1)

    lambda_reg = 0.1
    gamma = 0.5

    logger.info("=" * 60)
    logger.info("REGRESSION COMPARISON: Explicit vs. Kernel Methods")
    logger.info("=" * 60)
    logger.info("True function: y = 2*sin(2x) + 0.5*x^3")
    logger.info(f"Samples: {M}, Noise std: 0.5")

    # ---- (a) OLS and polynomial regression ----
    # Linear features [1, x]
    Phi_lin = np.vstack([np.ones_like(x), x]).T  # (M, 2)
    w_lin, *_ = np.linalg.lstsq(Phi_lin, y, rcond=None)
    y_ols_pred = (Phi_lin @ w_lin).flatten()

    # Polynomial features [1, x, ..., x^5]
    Phi_poly = np.vander(x, N=6, increasing=True)  # (M, 6)
    w_poly, *_ = np.linalg.lstsq(Phi_poly, y, rcond=None)
    y_poly_pred = (Phi_poly @ w_poly).flatten()

    # ---- (b) Gramian Ridge (linear features) ----
    K_lin = Phi_lin @ Phi_lin.T  # (M, M)
    alpha_lin = np.linalg.solve(K_lin + lambda_reg * np.eye(M), y)
    y_gram_lin_pred = (K_lin @ alpha_lin).flatten()

    # ---- (c) RBF Kernel Ridge ----
    K_rbf = rbf_kernel(x, x, gamma)  # (M, M)
    alpha_rbf = np.linalg.solve(K_rbf + lambda_reg * np.eye(M), y)
    y_rbf_pred = (K_rbf @ alpha_rbf).flatten()

    # ---- (d) MSE analysis ----
    mse_ols = mean_squared_error(y_true, y_ols_pred)
    mse_poly = mean_squared_error(y_true, y_poly_pred)
    mse_gram_lin = mean_squared_error(y_true, y_gram_lin_pred)
    mse_rbf = mean_squared_error(y_true, y_rbf_pred)

    logger.info("=" * 60)
    logger.info("MEAN SQUARED ERROR (MSE) vs. TRUE FUNCTION")
    logger.info("=" * 60)
    logger.info(f"{'Method':<25} {'MSE':<12}")
    logger.info("-" * 60)
    logger.info(f"{'Linear OLS':<25} {mse_ols:<12.4e}")
    logger.info(f"{'Gramian (Linear)':<25} {mse_gram_lin:<12.4e}")
    logger.info(f"{'Polynomial (deg 5)':<25} {mse_poly:<12.4e}")
    logger.info(f"{'RBF Kernel':<25} {mse_rbf:<12.4e}")

    methods = {
        "Linear OLS": mse_ols,
        "Gramian (Linear)": mse_gram_lin,
        "Polynomial (deg 5)": mse_poly,
        "RBF Kernel": mse_rbf,
    }
    best_method = min(methods, key=methods.get)
    logger.info(f"Best method: {best_method} (MSE = {methods[best_method]:.4e})")

    # ---- Plotting ----
    x_fine = np.linspace(-3, 3, 200)
    y_true_fine = 2 * np.sin(2 * x_fine) + 0.5 * x_fine**3

    Phi_lin_fine = np.vstack([np.ones_like(x_fine), x_fine]).T
    y_ols_fine = (Phi_lin_fine @ w_lin).flatten()

    # Gramian (Linear) is identical to OLS for linear features
    y_gram_lin_fine = y_ols_fine.copy()

    Phi_poly_fine = np.vander(x_fine, N=6, increasing=True)
    y_poly_fine = (Phi_poly_fine @ w_poly).flatten()

    # Kernel predictions on fine grid
    # RBF kernel between fine grid points (rows) and training points (cols)
    K_rbf_test = np.exp(-gamma * (x_fine[:, None] - x[None, :]) ** 2)
    y_rbf_fine = (K_rbf_test @ alpha_rbf).flatten()

    fig_path = Path("outputs/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    fname = fig_path / "regression_comparison.png"

    plt.figure(figsize=(12, 7))
    plt.scatter(x, y_obs, color="gray", alpha=0.6, label="Observed Data")
    plt.plot(x_fine, y_true_fine, "k--", linewidth=2, label="True Function")

    plt.plot(
        x_fine, y_ols_fine, "r-", linewidth=2, label=f"Linear OLS (MSE={mse_ols:.1e})"
    )

    plt.plot(
        x_fine,
        y_gram_lin_fine,
        color="orange",
        linestyle=":",
        linewidth=2,
        marker=".",
        markevery=20,
        label=f"Gramian (Linear) (MSE={mse_gram_lin:.1e})",
    )

    plt.plot(
        x_fine,
        y_poly_fine,
        "g--",
        linewidth=2,
        label=f"Poly Deg 5 (MSE={mse_poly:.1e})",
    )

    plt.plot(
        x_fine, y_rbf_fine, "b-", linewidth=2, label=f"RBF Kernel (MSE={mse_rbf:.1e})"
    )

    plt.title("Regression Methods Compared")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    logger.info(f"Saved regression comparison to {fname}")

    return {
        "mse_ols": mse_ols,
        "mse_poly": mse_poly,
        "mse_gram_lin": mse_gram_lin,
        "mse_rbf": mse_rbf,
        "best_method": best_method,
    }


if __name__ == "__main__":
    run_regression_comparison()
