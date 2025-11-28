# src/tls_experiments.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .logging_config import get_logger

logger = get_logger("tls_experiments")


def compute_ols(A: np.ndarray, y: np.ndarray) -> float:
    """
    Ordinary Least Squares estimate for y ≈ theta * x (A is (M,1)).
    Returns scalar theta_OLS.
    """
    # theta = (A^T A)^{-1} A^T y
    theta = np.linalg.inv(A.T @ A) @ (A.T @ y)
    return float(theta[0, 0])


def compute_tls(A: np.ndarray, y: np.ndarray) -> float:
    """
    Total Least Squares estimate from augmented matrix [A | y].
    For model y ≈ theta * x, solution is -v1/v2 where v = last right singular
    vector of [A | y].
    """
    C = np.hstack([A, y])  # shape (M, 2)
    _, _, Vt = np.linalg.svd(C)
    v = Vt.T[:, -1]  # last column of V
    v1, v2 = v[0], v[1]
    theta_tls = -v1 / v2
    return float(theta_tls)


def run_tls_experiments():
    np.random.seed(42)
    theta_true = 2.5
    M = 50
    x_clean = np.linspace(1, 10, M)
    y_clean = theta_true * x_clean
    noise_std = 1.2

    # ---------------- Experiment 1: noise only in y ----------------
    logger.info("=" * 50)
    logger.info("EXPERIMENT 1: Noise only in y (x is clean)")
    logger.info("=" * 50)

    x_obs1 = x_clean.copy()
    y_obs1 = y_clean + np.random.normal(0, noise_std, M)

    A1 = x_obs1.reshape(-1, 1)
    y1 = y_obs1.reshape(-1, 1)

    theta_OLS1 = compute_ols(A1, y1)
    theta_TLS1 = compute_tls(A1, y1)

    err_OLS1 = abs(theta_OLS1 - theta_true)
    err_TLS1 = abs(theta_TLS1 - theta_true)

    logger.info(f"True theta:   {theta_true:.4f}")
    logger.info(f"OLS estimate: {theta_OLS1:.4f} (error = {err_OLS1:.4f})")
    logger.info(f"TLS estimate: {theta_TLS1:.4f} (error = {err_TLS1:.4f})")

    # ---------------- Experiment 2: noise in both x and y ----------------
    logger.info("=" * 50)
    logger.info("EXPERIMENT 2: Noise in both x and y")
    logger.info("=" * 50)

    x_obs2 = x_clean + np.random.normal(0, noise_std, M)
    y_obs2 = y_clean + np.random.normal(0, noise_std, M)

    A2 = x_obs2.reshape(-1, 1)
    y2 = y_obs2.reshape(-1, 1)

    theta_OLS2 = compute_ols(A2, y2)
    theta_TLS2 = compute_tls(A2, y2)

    err_OLS2 = abs(theta_OLS2 - theta_true)
    err_TLS2 = abs(theta_TLS2 - theta_true)

    logger.info(f"True theta:   {theta_true:.4f}")
    logger.info(f"OLS estimate: {theta_OLS2:.4f} (error = {err_OLS2:.4f})")
    logger.info(f"TLS estimate: {theta_TLS2:.4f} (error = {err_TLS2:.4f})")

    # Plot
    fig_path = Path("outputs/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    fname = fig_path / "tls_experiments.png"

    plt.figure(figsize=(12, 5))

    # Exp 1
    plt.subplot(1, 2, 1)
    plt.scatter(x_obs1, y_obs1, alpha=0.6, label="Data (y noisy)")
    plt.plot(x_clean, y_clean, "k--", label="True line")
    plt.plot(x_clean, theta_OLS1 * x_clean, "r-", label=f"OLS (err={err_OLS1:.3f})")
    plt.plot(x_clean, theta_TLS1 * x_clean, "b-", label=f"TLS (err={err_TLS1:.3f})")
    plt.title("Exp 1: Noise only in y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    # Exp 2
    plt.subplot(1, 2, 2)
    plt.scatter(x_obs2, y_obs2, alpha=0.6, label="Data (x & y noisy)")
    plt.plot(x_clean, y_clean, "k--", label="True line")
    plt.plot(x_clean, theta_OLS2 * x_clean, "r-", label=f"OLS (err={err_OLS2:.3f})")
    plt.plot(x_clean, theta_TLS2 * x_clean, "b-", label=f"TLS (err={err_TLS2:.3f})")
    plt.title("Exp 2: Noise in x and y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    logger.info(f"Saved figure to {fname}")

    return {
        "theta_true": theta_true,
        "exp1": (theta_OLS1, theta_TLS1, err_OLS1, err_TLS1),
        "exp2": (theta_OLS2, theta_TLS2, err_OLS2, err_TLS2),
    }


if __name__ == "__main__":
    run_tls_experiments()
