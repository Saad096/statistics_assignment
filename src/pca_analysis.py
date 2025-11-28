# src/pca_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .logging_config import get_logger

logger = get_logger("pca_analysis")


def run_pca_steps():
    # 16x16 digit image X from your template
    X = np.array(
        [
            [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 200, 200, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 200, 200, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 200, 200, 200, 200, 200, 200, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=float,
    )

    logger.info("=" * 60)
    logger.info("PCA Step-by-Step: Geometry, Decorrelation, Compression")
    logger.info("=" * 60)
    logger.info(f"Image shape: {X.shape}, Total values: {X.size}")

    # Center columns
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    n = X_centered.shape[0]

    # --- (a) Geometric transformation via SVD ---
    logger.info("---- (a) Geometric Transformation ----")
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt.T
    Z = X_centered @ V  # rotation to principal axes

    logger.info(f"First principal component v1: {V[:, 0]}")

    fig_path = Path("outputs/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    fname = fig_path / "pca_steps.png"

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap="gray", vmin=0, vmax=255)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(Z, cmap="gray", aspect="auto")
    plt.title("Rotated (Z)")
    plt.xlabel("PC")
    plt.ylabel("Row")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    logger.info(f"Saved PCA step figure to {fname}")

    # --- (b) Statistical decorrelation ---
    logger.info("---- (b) Statistical Decorrelation ----")
    C_orig = (1.0 / n) * (X_centered.T @ X_centered)
    C_pca = (1.0 / n) * (Z.T @ Z)

    off_orig = np.sqrt(np.sum(C_orig**2) - np.sum(np.diag(C_orig) ** 2))
    off_pca = np.sqrt(np.sum(C_pca**2) - np.sum(np.diag(C_pca) ** 2))

    logger.info(f"Off-diagonal energy: Original = {off_orig:.2f}, PCA = {off_pca:.2e}")

    # Heatmaps (optional second figure)
    fname_cov = fig_path / "pca_covariance_heatmaps.png"
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(C_orig, cmap="coolwarm")
    plt.title("Cov (Original)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(C_pca, cmap="coolwarm")
    plt.title("Cov (PCA)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(fname_cov, dpi=300)
    plt.close()
    logger.info(f"Saved covariance heatmaps to {fname_cov}")

    # --- (c) Compression with different k ---
    logger.info("---- (c) Dimensionality Reduction (SVD) ----")
    k_values = [1, 3, 5, 10, 16]
    reconstructions_svd = {}
    errors_svd = {}

    for k in k_values:
        Uk = U[:, :k]
        Sk = np.diag(s[:k])
        Vk = V[:, :k]
        X_rec_centered = Uk @ Sk @ Vk.T
        X_rec = X_rec_centered + X_mean
        reconstructions_svd[k] = X_rec
        err = np.linalg.norm(X - X_rec, "fro")
        errors_svd[k] = err
        logger.info(f"k={k:2d} | Frobenius error = {err:6.2f}")

    storage_k3 = 17 * 3 + 16  # as in assignment
    comp_ratio = X.size / storage_k3
    logger.info(
        f"Storage: Original={X.size}, k=3 uses {storage_k3} → "
        f"{comp_ratio:.1f}× compression"
    )

    # Variance curve
    cumsum_var = np.cumsum(s**2) / np.sum(s**2)
    fname_var = fig_path / "pca_variance_curve.png"
    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(s) + 1), cumsum_var, "bo-")
    plt.axhline(0.95, color="r", ls="--", label="95%")
    plt.xlabel("k")
    plt.ylabel("Cumulative Variance")
    plt.legend()
    plt.grid(True)
    plt.title("Variance Explained")
    plt.tight_layout()
    plt.savefig(fname_var, dpi=300)
    plt.close()
    logger.info(f"Saved variance curve to {fname_var}")

    # --- (d) Covariance method ---
    logger.info("---- (d) Covariance Method vs SVD (k=3) ----")
    C = (1.0 / n) * (X_centered.T @ X_centered)
    eigenvals, eigenvecs = np.linalg.eigh(C)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    k = 3
    V_cov = eigenvecs[:, :k]
    X_rec_cov = X_centered @ V_cov @ V_cov.T + X_mean

    error_cov = np.linalg.norm(X - X_rec_cov, "fro")
    error_svd_k3 = errors_svd[3]
    diff = np.linalg.norm(reconstructions_svd[3] - X_rec_cov, "fro")

    logger.info(f"SVD error (k=3):     {error_svd_k3:.6f}")
    logger.info(f"Covariance error:    {error_cov:.6f}")
    logger.info(f"Reconstruction diff: {diff:.2e}")

    return {
        "errors_svd": errors_svd,
        "error_cov": error_cov,
        "diff": diff,
        "off_orig": off_orig,
        "off_pca": off_pca,
    }


if __name__ == "__main__":
    run_pca_steps()
