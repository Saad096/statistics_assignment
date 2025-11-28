# src/monomial_ls_tsvd.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .logging_config import get_logger
from .utils import build_monomial_design, gram_matrix

logger = get_logger("monomial_ls_tsvd")


def run_monomial_experiment(N: int = 50):
    # 1) Sample f(t) = e^t
    t = np.linspace(0, 1, N)
    y = np.exp(t)

    # 2) Design matrix, Gram matrix, RHS
    A = build_monomial_design(t, degree=2)  # columns [1, t, t^2]
    G = gram_matrix(A)
    b = A.T @ y

    # 3) Solve LS via normal equations
    a_LS = np.linalg.solve(G, b)
    kappa_G = np.linalg.cond(G)

    logger.info("--- (a) LS Solution ---")
    logger.info(f"a_LS = {a_LS}")
    logger.info(f"Condition number κ(G) = {kappa_G:.2e}")

    # 4) Truncated SVD with R' = 2
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    R_prime = 2

    Sigma_inv_trunc = np.diag(1.0 / s[:R_prime])
    A_dag_trunc = V[:, :R_prime] @ Sigma_inv_trunc @ U[:, :R_prime].T
    a_TSVD = A_dag_trunc @ y

    logger.info("--- (b) TSVD Solution (R'=2) ---")
    logger.info(f"Singular values of A: {s}")
    logger.info(f"a_TSVD = {a_TSVD}")

    # 5) Norms of coefficients
    norm_LS = np.linalg.norm(a_LS)
    norm_TSVD = np.linalg.norm(a_TSVD)

    logger.info("--- (c) Coefficient Norms ---")
    logger.info(f"||a_LS||_2   = {norm_LS:.4f}")
    logger.info(f"||a_TSVD||_2 = {norm_TSVD:.4f}")

    # 6) Plot approximations
    t_fine = np.linspace(0, 1, 200)
    p_LS = a_LS[0] + a_LS[1] * t_fine + a_LS[2] * t_fine**2
    p_TSVD = a_TSVD[0] + a_TSVD[1] * t_fine + a_TSVD[2] * t_fine**2

    fig_path = Path("outputs/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    fname = fig_path / "monomial_ls_tsvd.png"

    plt.figure(figsize=(10, 6))
    plt.plot(t_fine, np.exp(t_fine), "k--", label=r"$f(t) = e^t$")
    plt.plot(t_fine, p_LS, "r-", label="LS Solution")
    plt.plot(t_fine, p_TSVD, "b-", label="TSVD Solution (R'=2)")
    plt.scatter(t, y, color="gray", s=10, alpha=0.5, label="Sampled data")
    plt.title(f"Approximation of e^t (κ(G) = {kappa_G:.2e})")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

    logger.info(f"Saved figure to {fname}")

    # 7) Residual analysis
    res_LS = np.linalg.norm(A @ a_LS - y)
    res_TSVD = np.linalg.norm(A @ a_TSVD - y)

    logger.info("--- (d) Residual Errors ---")
    logger.info(f"LS residual      = {res_LS:.6f}")
    logger.info(f"TSVD residual    = {res_TSVD:.6f}")

    # Sweep over truncation levels R' = 1, 2, 3
    R_vals = [1, 2, 3]
    residuals = []
    coeff_norms = []
    for R in R_vals:
        Sigma_inv_R = np.diag(1.0 / s[:R])
        A_dag_R = V[:, :R] @ Sigma_inv_R @ U[:, :R].T
        a_R = A_dag_R @ y
        residuals.append(np.linalg.norm(A @ a_R - y))
        coeff_norms.append(np.linalg.norm(a_R))

    # You can optionally also save plots of residual vs R'.
    # For the assignment, a short explanation goes in the notebook.

    return {
        "a_LS": a_LS,
        "a_TSVD": a_TSVD,
        "res_LS": res_LS,
        "res_TSVD": res_TSVD,
        "kappa_G": kappa_G,
        "R_vals": R_vals,
        "residuals": residuals,
        "coeff_norms": coeff_norms,
    }


if __name__ == "__main__":
    run_monomial_experiment()
