# main.py
"""
Main runner for Homework 3 (Computational Part).

This script provides a simple CLI interface to run the different
computational experiments:
- Q13: Gram Matrix, LS, and Truncated SVD
- Q14: OLS vs TLS experiments
- Q15: PCA step-by-step
- Q18: Regression and kernels

Usage examples:
    python main.py                # run all experiments
    python main.py --part q13     # run only Q13
    python main.py --part q15     # run only Q15
"""

import argparse

from src.logging_config import get_logger
from src.monomial_ls_tsvd import run_monomial_experiment
from src.tls_experiments import run_tls_experiments
from src.pca_analysis import run_pca_steps
from src.regression_kernels import run_regression_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Homework 3 Computational Suite (Q13, Q14, Q15, Q18)"
    )
    parser.add_argument(
        "--part",
        type=str,
        default="all",
        choices=["all", "q13", "q14", "q15", "q18"],
        help="Which experiment to run (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("main")

    logger.info("=" * 70)
    logger.info("Mathematics and Statistics for Data Analysis – Homework 3 (Coding)")
    logger.info("=" * 70)
    logger.info(f"Selected part: {args.part}")

    if args.part in ("all", "q13"):
        logger.info("\n--- Running Q13: Monomial LS + Truncated SVD ---")
        res13 = run_monomial_experiment()
        logger.info(
            f"Q13 summary: κ(G)={res13['kappa_G']:.2e}, "
            f"||a_LS||={res13['coeff_norms'][-1]:.3f}"
        )

    if args.part in ("all", "q14"):
        logger.info("\n--- Running Q14: OLS vs TLS Experiments ---")
        res14 = run_tls_experiments()
        logger.info(
            "Q14 summary: "
            f"Exp1 OLS/TLS errors = {res14['exp1'][2]:.3f} / {res14['exp1'][3]:.3f}, "
            f"Exp2 OLS/TLS errors = {res14['exp2'][2]:.3f} / {res14['exp2'][3]:.3f}"
        )

    if args.part in ("all", "q15"):
        logger.info("\n--- Running Q15: PCA Geometry & Compression ---")
        res15 = run_pca_steps()
        logger.info(
            "Q15 summary: "
            f"off-diagonal energy (orig / PCA) = "
            f"{res15['off_orig']:.2f} / {res15['off_pca']:.2e}, "
            f"SVD vs Cov diff (k=3) = {res15['diff']:.2e}"
        )

    if args.part in ("all", "q18"):
        logger.info("\n--- Running Q18: Regression & Kernels ---")
        res18 = run_regression_comparison()
        logger.info(
            "Q18 summary: "
            f"MSEs = OLS: {res18['mse_ols']:.2e}, "
            f"Gram-Linear: {res18['mse_gram_lin']:.2e}, "
            f"Poly: {res18['mse_poly']:.2e}, "
            f"RBF: {res18['mse_rbf']:.2e}. "
            f"Best = {res18['best_method']}"
        )

    logger.info(
        "\nAll requested computations finished. "
        "See outputs/logs/homework4.log and outputs/figures/ for details."
    )


if __name__ == "__main__":
    main()
