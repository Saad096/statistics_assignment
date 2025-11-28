# src/utils.py
import numpy as np


def build_monomial_design(t: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Build a monomial design matrix [1, t, t^2, ..., t^degree].
    t: shape (N,)
    """
    return np.vander(t, N=degree + 1, increasing=True)


def gram_matrix(A: np.ndarray) -> np.ndarray:
    """Compute Gram matrix G = A^T A."""
    return A.T @ A


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Radial Basis Function (RBF) kernel matrix between vectors x1 and x2.
    x1: shape (M,)
    x2: shape (N,)
    """
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    return np.exp(-gamma * (X1 - X2) ** 2)
