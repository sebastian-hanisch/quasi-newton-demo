"""Testfunktionen (eigenstaendige Kopie aus gradient-descent-demo/newton-verfahren-demo). Jede
Funktion liefert Funktionswert, Gradient und Hesse-Matrix, von Hand hergeleitet."""
import numpy as np


def random_spd_matrix(dim: int, condition_number: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eigs = np.exp(np.linspace(0.0, np.log(condition_number), dim))
    M = rng.normal(size=(dim, dim))
    Q, _ = np.linalg.qr(M)
    A = Q @ np.diag(eigs) @ Q.T
    return 0.5 * (A + A.T)


def quadratic(A: np.ndarray):
    """f(x) = 0.5 x^T A x, Minimum bei x*=0, f*=0."""

    def f(x):
        return 0.5 * x @ A @ x

    def grad(x):
        return A @ x

    def hess(x):
        return A

    return f, grad, hess


def rosenbrock(a: float = 1.0, b: float = 100.0):
    """f(x,y) = (a-x)^2 + b(y-x^2)^2, Minimum bei (a, a^2), f*=0."""

    def f(x):
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2

    def grad(x):
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
        dy = 2 * b * (x[1] - x[0] ** 2)
        return np.array([dx, dy])

    def hess(x):
        dxx = 2 - 4 * b * (x[1] - x[0] ** 2) + 8 * b * x[0] ** 2
        dxy = -4 * b * x[0]
        dyy = 2 * b
        return np.array([[dxx, dxy], [dxy, dyy]])

    return f, grad, hess
