"""BFGS und L-BFGS (Quasi-Newton), plus eigenstaendige Kopien von Newton-Verfahren und
Gradientenabstieg aus Stueck 1/2, nur fuer die Vergleichscharts."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    trajectory: np.ndarray
    fvals: np.ndarray
    converged: bool
    n_iter: int


def _backtracking_step(f, x, d, fx, g, alpha=1e-4, beta=0.5, max_ls=60):
    step = 1.0
    for _ in range(max_ls):
        if f(x + step * d) <= fx + alpha * step * (g @ d):
            return step
        step *= beta
    return step


def _exact_step_quadratic(A, g, d):
    """Geschlossene Formel fuer die exakte Liniensuche auf f(x)=0.5 x^T A x entlang d."""
    denom = d @ A @ d
    if denom <= 0:
        return 1.0
    return -(g @ d) / denom


def bfgs_method(f, grad, x0, H0=None, mode="backtracking", A=None, max_iter=200,
                tol=1e-10) -> Result:
    """Volles BFGS mit dichter inverser Hesse-Approximation (Sherman-Morrison-Update).
    mode='exact' braucht A (nur fuer Quadratiken definiert, exakte Liniensuche)."""
    n = len(x0)
    x = np.asarray(x0, dtype=float).copy()
    H = H0.copy() if H0 is not None else np.eye(n)
    traj = [x.copy()]
    g = grad(x)
    fvals = [float(f(x))]
    converged = False
    for k in range(1, max_iter + 1):
        if float(g @ g) < tol ** 2:
            converged = True
            break
        d = -H @ g
        if mode == "exact":
            step = _exact_step_quadratic(A, g, d)
        else:
            step = _backtracking_step(f, x, d, fvals[-1], g)
        x_new = x + step * d
        g_new = grad(x_new)
        s = x_new - x
        y = g_new - g
        sy = float(s @ y)
        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, g = x_new, g_new
        traj.append(x.copy())
        fvals.append(float(f(x)))
        if not np.all(np.isfinite(x)):
            break
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1)


def lbfgs_method(f, grad, x0, m=10, mode="backtracking", A=None, max_iter=1000,
                 tol=1e-10) -> Result:
    """L-BFGS: Zwei-Schleifen-Rekursion, speichert nur die letzten m (s,y)-Paare statt einer
    vollen n x n-Matrix."""
    x = np.asarray(x0, dtype=float).copy()
    traj = [x.copy()]
    g = grad(x)
    fvals = [float(f(x))]
    converged = False
    s_hist, y_hist, rho_hist = [], [], []
    for k in range(1, max_iter + 1):
        if float(g @ g) < tol ** 2:
            converged = True
            break
        q = g.copy()
        alphas = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            alpha = rho * float(s @ q)
            q = q - alpha * y
            alphas.append(alpha)
        alphas.reverse()
        if s_hist:
            gamma = float(s_hist[-1] @ y_hist[-1]) / float(y_hist[-1] @ y_hist[-1])
        else:
            gamma = 1.0
        r = gamma * q
        for (s, y, rho), alpha in zip(zip(s_hist, y_hist, rho_hist), alphas):
            beta = rho * float(y @ r)
            r = r + s * (alpha - beta)
        d = -r
        if mode == "exact":
            step = _exact_step_quadratic(A, g, d)
        else:
            step = _backtracking_step(f, x, d, fvals[-1], g)
        x_new = x + step * d
        g_new = grad(x_new)
        s = x_new - x
        y = g_new - g
        sy = float(s @ y)
        if sy > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        x, g = x_new, g_new
        traj.append(x.copy())
        fvals.append(float(f(x)))
        if not np.all(np.isfinite(x)):
            break
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1)


def newton_method(f, grad, hess, x0, max_iter=50, tol=1e-12) -> Result:
    """Eigenstaendige Kopie aus newton-verfahren-demo, nur fuer den Vier-Wege-Vergleich hier."""
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    traj = [x.copy()]
    fvals = [float(f(x))]
    converged = False
    for k in range(1, max_iter + 1):
        g = grad(x)
        if float(g @ g) < tol ** 2:
            converged = True
            break
        H = hess(x)
        x = x - np.linalg.solve(np.atleast_2d(H), g)
        traj.append(x.copy())
        if not np.all(np.isfinite(x)):
            break
        fvals.append(float(f(x)))
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1)


def gradient_descent(f, grad, x0, eta, max_iter=5000, tol=1e-10) -> Result:
    """Eigenstaendige Kopie aus gradient-descent-demo, nur fuer den Vier-Wege-Vergleich hier."""
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    traj = [x.copy()]
    fvals = [float(f(x))]
    converged = False
    for k in range(1, max_iter + 1):
        g = grad(x)
        if float(g @ g) < tol ** 2:
            converged = True
            break
        x = x - eta * g
        traj.append(x.copy())
        fvals.append(float(f(x)))
    return Result(trajectory=np.array(traj), fvals=np.array(fvals), converged=converged,
                  n_iter=len(traj) - 1)
