"""Kennzahlen: Settings-Dataclass, analyse()-Einstiegspunkt, Reduktions-Check auf Newton,
endliche-Terminierungs-Check, Speicherbedarf-Vergleich, Vier-Wege-Vergleich (GD/Newton/BFGS/
L-BFGS), L-BFGS-vs-BFGS-Konditionszahl-Sweep, Gradienten-Check."""
from dataclasses import dataclass

import numpy as np

import qn_functions as fn
import qn_optimizer as opt


@dataclass(frozen=True)
class Settings:
    func: str
    optimizer: str
    max_iter: int
    seed: int
    condition_number: float = 20.0
    m: int = 10


def analyse(settings: Settings) -> dict:
    if settings.func == "quadratic":
        A = fn.random_spd_matrix(2, settings.condition_number, settings.seed)
        f, grad, hess = fn.quadratic(A)
        x_star = np.zeros(2)
        rng = np.random.default_rng(settings.seed + 1)
        x_start = rng.normal(size=2)
    else:
        f, grad, hess = fn.rosenbrock()
        x_star = np.array([1.0, 1.0])
        x_start = np.array([-1.2, 1.0])  # der klassische, in der Literatur zitierte Startpunkt

    # Fuer die Quadratik ist eine exakte Liniensuche billig verfuegbar (geschlossene Formel,
    # siehe Stueck 1) - damit zeigt die interaktive App auch die endliche Terminierung aus der
    # Korrektheits-Kette, nicht nur ein qualitatives Beispiel. Rosenbrock hat keine geschlossene
    # Formel, dort bleibt es bei Backtracking.
    mode = "exact" if settings.func == "quadratic" else "backtracking"

    if settings.optimizer == "bfgs":
        result = opt.bfgs_method(f, grad, x_start, mode=mode, A=(A if mode == "exact" else None),
                                 max_iter=settings.max_iter)
    elif settings.optimizer == "lbfgs":
        result = opt.lbfgs_method(f, grad, x_start, m=settings.m, mode=mode,
                                  A=(A if mode == "exact" else None), max_iter=settings.max_iter)
    elif settings.optimizer == "newton":
        result = opt.newton_method(f, grad, hess, x_start, max_iter=settings.max_iter)
    else:
        lam_max = np.linalg.eigvalsh(fn.random_spd_matrix(2, settings.condition_number,
                                                          settings.seed)).max() if settings.func == "quadratic" else 400.0
        result = opt.gradient_descent(f, grad, x_start, eta=1.0 / lam_max,
                                      max_iter=settings.max_iter)
    return {"result": result, "f_star": 0.0, "x_star": x_star}


def reduction_to_newton_check(dim=5, condition_number=20.0, seed=0) -> float:
    """BFGS mit H0=A^-1 (exakte inverse Hesse-Matrix) reproduziert Newtons ersten Schritt exakt."""
    A = fn.random_spd_matrix(dim, condition_number, seed)
    f, grad, hess = fn.quadratic(A)
    Ainv = np.linalg.inv(A)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=dim)
    r_bfgs = opt.bfgs_method(f, grad, x0, H0=Ainv, mode="exact", A=A, max_iter=5)
    r_newton = opt.newton_method(f, grad, hess, x0, max_iter=5)
    n = min(len(r_bfgs.trajectory), len(r_newton.trajectory))
    return float(np.max(np.abs(r_bfgs.trajectory[:n] - r_newton.trajectory[:n])))


def finite_termination_check(dims=(2, 5, 10, 15, 20, 25, 30, 40, 50), condition_number=20.0,
                             seed=2) -> list:
    """BFGS mit H0=I und exakter Liniensuche loest eine n-dimensionale Quadratik in HOECHSTENS n
    Schritten - exakt in Gleitkomma-Arithmetik bis n~20, danach ein moderater, gemessener
    Rundungsfehler-Aufschlag."""
    rows = []
    for n in dims:
        A = fn.random_spd_matrix(n, condition_number, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 10)
        x0 = rng.normal(size=n)
        result = opt.bfgs_method(f, grad, x0, mode="exact", A=A, max_iter=n + 20)
        rows.append({"n": n, "n_iter": result.n_iter, "within_bound": result.n_iter <= n,
                    "f_final": result.fvals[-1]})
    return rows


def memory_footprint_comparison(ns=(10, 100, 1000, 10000), m=10) -> list:
    """BFGS speichert eine volle n x n-Matrix, L-BFGS nur m Vektorpaare der Laenge n - eine
    strukturelle, exakt zaehlbare Zahl, keine Laufzeitmessung."""
    rows = []
    for n in ns:
        bfgs_numbers = n * n
        lbfgs_numbers = m * n * 2
        rows.append({"n": n, "bfgs_numbers": bfgs_numbers, "lbfgs_numbers": lbfgs_numbers,
                    "ratio": bfgs_numbers / lbfgs_numbers})
    return rows


def four_way_comparison(dim=10, condition_number=100.0, seed=5) -> dict:
    """Gradientenabstieg vs. Newton vs. BFGS vs. L-BFGS auf derselben Quadratik."""
    A = fn.random_spd_matrix(dim, condition_number, seed)
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=dim)
    lam_max = float(np.linalg.eigvalsh(A).max())
    r_gd = opt.gradient_descent(f, grad, x0, eta=1.0 / lam_max, max_iter=5000)
    r_newton = opt.newton_method(f, grad, hess, x0, max_iter=10)
    r_bfgs = opt.bfgs_method(f, grad, x0, max_iter=300)
    r_lbfgs = opt.lbfgs_method(f, grad, x0, m=10, max_iter=300)
    return {"gd": r_gd, "newton": r_newton, "bfgs": r_bfgs, "lbfgs": r_lbfgs}


def lbfgs_vs_bfgs_condition_sweep(kappas=(5, 20, 100, 500), dim=10, m=10, seed=7,
                                  max_iter=600) -> list:
    """Ehrlicher Befund: BFGS bleibt bei jeder Konditionszahl bei einer aehnlichen, niedrigen
    Iterationszahl, L-BFGS braucht mit wachsender Konditionszahl deutlich mehr Schritte - der
    begrenzte Speicher hat einen echten, messbaren Preis."""
    rows = []
    for kappa in kappas:
        A = fn.random_spd_matrix(dim, kappa, seed)
        f, grad, hess = fn.quadratic(A)
        rng = np.random.default_rng(seed + 1)
        x0 = rng.normal(size=dim)
        r_bfgs = opt.bfgs_method(f, grad, x0, max_iter=max_iter)
        r_lbfgs = opt.lbfgs_method(f, grad, x0, m=m, max_iter=max_iter)
        rows.append({"kappa": kappa, "bfgs_iters": r_bfgs.n_iter, "lbfgs_iters": r_lbfgs.n_iter,
                    "bfgs_final": r_bfgs.fvals[-1], "lbfgs_final": r_lbfgs.fvals[-1]})
    return rows


def rosenbrock_bfgs_vs_lbfgs(m=10, max_iter_bfgs=200, max_iter_lbfgs=1000) -> dict:
    f, grad, hess = fn.rosenbrock()
    x0 = np.array([-1.2, 1.0])
    r_bfgs = opt.bfgs_method(f, grad, x0, max_iter=max_iter_bfgs)
    r_lbfgs = opt.lbfgs_method(f, grad, x0, m=m, max_iter=max_iter_lbfgs)
    return {"bfgs": r_bfgs, "lbfgs": r_lbfgs}


def gradient_check(eps: float = 1e-6) -> dict:
    rng = np.random.default_rng(0)
    A = fn.random_spd_matrix(6, 10.0, seed=1)
    f, grad, hess = fn.quadratic(A)
    x = rng.normal(size=6)
    analytic = grad(x)
    numeric = np.zeros(6)
    for i in range(6):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    quad_err = float(np.max(np.abs(analytic - numeric) / np.maximum(np.abs(analytic), 1e-8)))

    f2, grad2, hess2 = fn.rosenbrock()
    x2 = np.array([0.3, -0.7])
    analytic2 = grad2(x2)
    numeric2 = np.zeros(2)
    for i in range(2):
        xp, xm = x2.copy(), x2.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric2[i] = (f2(xp) - f2(xm)) / (2 * eps)
    rosen_err = float(np.max(np.abs(analytic2 - numeric2) / np.maximum(np.abs(analytic2), 1e-8)))
    return {"quadratic_grad_max_rel_err": quad_err, "rosenbrock_grad_max_rel_err": rosen_err}
