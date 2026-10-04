"""Unabhaengige Orakel-Tests (anderer Rechenweg als der Code):
- BFGS: Referenz mit dem ausgeschriebenen Lehrbuch-Update H += (sy + y'Hy)/(sy)^2 ss' - (Hys' + sy'H)/sy
  statt der Sherman-Morrison-Produktform im Code.
- L-BFGS: Referenz mit expliziter Matrix (Update der gespeicherten Paare auf gamma*I) statt der
  Zwei-Schleifen-Rekursion.
- BFGS mit exakter Liniensuche und H0=I erzeugt auf einer Quadratik dieselben Iterierten wie das
  lineare CG-Verfahren (Nazareth 1979) - Referenz-CG hier von Hand geschrieben.
- Rosenbrock-Minimum gegen scipy.optimize.minimize (BFGS, L-BFGS-B); Gradientenabstieg-Schrittzahl
  aus der Eigenzerlegung."""
import numpy as np
import pytest

import qn_evaluation as ev
import qn_functions as fn
import qn_optimizer as opt

scipy_opt = pytest.importorskip("scipy.optimize")


def _armijo(f, x, d, fx, g, alpha=1e-4, beta=0.5):
    t = 1.0
    for _ in range(60):
        if f(x + t * d) <= fx + alpha * t * (g @ d):
            return t
        t *= beta
    return t


def _ref_bfgs(f, grad, x0, max_iter, A=None, tol=1e-10):
    x, H, g = x0.copy(), np.eye(len(x0)), grad(x0)
    path = [x.copy()]
    for _ in range(max_iter):
        if g @ g < tol ** 2:
            break
        d = -H @ g
        t = -(g @ d) / (d @ A @ d) if A is not None else _armijo(f, x, d, f(x), g)
        xn = x + t * d
        gn = grad(xn)
        s, y = xn - x, gn - g
        sy = s @ y
        if sy > 1e-12:
            Hy = H @ y
            H = H + (sy + y @ Hy) / sy ** 2 * np.outer(s, s) - (np.outer(Hy, s) + np.outer(s, Hy)) / sy
        x, g = xn, gn
        path.append(x.copy())
    return np.array(path)


def _ref_lbfgs(f, grad, x0, m, max_iter, A=None, tol=1e-10):
    x, g = x0.copy(), grad(x0)
    n = len(x0)
    path, pairs = [x.copy()], []
    for _ in range(max_iter):
        if g @ g < tol ** 2:
            break
        gamma = (pairs[-1][0] @ pairs[-1][1]) / (pairs[-1][1] @ pairs[-1][1]) if pairs else 1.0
        H = gamma * np.eye(n)
        for s, y in pairs:
            rho, I = 1 / (s @ y), np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        d = -H @ g
        t = -(g @ d) / (d @ A @ d) if A is not None else _armijo(f, x, d, f(x), g)
        xn = x + t * d
        gn = grad(xn)
        s, y = xn - x, gn - g
        if s @ y > 1e-12:
            pairs = (pairs + [(s, y)])[-m:]
        x, g = xn, gn
        path.append(x.copy())
    return np.array(path)


def _cg(A, x0, tol=1e-10):
    x, r = x0.copy(), A @ x0
    p, path = -r, [x.copy()]
    for _ in range(200):
        if r @ r < tol ** 2:
            break
        a = (r @ r) / (p @ A @ p)
        x = x + a * p
        rn = r + a * (A @ p)
        p = -rn + (rn @ rn) / (r @ r) * p
        r = rn
        path.append(x.copy())
    return np.array(path)


def _quadratics(n, seed=21):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        d = int(rng.integers(2, 8))
        kappa = float(np.exp(rng.uniform(np.log(1.5), np.log(100))))
        yield fn.random_spd_matrix(d, kappa, int(rng.integers(0, 10**6))), rng.normal(size=d), rng


def _close(a, b, k):
    m = min(len(a), len(b), k)
    np.testing.assert_allclose(a[:m], b[:m], rtol=1e-5, atol=1e-7)


def test_bfgs_matches_textbook_update_reference():
    for A, x0, _ in _quadratics(40):
        f, g, _h = fn.quadratic(A)
        for mode, AA in (("exact", A), ("backtracking", None)):
            r = opt.bfgs_method(f, g, x0, mode=mode, A=AA, max_iter=40)
            _close(r.trajectory, _ref_bfgs(f, g, x0, 40, A=AA), 12)
    f, g, _h = fn.rosenbrock()
    for x0 in ([-1.2, 1.0], [0.5, -0.5], [1.5, 2.0]):
        x0 = np.array(x0)
        _close(opt.bfgs_method(f, g, x0, max_iter=300).trajectory, _ref_bfgs(f, g, x0, 300), 25)


def test_lbfgs_two_loop_matches_explicit_matrix_reference():
    for A, x0, rng in _quadratics(40):
        f, g, _h = fn.quadratic(A)
        m = int(rng.integers(1, 6))
        for mode, AA in (("exact", A), ("backtracking", None)):
            r = opt.lbfgs_method(f, g, x0, m=m, mode=mode, A=AA, max_iter=40)
            _close(r.trajectory, _ref_lbfgs(f, g, x0, m, 40, A=AA), 12)
    f, g, _h = fn.rosenbrock()
    for m in (1, 3, 10):
        x0 = np.array([-1.2, 1.0])
        _close(opt.lbfgs_method(f, g, x0, m=m, max_iter=300).trajectory,
               _ref_lbfgs(f, g, x0, m, 300), 25)


def test_bfgs_with_exact_line_search_reproduces_linear_cg():
    rng = np.random.default_rng(5)
    for _ in range(40):
        d = int(rng.integers(2, 9))
        A = fn.random_spd_matrix(d, float(np.exp(rng.uniform(np.log(1.5), np.log(100)))),
                                 int(rng.integers(0, 10**6)))
        f, g, _h = fn.quadratic(A)
        x0 = rng.normal(size=d)
        r = opt.bfgs_method(f, g, x0, mode="exact", A=A, max_iter=d + 10)
        assert r.converged and r.n_iter <= d + 2
        ref = _cg(A, x0)
        np.testing.assert_allclose(r.trajectory[:d], ref[:d], atol=1e-6 * np.linalg.norm(x0))


def test_rosenbrock_minimum_matches_scipy():
    f, g, _h = fn.rosenbrock()
    rng = np.random.default_rng(8)
    for x0 in rng.uniform(-2, 2, size=(8, 2)):
        r_b = opt.bfgs_method(f, g, x0, max_iter=500)
        r_l = opt.lbfgs_method(f, g, x0, m=10, max_iter=3000)
        sol = scipy_opt.minimize(scipy_opt.rosen, x0, jac=scipy_opt.rosen_der, method="BFGS",
                                 options=dict(gtol=1e-10))
        for r in (r_b, r_l):
            assert r.converged
            np.testing.assert_allclose(r.trajectory[-1], sol.x, atol=1e-4)


def test_four_way_gradient_descent_step_count_matches_eigen_closed_form():
    out = ev.four_way_comparison()
    A = fn.random_spd_matrix(10, 100.0, 5)
    x0 = np.random.default_rng(6).normal(size=10)
    w, V = np.linalg.eigh(A)
    c0, eta = V.T @ x0, 1.0 / w[-1]
    k_star = next(k for k in range(100000)
                  if np.linalg.norm(w * ((1 - eta * w) ** k * c0)) < 1e-10)
    assert out["gd"].n_iter == k_star
    assert out["newton"].n_iter == 1
    assert out["bfgs"].n_iter < 30 < out["lbfgs"].n_iter


def test_bfgs_with_exact_inverse_hessian_is_newton_step():
    assert ev.reduction_to_newton_check() < 1e-12
