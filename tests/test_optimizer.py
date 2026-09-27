import numpy as np

import qn_functions as fn
import qn_optimizer as opt


def test_bfgs_with_exact_inverse_hessian_matches_newton():
    A = fn.random_spd_matrix(5, 20.0, seed=0)
    f, grad, hess = fn.quadratic(A)
    Ainv = np.linalg.inv(A)
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=5)
    r_bfgs = opt.bfgs_method(f, grad, x0, H0=Ainv, mode="exact", A=A, max_iter=5)
    r_newton = opt.newton_method(f, grad, hess, x0, max_iter=5)
    np.testing.assert_allclose(r_bfgs.trajectory, r_newton.trajectory, atol=1e-9)


def test_bfgs_solves_quadratic_within_roughly_n_steps():
    n = 10
    A = fn.random_spd_matrix(n, 20.0, seed=2)
    f, grad, hess = fn.quadratic(A)
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=n)
    result = opt.bfgs_method(f, grad, x0, mode="exact", A=A, max_iter=n + 5)
    assert result.n_iter <= n
    np.testing.assert_allclose(result.trajectory[-1], np.zeros(n), atol=1e-6)


def test_bfgs_converges_on_rosenbrock():
    f, grad, hess = fn.rosenbrock()
    result = opt.bfgs_method(f, grad, np.array([-1.2, 1.0]), max_iter=100)
    assert result.converged
    np.testing.assert_allclose(result.trajectory[-1], np.array([1.0, 1.0]), atol=1e-4)


def test_lbfgs_converges_on_rosenbrock_given_enough_budget():
    f, grad, hess = fn.rosenbrock()
    result = opt.lbfgs_method(f, grad, np.array([-1.2, 1.0]), m=10, max_iter=1000)
    assert result.converged
    np.testing.assert_allclose(result.trajectory[-1], np.array([1.0, 1.0]), atol=1e-4)


def test_lbfgs_uses_more_iterations_than_bfgs_on_rosenbrock():
    f, grad, hess = fn.rosenbrock()
    x0 = np.array([-1.2, 1.0])
    r_bfgs = opt.bfgs_method(f, grad, x0, max_iter=100)
    r_lbfgs = opt.lbfgs_method(f, grad, x0, m=10, max_iter=1000)
    assert r_lbfgs.n_iter > r_bfgs.n_iter


def test_gradient_descent_and_newton_copies_still_work():
    A = fn.random_spd_matrix(4, 10.0, seed=0)
    f, grad, hess = fn.quadratic(A)
    x0 = np.array([1.0, 1.0, 1.0, 1.0])
    r_gd = opt.gradient_descent(f, grad, x0, eta=0.05, max_iter=500)
    r_newton = opt.newton_method(f, grad, hess, x0, max_iter=5)
    assert r_gd.converged
    assert r_newton.n_iter == 1
