import numpy as np

import qn_functions as fn


def test_random_spd_matrix_has_prescribed_condition_number():
    A = fn.random_spd_matrix(8, 50.0, seed=1)
    eigs = np.linalg.eigvalsh(A)
    kappa = eigs.max() / eigs.min()
    assert abs(kappa - 50.0) < 1e-6


def test_quadratic_minimum_is_zero_at_origin():
    A = fn.random_spd_matrix(4, 10.0, seed=0)
    f, grad, hess = fn.quadratic(A)
    assert f(np.zeros(4)) == 0.0
    np.testing.assert_allclose(grad(np.zeros(4)), np.zeros(4))


def test_rosenbrock_minimum_is_zero_at_one_one():
    f, grad, hess = fn.rosenbrock()
    x_star = np.array([1.0, 1.0])
    assert f(x_star) == 0.0
    np.testing.assert_allclose(grad(x_star), np.zeros(2), atol=1e-10)


def test_rosenbrock_gradient_matches_finite_differences():
    f, grad, hess = fn.rosenbrock()
    x = np.array([0.5, 0.5])
    eps = 1e-6
    numeric = np.zeros(2)
    for i in range(2):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        numeric[i] = (f(xp) - f(xm)) / (2 * eps)
    np.testing.assert_allclose(grad(x), numeric, rtol=1e-4)
