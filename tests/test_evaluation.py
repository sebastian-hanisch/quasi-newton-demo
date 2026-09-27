"""Korrektheits-/Verhaltenstests mit billigen Parametern. Die offiziellen, teuren Sweep-Werte
stehen mit Toleranzband in test_claims.py (Modul-Fixtures, je Sweep nur einmal berechnet)."""
import qn_evaluation as ev


def test_reduction_to_newton_check_is_near_machine_precision():
    assert ev.reduction_to_newton_check() < 1e-10


def test_finite_termination_check_runs_with_small_dims():
    rows = ev.finite_termination_check(dims=(2, 5))
    for row in rows:
        assert row["within_bound"]


def test_memory_footprint_bfgs_grows_quadratically():
    rows = ev.memory_footprint_comparison(ns=(10, 100))
    assert rows[1]["bfgs_numbers"] == rows[0]["bfgs_numbers"] * 100


def test_analyse_returns_valid_result_for_each_combination():
    for func in ("quadratic", "rosenbrock"):
        for optimizer in ("bfgs", "lbfgs", "newton", "gd"):
            settings = ev.Settings(func=func, optimizer=optimizer, max_iter=20, seed=0)
            out = ev.analyse(settings)
            assert out["result"].fvals[0] >= 0.0
