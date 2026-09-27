"""Jede Zahl aus README.md und App wird hier aus den echten Auswertungsfunktionen neu berechnet.
Modul-Fixtures berechnen jeden Sweep nur einmal; Toleranzband, wo numerische Details
(Rundungsfehler bei großem n) selbst der Befund sind."""
import pytest

import qn_evaluation as ev


@pytest.fixture(scope="module")
def finite_termination_rows():
    return ev.finite_termination_check()


@pytest.fixture(scope="module")
def condition_sweep_rows():
    return ev.lbfgs_vs_bfgs_condition_sweep()


@pytest.fixture(scope="module")
def four_way():
    return ev.four_way_comparison()


@pytest.fixture(scope="module")
def rosenbrock_comparison():
    return ev.rosenbrock_bfgs_vs_lbfgs()


def test_claim_reduction_to_newton_is_near_machine_precision():
    assert ev.reduction_to_newton_check() < 1e-10


def test_claim_finite_termination_holds_exactly_through_n_20(finite_termination_rows):
    for row in finite_termination_rows:
        if row["n"] <= 20:
            assert row["within_bound"]


def test_claim_finite_termination_overshoot_stays_bounded_above_n_20(finite_termination_rows):
    """Ehrlicher Befund: oberhalb n=20 kostet Rundungsfehler zusaetzliche Schritte, aber nie
    mehr als ca. 20% ueber der theoretischen Schranke n."""
    for row in finite_termination_rows:
        if row["n"] > 20:
            assert row["n_iter"] <= row["n"] * 1.25


def test_claim_memory_footprint_ratio_grows_with_n():
    rows = ev.memory_footprint_comparison()
    ratios = [r["ratio"] for r in rows]
    assert ratios == sorted(ratios)
    assert ratios[-1] > 100


def test_claim_bfgs_beats_gradient_descent_and_approaches_newton(four_way):
    assert four_way["bfgs"].n_iter < four_way["gd"].n_iter
    assert four_way["newton"].n_iter < four_way["bfgs"].n_iter < four_way["gd"].n_iter / 10


def test_claim_bfgs_iterations_stay_flat_across_condition_numbers(condition_sweep_rows):
    bfgs_iters = [r["bfgs_iters"] for r in condition_sweep_rows]
    assert max(bfgs_iters) < 30


def test_claim_lbfgs_iterations_grow_with_condition_number(condition_sweep_rows):
    """Ehrlicher Befund: L-BFGS haelt bei niedriger Konditionszahl mit BFGS mit, braucht aber bei
    hoher Konditionszahl deutlich mehr Schritte - der begrenzte Speicher hat einen echten Preis."""
    lbfgs_iters = [r["lbfgs_iters"] for r in condition_sweep_rows]
    assert lbfgs_iters == sorted(lbfgs_iters)
    assert lbfgs_iters[-1] > lbfgs_iters[0] * 10


def test_claim_lbfgs_needs_far_more_iterations_on_rosenbrock(rosenbrock_comparison):
    assert rosenbrock_comparison["lbfgs"].n_iter > rosenbrock_comparison["bfgs"].n_iter * 10


def test_claim_gradient_check_below_1e_minus_6():
    out = ev.gradient_check()
    assert out["quadratic_grad_max_rel_err"] < 1e-6
    assert out["rosenbrock_grad_max_rel_err"] < 1e-6
