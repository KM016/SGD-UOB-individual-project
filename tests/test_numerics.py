from pathlib import Path

import numpy as np

from scripts import experiment as well
from scripts import ill_conditioned_experiment as ill


def test_toy_problem_is_reproducible() -> None:
    first = well.make_toy_least_squares(n=40, d=5, seed=17)
    second = well.make_toy_least_squares(n=40, d=5, seed=17)

    for left, right in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(left, right)
    assert first[3] == second[3]


def test_full_gradient_matches_directional_finite_difference() -> None:
    a, b, _, mu = well.make_toy_least_squares(n=80, d=6, seed=3)
    a, b = well.standardise_problem(a, b)
    rng = np.random.default_rng(9)
    x = rng.normal(size=a.shape[1])
    direction = rng.normal(size=a.shape[1])
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6

    numerical = (
        well.F_value(a, b, x + epsilon * direction, mu)
        - well.F_value(a, b, x - epsilon * direction, mu)
    ) / (2 * epsilon)
    analytic = float(well.F_grad(a, b, x, mu) @ direction)

    assert np.isclose(numerical, analytic, rtol=1e-6, atol=1e-7)


def test_closed_form_smooth_solution_is_stationary() -> None:
    a, b, _, mu = well.make_toy_least_squares(n=100, d=8, seed=5)
    a, b = well.standardise_problem(a, b)
    optimum = well.solve_smooth_star(a, b, mu)

    assert np.linalg.norm(well.F_grad(a, b, optimum, mu)) < 1e-10


def test_composite_reference_solution_has_small_gradient_mapping() -> None:
    a, b, _, mu = well.make_toy_least_squares(n=120, d=8, seed=7)
    a, b = well.standardise_problem(a, b)
    lam = 0.02
    optimum, residual = well.solve_composite_star(a, b, mu, lam, verbose=False)
    lipschitz, _ = well.estimate_L_and_m(a, mu)
    mapping = well.gradient_mapping(a, b, optimum, 0.99 / lipschitz, mu, lam)

    assert residual < 1e-8
    assert np.linalg.norm(mapping) < 1e-8


def test_svrg_estimator_has_zero_variance_at_its_snapshot() -> None:
    a, b, _, mu = well.make_toy_least_squares(n=60, d=5, seed=11)
    a, b = well.standardise_problem(a, b)
    x_snapshot = np.linspace(-0.2, 0.3, a.shape[1])
    full_gradient = well.F_grad(a, b, x_snapshot, mu)
    indices = np.arange(a.shape[0])

    variance = well.estimate_variance(
        a,
        b,
        x_snapshot,
        mu,
        indices,
        estimator="svrg",
        x_snap=x_snapshot,
        mu_grad_snap=full_gradient,
    )

    assert variance < 1e-28


def test_seeded_svrg_is_reproducible_and_reduces_the_objective_gap() -> None:
    a, b, _, mu = well.make_toy_least_squares(n=100, d=6, seed=13)
    a, b = well.standardise_problem(a, b)
    optimum = well.solve_smooth_star(a, b, mu)
    optimum_value = well.F_value(a, b, optimum, mu)
    x0 = np.zeros(a.shape[1])
    lipschitz, _ = well.estimate_L_and_m(a, mu)

    first_x, first_logs = well.run_svrg(
        a, b, x0, epochs=12, m_inner=40, alpha0=0.01 / lipschitz,
        mu=mu, x_star=optimum, obj_star=optimum_value, seed=23,
        variance_sample_size=20,
    )
    second_x, second_logs = well.run_svrg(
        a, b, x0, epochs=12, m_inner=40, alpha0=0.01 / lipschitz,
        mu=mu, x_star=optimum, obj_star=optimum_value, seed=23,
        variance_sample_size=20,
    )

    np.testing.assert_array_equal(first_x, second_x)
    assert first_logs == second_logs
    assert first_logs[-1]["obj_gap"] < first_logs[0]["obj_gap"]


def test_ill_conditioned_generator_reaches_target_scale() -> None:
    a, _, _, mu = ill.make_ill_conditioned_least_squares(
        n=300, d=12, mu_ridge=1e-4, seed=29, kappa_target=1e4
    )
    lipschitz, strong_convexity = ill.estimate_L_and_m(a, mu)
    condition_number = lipschitz / strong_convexity

    assert np.isclose(lipschitz, 1.0, rtol=1e-10, atol=1e-10)
    assert 4_000 < condition_number < 6_000


def test_log_writer_creates_missing_parent_directories(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "logs.json"
    well.dump_logs_json(destination, [[{"objective": 1.0}]])

    assert destination.is_file()
    assert '"objective": 1.0' in destination.read_text()
