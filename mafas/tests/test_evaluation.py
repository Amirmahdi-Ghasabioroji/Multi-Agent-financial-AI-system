"""Tests for the evaluation harness."""

from eval.runner import run_evaluation
from eval.simulation_eval import _calibration_case, _trending_df


def test_simulation_calibration_case_has_metrics():
    frame = _trending_df(300, 0.003)
    case = _calibration_case(
        "trend",
        "Trend long",
        frame,
        direction="long",
        stop_pct=0.03,
        target_pct=0.06,
        horizon=20,
    )
    names = {metric.name for metric in case.metrics}
    assert "calibration_error" in names
    assert "empirical_tp_rate" in names
    assert "mc_prob_tp_before_sl" in names


def test_run_evaluation_simulation_only():
    report = run_evaluation(suites=["simulation"])
    assert len(report.suites) == 1
    assert report.suites[0].suite == "simulation"
    assert report.suites[0].status == "completed"
    assert report.suites[0].metrics
