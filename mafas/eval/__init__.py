"""MAFAS evaluation harness for RAG, simulation, risk, analyst, gates, and strategy."""

from eval.runner import run_evaluation
from eval.schemas import EvaluationReport, SuiteResult

__all__ = ["EvaluationReport", "SuiteResult", "run_evaluation"]
