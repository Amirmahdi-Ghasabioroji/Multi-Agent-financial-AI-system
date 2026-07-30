"""MAFAS evaluation harness for RAG, simulation, and risk metrics."""

from eval.runner import run_evaluation
from eval.schemas import EvaluationReport, SuiteResult

__all__ = ["EvaluationReport", "SuiteResult", "run_evaluation"]
