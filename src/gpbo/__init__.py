"""Gaussian Process regression and Bayesian optimization, from scratch."""

from gpbo.kernels import RBFKernel
from gpbo.gp import GaussianProcess
from gpbo.acquisition import expected_improvement
from gpbo.optimizer import BayesianOptimizer, OptimizationResult
from gpbo.model_selection import (
    TuningResult,
    build_cv_objective,
    decode_parameters,
    tune_model,
)

__all__ = [
    "RBFKernel",
    "GaussianProcess",
    "expected_improvement",
    "BayesianOptimizer",
    "OptimizationResult",
    "TuningResult",
    "build_cv_objective",
    "decode_parameters",
    "tune_model",
]
