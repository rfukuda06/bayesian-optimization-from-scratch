"""Gaussian Process regression and Bayesian optimization, from scratch."""

from gpbo.kernels import RBFKernel
from gpbo.gp import GaussianProcess
from gpbo.acquisition import expected_improvement
from gpbo.optimizer import BayesianOptimizer, OptimizationResult

__all__ = [
    "RBFKernel",
    "GaussianProcess",
    "expected_improvement",
    "BayesianOptimizer",
    "OptimizationResult",
]
