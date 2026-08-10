"""Gaussian Process regression and Bayesian optimization, from scratch."""

from gpbo.acquisition import expected_improvement
from gpbo.gp import GaussianProcess
from gpbo.kernels import RBFKernel

__all__ = ["expected_improvement", "GaussianProcess", "RBFKernel"]
