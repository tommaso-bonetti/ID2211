from typing import Union

import numpy as np
from numpy import ndarray

from functions import StrengthFunction, CostFunction
from instance import Instances
from scipy.optimize import dual_annealing, minimize

class ParameterLearner:

	def __init__(self, strength: StrengthFunction, cost: CostFunction, alpha: float = 0.3, time_limit: int = 30):
		self.strength_fun = strength
		self.cost_fun = cost
		self.alpha = alpha
		self.time_limit = time_limit
		self.print_progress = True

	def learn(self, instances: Instances):
		nf = instances.num_features()
		# w0 = np.full(nf, 0.001)
		w0 = np.random.normal(0, 1e-3, nf)

		def objective_function(x: ndarray) -> float:
			return instances.compute_cost(self.strength_fun, x, self.cost_fun, self.alpha)

		def grad(x: ndarray) -> ndarray:
			return instances.compute_grad(self.strength_fun, x, self.cost_fun, self.alpha)

		def objective_function_and_grad(x: ndarray) -> tuple[float, ndarray]:
			return instances.compute_cost_and_grad(self.strength_fun, x, self.cost_fun, self.alpha)

		print('Starting optimization...')

		bounds = [(-5, 5) for _ in range(nf)]
		res = dual_annealing(objective_function, x0=w0, bounds=bounds, maxiter=100, minimizer_kwargs={'jac': grad})
		return w0, res.x, res.fun

		options = {
			'maxiter': 100,
			# 'maxls': 40,
			'disp': self.print_progress,
			'ftol': 1e-8,
			'gtol': 1e-8,
		}
		# TODO strict bounds: maybe gradient needs to be normalized?
		res = minimize(objective_function, x0=w0, jac=True, bounds=bounds, method='L-BFGS-B', options=options)
		return w0, res.x, res.fun