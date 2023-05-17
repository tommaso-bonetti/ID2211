from typing import Union

import numpy as np
from numpy import ndarray

from functions import StrengthFunction, CostFunction
from instance import Instances
from scipy.optimize import dual_annealing, minimize

class ParameterLearner:

	def __init__(self, alpha: float = 0.3, time_limit: int = 30, print_progress: bool = True):
		self.strengthf = StrengthFunction(fun_type=2)
		self.costf = CostFunction(fun_type=1)
		self.alpha = alpha
		self.time_limit = time_limit
		self.print_progress = print_progress

	def learn(self, instances: Instances):
		nf = instances.num_features()
		w0 = np.full(nf, 0.001)

		def objective_function(x: ndarray) -> tuple[float, ndarray]:
			return instances.compute_cost_and_grad(self.strengthf, x, self.costf, self.alpha)

		print('Starting optimization...')

		bounds = [(-10.0, 10.0) for _ in range(nf)]
		options = {
			'maxiter': 100,
			# 'maxls': 40,
			'disp': self.print_progress,
			'ftol': 1e-8,
			'gtol': 1e-8,
		}
		# TODO strict bounds: maybe gradient needs to be normalized?
		res = minimize(objective_function, x0=w0, jac=True, bounds=bounds, method='L-BFGS-B', options=options)
		return res.x, res.fun