import numpy as np

from functions import StrengthFunction, CostFunction
from instance import Instances
from scipy.optimize import dual_annealing, minimize

class ParameterLearner:

	def __init__(self, alpha: float = 0.3, time_limit: int = 30, print_progress: bool = True):
		self.strengthf = StrengthFunction()
		self.costf = CostFunction()
		self.alpha = alpha
		self.time_limit = time_limit
		self.print_progress = print_progress

	def learn(self, instances: Instances):
		nf = instances.num_features()
		w0 = np.full(nf, 0.000001)

		def objective_function(x):
			return instances.compute_cost_and_grad(self.strengthf, x, self.costf, self.alpha)

		def print_info(xk, state):
			print(f'Iteration {state.nit}:')
			print(f'\tCost function: {state.fun}')
			print(f'\tParameter vector: {xk}')

		bounds = [(-3.0, 3.0) for _ in range(nf)]
		options = {
			'maxiter': 1000,
			'disp': self.print_progress,
			'ftol': 1e-10,
			'gtol': 1e-10
		}
		print('Starting optimization...')
		res = minimize(objective_function, x0=w0, bounds=bounds, method='L-BFGS-B', options=options, callback=print_info)
		# res = dual_annealing(
		# 			objective_function,
		# 			x0=w0,
		# 			bounds=bounds,
		# 			maxiter=100,
		# 			minimizer_kwargs={'method': 'L-BFGS-B', 'options': options})

		return res.x, res.fun