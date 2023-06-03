import numpy as np
from scipy.sparse import csr_array

class StrengthFunction:
	def __init__(self, fun_type: int = 1):
		"""
		Constructs an edge strength function.
		:param fun_type: 1 for exponential strength, 2 for squared hinge strength, 3 for logistic strength.
		"""
		self.type = fun_type

	def compute_strength(self, dot_product: csr_array) -> csr_array:
		res = dot_product.copy()

		if self.type == 1:
			res.data = np.nan_to_num(np.exp(res.data))

		elif self.type == 2:
			res.data = np.maximum(0, res.data) ** 2

		elif self.type == 3:
			with np.errstate(over='ignore'):
				res.data = np.reciprocal(1 + np.exp(-res.data))

		return res

	def compute_gradient(self, w: float, psi: csr_array) -> csr_array:
		res = psi * w

		if self.type == 1:
			res.data = np.nan_to_num(np.exp(res.data))

		elif self.type == 2:
			res.data = np.maximum(0, 2 * res.data)

		elif self.type == 3:
			with np.errstate(over='ignore'):
				exp = np.nan_to_num(np.exp(-res.data))
				res.data = exp / ((1 + exp) ** 2)

		return psi * res

class CostFunction:
	def __init__(self, fun_type: int = 1):
		"""
		Constructs a cost function.
		:param fun_type: 1 for step loss, 2 for squared hinge loss, 3 for step + squared hinge.
		"""
		self.b = 1e-6
		self.type = fun_type

	def compute_cost(self, x):
		with np.errstate(over='ignore'):
			step = np.reciprocal(1 + np.exp(-x / self.b))
		sq_hinge = np.maximum(0, x) ** 2

		if self.type == 1:
			return step
		if self.type == 2:
			return sq_hinge
		if self.type == 3:
			return step + sq_hinge

	def compute_gradient(self, x):
		temp = np.reciprocal(1 + np.exp(x / self.b))
		step = temp * (1 - temp) / self.b
		sq_hinge = np.maximum(0, 2 * x)

		if self.type == 1:
			return step
		if self.type == 2:
			return sq_hinge
		if self.type == 3:
			return step + sq_hinge

	def compute_cost_and_gradient(self, x):
		with np.errstate(over='ignore'):
			cost_step = np.reciprocal(1 + np.exp(-x / self.b))
		grad_step = cost_step * (1 - cost_step) / self.b

		cost_hinge = np.maximum(0, x) ** 2
		grad_hinge = np.maximum(0, 2 * x)
		if self.type == 1:
			return cost_step, grad_step
		if self.type == 2:
			return cost_hinge, grad_hinge
		if self.type == 3:
			return cost_step + cost_hinge, grad_step + grad_hinge