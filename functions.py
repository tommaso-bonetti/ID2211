import numpy as np

class StrengthFunction:
	def __init__(self):
		self.type = 2

	def compute_strength(self, dot_product):
		if self.type == 1:
			return np.exp(dot_product)
		if self.type == 2:
			return np.maximum(0, dot_product + 1) ** 2
		return np.reciprocal(1 + np.exp(-dot_product))

	def compute_gradient(self, w, psi):
		if self.type == 1:
			return psi * np.exp(psi @ w)
		if self.type == 2:
			return np.maximum(0, 2 * (psi @ w + 1))
		return psi * np.exp(-psi @ w) / ((1 + np.exp(-psi @ w)) ** 2)

class CostFunction:
	def __init__(self):
		self.b = 1e-3
		self.type = 2

	def compute_cost(self, x):
		if self.type == 1:
			temp = np.exp(-x / self.b)
			return np.reciprocal(1 + temp)
		if self.type == 2:
			return np.maximum(0, x) ** 2

	def compute_gradient(self, x):
		if self.type == 1:
			temp = np.reciprocal(1 + np.exp(x / self.b))
			gradient = temp * (1 - temp) / self.b
			return gradient
		if self.type == 2:
			return np.maximum(0, 2 * x)

	def compute_cost_and_gradient(self, x):
		if self.type == 1:
			cost = np.reciprocal(1 + np.exp(-x / self.b))
			gradient = cost * (1 - cost) / self.b
			return cost, gradient