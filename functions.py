import numpy as np

class StrengthFunction:
	compute_strength = None
	compute_gradient = None

	def __init__(self, strength, gradient):
		self.compute_strength = strength
		self.compute_gradient = gradient

class CostFunction:
	compute_cost = None
	compute_gradient = None
	compute_cost_and_gradient = None

	def __init__(self, cost, gradient):
		self.compute_cost = cost
		self.compute_gradient = gradient