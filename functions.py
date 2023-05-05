import numpy as np

class StrengthFunction:
	def compute_strength(self):
		return

	def compute_gradient(self):
		return
class CostFunction:
    def __init__(self):
        self.b = .00001
        self.type = 1
    
    def calcCost(self,x):
        if self.type==1:
            return 1.0 /(1.0+math.exp(-x/self.b))
    
    def calcGradient(self,x):
        if self.type==1:
            tmp = 1.0 / (1+np.exp(x/self.b))
            gradient = tmp * (1-tmp) / self.b
            return gradient
    def calcCostAndGradient(self,x):
        if self.type==1:
            cost = 1.0/(1.0+np.exp(-x/self.b))
            gradient = cost * (1-cost) / self.b
            return cost, gradient

class ParameterLearner:
    def __init__(self):
        self.strengthf=StrengthFunction()
        self.costf=CostFunction()
        self.alpha=0.3
        self.time_limit=30
        self.print_progress=True
