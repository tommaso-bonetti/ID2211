import numpy as np

class StrengthFunction:
    def __init__(self):
        self.type = 1
    
    def compute_strength(self, dot_product):
        if self.type == 1:
            return np.exp(dot_product)
        else:
            return 1 / (1 + np.exp(-dot_product))
        
    def compute_gradient(self, w, psi):
        if self.type == 1:
            return psi * np.exp(psi * w)
        else:
            return psi * np.exp(-psi * w) / (1 + np.exp(-psi * w))**2
	    
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
        
    def learn(self, instances):
        nf = instances[0].GetNumFeatures()
        w0 = np.full(nf, 0.000001)
        
        def objective_function(x):
            return instances.CalcCost(self.weighter, self.alpha, self.costf, x)
        
        display = 'iter' if self.print_progress else 'off'
        
        options = snap.TSASimAnnOptions()
        options.Display = display
        options.ReannealInterval = 10
        options.ObjectiveLimit = nf + 1
        options.TimeLimit = self.time_limit
        options.TemperatureFunction = snap.TSABoltzmann
        options.Tolerance = 1e-10
        
        w = snap.TSASimAnnMinimize(objective_function, w0, -3 * np.ones(nf), 3 * np.ones(nf), options)
        
        return w
