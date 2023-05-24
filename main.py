import numpy as np

from functions import CostFunction, StrengthFunction
from graph import GraphData
from instance import Instances
from learner import ParameterLearner

def load_user_graph(rumor_number):
	g = GraphData(rumor_number)
	g.fetch_static_features(load_from_memory=False)

def main():

	print("Starting code")

	rumor_number = 2
	load_user_graph(rumor_number)
	"""
	rumor_number = 1
	strength = StrengthFunction(fun_type=3)
	cost = CostFunction(fun_type=3)
	alpha = .25
	instances = Instances(rumor_number, sizes=list(range(50, 500, 10)))
	learner = ParameterLearner(strength, cost, alpha=alpha)
	initial_w, learned_w, final_f = learner.learn(instances)
	print(f'The initial parameters were: {initial_w}')
	print(f'The learned parameters are: {learned_w}')
	print(f'The final value of the cost function is: {final_f}')

	test_instances = Instances(rumor_number, sizes=list(range(51, 501, 10)))
	true, pred = test_instances.predict(strength, learned_w, alpha)
	true, pred = np.array(true), np.array(pred)
	print(f'True links: {true}')
	print(f'Predicted links: {pred}')
	print(f'Accuracy: {np.count_nonzero(pred[pred == true]) / np.size(true)}')
	"""

if __name__ == '__main__':
	main()