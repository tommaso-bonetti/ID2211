import numpy as np
from datetime import datetime
import json

from functions import CostFunction, StrengthFunction
from graph import GraphData
from instance import Instances
from learner import ParameterLearner

def load_user_graph(rumor_number):
	g = GraphData(rumor_number)
	g.fetch_static_features(load_from_memory=False)

def main():
	strength = StrengthFunction(fun_type=3)
	cost = CostFunction(fun_type=1)
	alpha = # TODO set alpha value Tommaso 0.1 - August 0.2 - Hongjun 0.3 - Donglin 0.4

	# Both train_rumors and test_rumors need to be lists where the rumor number(s) are specified. The train_split is
	# applied to all graphs. If you want to train and test on the same rumor, that is possible. This constructor
	# instantiates both training and test graphs, thus there is no need to load anything else for testing,
	# just call the predict() function (see below).
	instances = Instances(train_rumors=[1, 2, 3, 4, 5], train_split=.6)
	learner = ParameterLearner(strength, cost, alpha=alpha)
	initial_w, learned_w, final_f = learner.learn(instances)
	print(f'The initial parameters were: {initial_w}')
	print(f'The learned parameters are: {learned_w}')
	print(f'The final value of the cost function is: {final_f}')

if __name__ == '__main__':
	main()