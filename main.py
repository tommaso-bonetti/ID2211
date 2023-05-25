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
	alpha = .2

	# Both train_rumors and test_rumors need to be lists where the rumor number(s) are specified. The train_split is
	# applied to all graphs. If you want to train and test on the same rumor, that is possible. This constructor
	# instantiates both training and test graphs, thus there is no need to load anything else for testing,
	# just call the predict() function (see below).
	instances = Instances(train_rumors=[1, 2], test_rumors=[5], train_split=.8)
	learner = ParameterLearner(strength, cost, alpha=alpha)
	initial_w, learned_w, final_f = learner.learn(instances)
	print(f'The initial parameters were: {initial_w}')
	print(f'The learned parameters are: {learned_w}')
	print(f'The final value of the cost function is: {final_f}')

	# TODO: print to file the indices of training and test set and the learned parameters learned_w.
	training_indices = instances.train_idx  # Dictionary in the form { rumor_number: list_of_indices }
	test_indices = instances.test_idx  # Dictionary in the form { rumor_number: list_of_indices }

	now = datetime.now()
	current_time = now.strftime("%H_%M_%S")
	
	f_w = open("datafile_"+current_time+"_learned_w.txt", "a")
	f_w.write(learned_w)
	f_w.close()

	f_training = open("datafile_"+current_time+"_training_indices.txt", "a")
	f_training.write(json.dumps(training_indices))
	f_training.close()

	f_test = open("datafile_"+current_time+"_test_indices.txt", "a")
	f_test.write(json.dumps(test_indices))
	f_test.close()

	print(f'Train: {training_indices}')
	print(f'Test: {test_indices}')

	true, pred = instances.predict(strength, learned_w, alpha)
	true, pred = np.array(true), np.array(pred)
	print(f'True links: {true}')
	print(f'Predicted links: {pred}')
	print(f'Accuracy: {np.count_nonzero(pred[pred == true]) / np.size(true)}')

if __name__ == '__main__':
	main()