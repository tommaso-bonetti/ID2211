from graph import GraphData
from instance import Instances
from learner import ParameterLearner

def temp():
	g = GraphData(1)
	g.fetch_static_features(True)
	print(g.get_snapshot().get_feature('dst_in_degree'))

def main():
	# Generates a random dataset with the specified parameters, then uses a parameter learner to learn them.

	# Generates a random dataset. If you want to change the number of instances, features or the strength function, change
	# the properties of the generator before invoking	generate.

	rumor_number = 1
	instances = Instances(rumor_number, sizes=list(range(50, 500, 50)))
	# generator = random_generators.RandomInstancesGenerator()
	# [dataset, weighter, alpha, true_w] = generator.generate()
	# Use a WMW cost function, a time limit of 30 seconds and progress printing
	learner = ParameterLearner()
	learned_w, final_f = learner.learn(instances)
	# print(['The true parameters are: ', num2str(true_w)])
	print(f'The learned parameters are: {learned_w}')
	print(f'The final value of the cost function is: {final_f}')

if __name__ == '__main__':
	main()