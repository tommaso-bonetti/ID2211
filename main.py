from graph import GraphData
from instance import Instances
from learner import ParameterLearner

def temp():
	g = GraphData(1)
	g.fetch_static_features(True)
	print(g.get_snapshot().get_feature('dst_in_degree'))

def main():
	rumor_number = 1
	instances = Instances(rumor_number, sizes=list(range(50, 500, 100)))
	learner = ParameterLearner(alpha=.4)
	learned_w, final_f = learner.learn(instances)
	print(f'The learned parameters are: {learned_w}')
	print(f'The final value of the cost function is: {final_f}')

if __name__ == '__main__':
	main()