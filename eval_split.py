import numpy as np
from instance import TestInstances
from functions import StrengthFunction

strength = StrengthFunction(fun_type=3)
w = np.array([-0.0706009, 3.98564443, 1.72315245, 0.01222215, 0.09795251, -0.25704887, -0.27046902, -0.02782136])
alpha = .1

hubs = {
	1: [18, 446, 476, 5, 170],
	2: [491, 181, 13, 270, 309, 172, 134, 603, 957],
	3: [22, 21, 214, 270, 446, 0],
	4: [0, 632],
	5: [22, 210, 996, 189, 191, 140]
}

for num in (1, 2):
	pred_attractor = []
	pred_self = []
	pred_other = []

	for i in range(1, 6):
		test_instances = TestInstances(test_rumors=[i], split=(.6, .2, .2), is_valid=False)
		sources, true, pred, _, _ = test_instances.predict_top_k(strength, w, alpha, k=num)

		for s in sources:
			d = true[s]
			p = pred[s].keys()

			if d == s:
				if d in p:
					pred_self.append(1)
				else:
					pred_self.append(0)
			elif d in hubs[i]:
				if d in p:
					pred_attractor.append(1)
				else:
					pred_attractor.append(0)
			else:
				if d in p:
					pred_other.append(1)
				else:
					pred_other.append(0)

	print(f'+++ TOP {num} +++')
	print(f'Recall on attractors: {sum(pred_attractor) / len(pred_attractor)}')
	print(f'Recall on self: {sum(pred_self) / len(pred_self)}')
	print(f'Recall on others: {sum(pred_other) / len(pred_other)}')