import matplotlib.pyplot as plt
import numpy as np
from instance import TestInstances
from functions import StrengthFunction

strength = StrengthFunction(fun_type=3)
w = np.array([-0.0706009, 3.98564443, 1.72315245, 0.01222215, 0.09795251, -0.25704887, -0.27046902, -0.02782136])
alpha = .3

for i in range(1, 6):
	# is valid uses eval split, false uses the test set
	test_instances = TestInstances(test_rumors=[i], split=(.6, .2, .2), is_valid=False)
	# test_instances = TestInstances(test_rumors=[1, 2, 3, 4, 5], split=(.6, .2, .2), is_valid=True)
	print(f'Graph {i}:')

	_, _, _, _, accuracy = test_instances.predict_top_k(strength, w, alpha, k=10)
	print(f'\tAccuracy: {accuracy}')

	auc, _, _, _ = test_instances.predict_auc_roc(strength, w, alpha)
	print(f'\tAUC: {auc}')