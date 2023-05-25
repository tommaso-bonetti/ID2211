import numpy as np
from instance import Instances
from functions import StrengthFunction

strength = StrengthFunction(fun_type=3)
w = np.array([
	0.09317276, -0.25391533, 1.89842937, 2.59772972, 1.06444034, 2.64647977, -2.89376632, 0.00516021, 0.03429735,
	-0.01369599, 0.02015039, 0.13989816, -0.0264989, 0.15804016, -0.12739018, -2.96182103
])
alpha = .2

test_instances = Instances(train_rumors=[], test_rumors=[3], train_split=.8)
true, pred, _, accuracy = test_instances.predict(strength, w, alpha, k=10)
print(f'True links: {true}')
print(f'Predicted links: {pred}')
print(f'Accuracy: {accuracy}')