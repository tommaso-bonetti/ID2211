import matplotlib.pyplot as plt
import numpy as np
from instance import TestInstances
from functions import StrengthFunction

strength = StrengthFunction(fun_type=3)
w = np.array([-0.07572922, -1.42684673, 3.78953302, 0.01810841, -0.10976192, 0.15850626, -0.05775282, -0.10935978])
alpha = .3

test_instances = TestInstances(test_rumors=[1, 2, 3, 4, 5], split=(.6, .2, .2), is_valid=False)
true, pred, _, accuracy = test_instances.predict_top_k(strength, w, alpha, k=10)
print(f'True links: {true}')
print(f'Predicted links: {pred}')
print(f'Accuracy: {accuracy}')

auc, fpr, tpr, th = test_instances.predict_auc_roc(strength, w, alpha)
print(f'AUC: {auc}')
plt.plot(fpr, tpr)
plt.show()