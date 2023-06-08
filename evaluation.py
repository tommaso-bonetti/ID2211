import matplotlib.pyplot as plt
import numpy as np
from instance import TestInstances
from functions import StrengthFunction

strength = StrengthFunction(fun_type=3)

w = [0]*4
alpha = [0]*4


w[0] = np.array([-0.0706009,   3.98564443,  1.72315245,  0.01222215,  0.09795251, -0.25704887, -0.27046902, -0.02782136])
alpha[0] = .1


w[1] = np.array([0.75985795, 3.05375427, 0.01756726, 0.94059403, -0.1289283, -0.13990088, 0.13292095, -0.18505893])#([-0.07572922, -1.42684673, 3.78953302, 0.01810841, -0.10976192, 0.15850626, -0.05775282, -0.10935978])
alpha[1] = .2

w[2] = np.array([-0.01011709, 0.06906795, 1.04405384, 4.7438058, 0.17770004, -1.40017573, -0.22598882, 0.11433051])
alpha[2] = .3

w[3] = np.array([0.96202257,  2.67914466,  0.46315231,  1.22481268,  0.02598581,  0.03739628, -0.10810647, -0.2294527])
alpha[3] = .4


#is valid uses eval split, false uses the test set
test_instances = TestInstances(test_rumors=[1, 2, 3, 4, 5], split=(.6, .2, .2), is_valid=False) 
#test_instances = TestInstances(test_rumors=[1, 2, 3, 4, 5], split=(.6, .2, .2), is_valid=True) 

print("done reading data, running test...\n\n")

for i in range(0,4):
    print("for alpha ", alpha[i])
    true, pred, _, accuracy = test_instances.predict_top_k(strength, w[i], alpha[i], k=10)
    #print(f'True links: {true}')
    #print(f'Predicted links: {pred}')
    print(f'Accuracy: {accuracy}')

    auc, fpr, tpr, th = test_instances.predict_auc_roc(strength, w[i], alpha[i])
    print(f'AUC: {auc}')
    print("\n\n")
    #plt.plot(fpr, tpr)
    #plt.show()