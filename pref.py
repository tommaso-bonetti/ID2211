import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched by preferential attatchment
def pref_add_nodes(adjacency, toAdd):
    
    nodeslen = len(adjacency)

    #special case for 0
    if(len(adjacency) == 0):
        adjacency = [[1]]
        toAdd = toAdd - 1

    extended = [[0]*(nodeslen+toAdd)]*(nodeslen+toAdd)
    extended = np.array(extended)
    
    weights = np.array(range(nodeslen))
    incomming = np.sum(adjacency, axis=0)
    for i in range(nodeslen):
        while incomming[i] > 0:
            np.append(weights,i)
            incomming[i] = incomming[i] - 1
    
    for i in range(nodeslen):
        for j in range(nodeslen):
            extended[i][j] = adjacency[i][j]

    for i in (range(toAdd)):
        #select node
        weights = np.append(weights,i)
        connection = np.random.choice(weights, 1)[0]
        weights = np.append(weights,connection)
        extended[i+nodeslen][connection] = 1

    return extended

#testfunction
def main():
    adjacency = pref_add_nodes([[0,1],[0,1]], 1000)
    #print(adjacency)
    print(np.sum(adjacency))
    print(max(np.sum(adjacency, axis=0)))
    print(max(np.sum(adjacency, axis=1)))
    
if __name__ == '__main__':
    main()
