import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched randomly

def random_add_nodes(adjacency, toAdd):

    nodeslen = len(adjacency)

    #special case for 0
    if(len(adjacency) == 0):
        adjacency = [[1]]
        toAdd = toAdd - 1

    extended = [[0]*(nodeslen+toAdd)]*(nodeslen+toAdd)
    extended = np.array(extended)
    
    for i in range(nodeslen):
        for j in range(nodeslen):
            extended[i][j] = adjacency[i][j]              
    
    for i in (range(toAdd)):
        #pic a random number up to len of graph node lsit
        #if it is original graph or the new_graph decides if new nodes can be connected
        connection = np.random.choice((nodeslen+i),1)[0] 
        extended[i+nodeslen][connection] = 1
    return extended

#testfunction
def main():
    adjacency = random_add_nodes([[]], 1000)
    print(np.sum(adjacency))
    print(max(np.sum(adjacency, axis=0)))
    print(max(np.sum(adjacency, axis=1)))


if __name__ == '__main__':
    main()
