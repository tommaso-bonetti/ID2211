import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched randomly
#TODO, handel non connected nodes (add node as option to choose from and interpret self connection as non connection)
#TODO, should nodes be able to attatch to other nodes in the "nodes" list? (recalculate length)
def random_add_nodes(adjacency, toAdd):
    #new_graph = graph

    nodeslen = len(adjacency)
    extended = np.vstack([adjacency, [0]*len(adjacency)])
    extended = np.column_stack([extended, [0]*len(extended)])
    
    for i in (range(toAdd)):
        #pic a random number up to len of graph node lsit
        #if it is original graph or the new_graph decides if new nodes can be connected
        connection = np.random.choice(nodeslen+i) 
        extended[connection][i+nodeslen] = 1

    return extended


