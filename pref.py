import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched by preferential attatchment
#TODO, handel non connected nodes
#TODO, should nodes be able to attatch to other nodes in the "nodes" list?
#TODO, give self connected nodes a weight?
#TODO, add 1 base weight
def pref_add_nodes(adjacency, toAdd):
    #new_graph = graph
    #adjacency = graph.get_adj_matrix()
    nodelen = len(adjacency)
    extended = np.vstack([adjacency, [0]*len(adjacency)])
    extended = np.column_stack([extended, [0]*len(extended)])

    for i in (range(toAdd)):
        #select node
        #TODO, check
        weights = sum(adjacency, axsis = 1)/np.sum(adjacency)
        connection = np.random.choice(nodelen+i, 1, weights[:nodelen+i])
        adjacency[connection][i+nodelen] = 1
        #nodelen += 1
    return extended