import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched randomly
#TODO, handel non connected nodes
#TODO, should nodes be able to attatch to other nodes in the "nodes" list?
def random_add_nodes(graph, nodes):
    new_graph = graph
    for node in nodes:
        #pic a random number up to len of graph node lsit
        graphlen = 0 #if it is original graph or the new_graph decides if new nodes can be connected
        nodeslen = 0 #either 1 or do all nodeslen before for loop
        np.random.choice(graphlen, nodeslen)  
        
        #attatch to new graph
        print("temp")
    return new_graph

