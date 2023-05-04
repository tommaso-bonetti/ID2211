import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes current graph and list of nodes to add
#returns a graph with these nodes attatched
#TODO, handel non connected nodes
#TODO, should nodes be able to attatch to other nodes in the "nodes" list?
def random_add_nodes(graph, nodes):
    new_graph = graph
    for node in nodes:
        #attatch to new graph
        print("temp")
    return new_graph


