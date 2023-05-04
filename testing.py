import re
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

#takes graph and a batchsize
#predicts edges to new batchsized blocks 
#returns list woth correct predicitons of each batch
def correct_edges(graph, batchsize):

    #creat subgraph of first batchsize nodes
    #creat list of next batchsize of nodes
    #call prediction functions
    #check for correct edges
    #repeat untill all nodes in graph have been added 

    return correct_ml, correct_pref, correct_rand

def proportion_batchwise(graph, batchsize):
    correct_ml, correct_pref, correct_rand = correct_edges(graph, batchsize)
    return correct_ml/batchsize, correct_pref/batchsize, correct_rand/batchsize

def proportion_cumalative(graph, batchsize):
    correct_ml, correct_pref, correct_rand = correct_edges(graph, batchsize)
    for i in range(1, len(correct_ml)):
        correct_ml[i] += correct_ml[i-1]
        correct_pref[i] += correct_pref[i-1]
        correct_rand[i] += correct_rand[i-1]
    denominators = [(j+1)*batchsize for j in range(len(correct_ml))]   
    return correct_ml/denominators, correct_pref/denominators, correct_rand/denominators
 
