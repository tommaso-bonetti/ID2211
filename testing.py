import networkx as nx
import numpy as np
from graph import GraphWrapper
from rand import random_add_nodes
from pref import pref_add_nodes

#takes graph and a batchsize
#predicts edges to new batchsized blocks 
#returns list woth correct predicitons of each batch
def correct_edges(graph, batchsize):
    correct_ml = [0]*graph.num_nodes
    correct_pref = [0]*graph.num_nodes
    correct_rand = [0]*graph.num_nodes

    adjacency = graph.get_adj_matrix()   

    #TODO, what if batchsize is not congruent with nodes in graph
    for i in range(graph.num_nodes/batchsize):
        in_graph = i*batchsize
        #subgraph = graph.getsubgraph(in_graph)
        #to_add = graph.getnodes(in_graph, batchsize)
        sub_adjacency = adjacency[:in_graph][in_graph]
        goal_adjacency = adjacency[:(i+1)*batchsize][:(i+1)*batchsize]

        correct_ml[i] = batchsize #TODO, is eqal to max correct atm
        correct_pref[i] = sum(pref_add_nodes(sub_adjacency, batchsize)!= goal_adjacency)/2
        correct_rand[i] = sum(random_add_nodes(sub_adjacency, batchsize)!= goal_adjacency)/2

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
 
def main():
    
    #read graf data

    #read model

    #run prediction test

    #graph results

    print("test")


if __name__ == '__main__':
    main()
