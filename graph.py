import networkx as nx
import numpy as np

from functions import StrengthFunction

class GraphDecorator:
	graph: nx.DiGraph = None
	features: list[str] = []

	def __init__(self, graph, features):
		self.graph = graph
		self.features = features

	def get_feature(self, name: str) -> np.array:
		"""
		Computes a matrix containing the value of the specified feature for each potential edge.

		:param name: the name of the desired feature.
		:return: an n x n matrix of doubles.
		"""

		feature_dict = nx.get_edge_attributes(self.graph, name)
		n = self.graph.number_of_nodes()
		res = np.zeros((n, n))

		for i in self.graph.nodes:
			for j in self.graph.nodes:
				res[i, j] = feature_dict[i, j] if (i, j) in feature_dict.keys() else 0

		return res

	def get_features(self):
		return list(self.features)

	def get_adj_matrix(self):
		return nx.to_numpy_array(self.graph)

	def get_weighted_adj_matrix(self, strength: StrengthFunction, w: np.array):
		"""
		Uses the strength function and its parameters to combine the features of each edge into a single double value (the
		strength).

		:param strength: the strength function.
		:param w: the weight parameters.
		:return: an n x n matrix of doubles.
		"""

		dot_product = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))

		for i, feat in enumerate(self.features):
			dot_product += self.get_feature(feat) * w[i]

		return nx.to_numpy_array(self.graph) * strength.compute_strength(dot_product)

	def num_nodes(self):
		return self.graph.number_of_nodes()

	def num_features(self):
		return len(self.features)