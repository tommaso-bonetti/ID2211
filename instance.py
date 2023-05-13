import numpy as np

from functions import StrengthFunction, CostFunction
from graph import GraphData, GraphWrapper

class Instance:
	# Instance holds information on positive and negative links for a source node in a graph.
	source_node_index: int = None
	positive_links: np.array = None
	negative_links: np.array = None
	graph: GraphWrapper = None

	def __init__(self, src: int, graph: GraphWrapper):
		self.source_node_index = src
		self.graph = graph
		self.positive_links = graph.get_positive_links(src)
		self.negative_links = graph.get_negative_links(src)

	def compute_cost(self, strength_fun: StrengthFunction, w: np.array, cost_fun: CostFunction, alpha: float) -> float:
		"""
		Computes the cost for the given weight parameters as defined by Leskovec. The strength and cost functions need
		to be specified as well as the restart probability.

		:param strength_fun: the strength function used to compute the PageRank transition probabilities.
		:param w: the weight parameters for the strength function.
		:param cost_fun: the cost function.
		:param alpha: the PageRank restart probability.
		:return: the cost (float).
		"""

		page_rank = self.compute_page_rank(strength_fun, w, alpha)
		cost = 0

		for positive in self.positive_links:
			cost += np.sum(cost_fun.compute_cost(page_rank[self.negative_links] - page_rank[positive]))

		return cost

	def compute_cost_and_grad(
				self,
				strength_fun: StrengthFunction,
				w: np.array,
				cost_fun: CostFunction,
				alpha: float):
		"""
		Computes the cost and the gradient for the given weight parameters as defined by Leskovec. The strength and cost
		functions need to be specified as well as the restart probability.

		:param strength_fun: the strength function used to compute the PageRank transition probabilities.
		:param w: the weight parameters for the strength function.
		:param cost_fun: the cost function.
		:param alpha: the PageRank restart probability.
		:return: the cost (float) and the gradient (a float for each feature).
		"""

		# Compute the weighted adjacency matrix
		adj_mat = self.graph.get_weighted_adj_matrix(strength_fun, w)

		# Compute the squared sum of the rows of the adjacency matrix
		row_sums = np.sum(adj_mat, 1)
		row_sums_sq = row_sums ** 2

		# Compute the transition probability matrix with respect to a starting node s
		Q = self.compute_transition_prob_matrix(adj_mat, alpha, row_sums)

		# Compute PageRank
		p = np.zeros((self.graph.get_size(), 100))
		p[:, 0] = np.ones(self.graph.get_size()) / self.graph.get_size()

		last_iteration = 0
		i = 1
		while i < 100 and last_iteration == 0:
			p[:, i] = Q.T @ p[:, i-1]
			if np.sum((p[:, i] - p[:, i-1]) ** 2) < 1e-12:
				last_iteration = i
			if i == 99:
				print('p did not converge.')

		page_rank = p[:, last_iteration]

		# Compute the derivative for every feature
		diff_p = np.zeros((self.graph.num_features(), self.graph.get_size()))

		for k, feat in enumerate(self.graph.get_features()):
			# Initialize gradient
			diff_p_t1 = np.zeros(self.graph.get_size())
			diff_p_t2 = np.zeros(self.graph.get_size())

			# Compute dQ
			diff_Q = self.compute_diff_Q(strength_fun, w[k], alpha, feat, adj_mat, row_sums, row_sums_sq)
			i = 0
			conv = False
			while i < 100 and not conv:
				diff_p_t1 = Q.T @ diff_p_t2 + diff_Q.T * p[:, min(i, last_iteration)]
				if np.sum((diff_p_t2 - diff_p_t1) ** 2) < 1e-12:
					conv = True

				diff_p_t2 = diff_p_t1
				if i == 99:
					print('dp did not converge.')

			diff_p[k] = diff_p_t1.T

		# Compute	cost and gradient
		l = np.repeat(self.negative_links.reshape(1, self.negative_links.size), self.positive_links.size, axis=0).T
		d = np.repeat(self.positive_links.reshape(1, self.positive_links.size), self.negative_links.size, axis=0)
		gradient = np.zeros(self.graph.num_features())

		costs, gradients = cost_fun.compute_cost_and_gradient(page_rank[l] - page_rank[d])
		cost = np.sum(costs)
		for k in range(self.graph.num_features()):
			gradient[k] = np.sum(gradients * (diff_p[k][l] - diff_p[k][d]))

		return cost, gradient

	def compute_page_rank(self, strength_fun: StrengthFunction, w: np.array, alpha: float) -> np.array:
		"""
		Computes the PageRank for this instance and the specified parameters, strength function and restart	probability.
		:param strength_fun: the strength function.
		:param w: the weight parameters.
		:param alpha: the restart probability.
		:return: the PageRank rankings for each candidate node.
		"""

		# Compute the	weighted adjacency matrix
		adj_mat = self.graph.get_weighted_adj_matrix(strength_fun, w)
		# Compute the squared sum of the rows of the adjacency matrix
		row_sums = np.sum(adj_mat, 1)

		# Compute the transition probability matrix with respect to a starting node	s
		Q = self.compute_transition_prob_matrix(adj_mat, alpha, row_sums)

		# Compute the	actual PageRank	using	the transition probability matrix
		page_rank_curr = np.ones((self.graph.get_size(), 1)) / self.graph.get_size()
		page_rank_prev = np.ones((self.graph.get_size(), 1)) / self.graph.get_size()

		conv = False
		i = 0
		while i < 100 and not conv:
			page_rank_curr = Q.T @ page_rank_prev
			page_rank_curr /= np.sum(page_rank_curr)

			if np.sum((page_rank_curr - page_rank_prev) ** 2) < 1e-12:
				conv = True
			if i == 99:
				print('p did not converge.')

			page_rank_prev = page_rank_curr

		return page_rank_curr

	def compute_diff_Q(
				self,
				strength_fun: StrengthFunction,
				w_k: float,
				alpha: float,
				feat: str,
				adj_mat: np.array,
				row_sums: np.array,
				row_sums_sq: np.array) -> np.array:

		diff_u = self.graph.get_adj_matrix() * strength_fun.compute_gradient(w_k, self.graph.get_feature(feat))
		row_sums_u = np.sum(diff_u, 1)
		res = (1 - alpha) * (diff_u * row_sums - adj_mat * row_sums_u) / row_sums_sq
		return res

	def compute_transition_prob_matrix(self, adj_mat: np.array, alpha: float, row_sums: np.array) -> np.array:
		n = self.graph.get_size()
		src_idx = self.source_node_index

		# Normalize the adjacency matrix to make it stochastic
		res = np.zeros((n, n))
		for i in range(n):
			res[i] = adj_mat[i] / (1 if row_sums[i] == 0 else row_sums[i])
		# Weight the transition normalized adjacency matrix
		res *= 1 - alpha
		# Add the restart probability
		res[:, src_idx] += alpha

		return res

class Instances:
	# Instances represents a set of source/positive/negative groups with the respective graphs, which can be used as a
	# training or test dataset for link prediction.

	graph: GraphData = None
	instances: list[Instance] = None
	n: int = None

	def __init__(self, rumor_number: int, timestamps: list[int] = None, sizes: list[int] = None):
		self.graph = GraphData(rumor_number)
		self.graph.fetch_static_features(True)
		self.instances = []

		if timestamps is not None:
			for t in timestamps:
				snapshot = self.graph.get_snapshot(time_offset=t)
				self.instances.append(Instance(snapshot.get_size() - 1, snapshot))
		elif sizes is not None:
			dim = len(sizes)
			for i, s in enumerate(sizes):
				print(f'Loading snapshot {i+1}/{dim}...')
				snapshot = self.graph.get_snapshot(num_nodes=s)
				self.instances.append(Instance(s - 1, snapshot))
			snapshot = self.graph.get_snapshot()
			self.instances.append(Instance(snapshot.get_size() - 1, snapshot))
		else:
			snapshot = self.graph.get_snapshot()
			self.instances.append(Instance(snapshot.get_size() - 1, snapshot))

		print('Snapshots loaded')

		self.n = len(self.instances)

	def num_instances(self):
		return self.n

	def compute_cost_and_grad(self, strength_fun: StrengthFunction, w: np.array, cost_fun: CostFunction, alpha: float):
		"""
		Computes the cost and gradient for the instances using the given cost function by invoking the
		compute_cost_and_grad method on each instance.

		:param strength_fun: the strength function.
		:param w: the weight parameters
		:param cost_fun: the cost function.
		:param alpha: the restart probability.
		:return: the cumulative cost (float) and the cumulative gradient (a float for each feature).
		"""

		costs = np.zeros(self.n)
		gradients = np.zeros((self.n, w.size))

		for i, instance in enumerate(self.instances):
			temp_c, temp_g = instance.compute_cost_and_grad(strength_fun, w, cost_fun, alpha)
			costs[i] = temp_c
			gradients[i, :] = temp_g

		cost = np.sum(w ** 2) + np.sum(costs)
		gradient = 2 * w + np.sum(gradients)
		return cost, gradient

	def compute_cost(self, strength_fun: StrengthFunction, w: np.array, cost_fun: CostFunction, alpha: float):
		"""
		Computes the cost for the instances using the given cost function by invoking the compute_cost method on each
		instance.

		:param strength_fun: the strength function.
		:param w: the weight parameter.
		:param cost_fun: the cost function.
		:param alpha: the restart probability.
		:return: the cumulative cost (float).
		"""

		costs = np.zeros(self.n)
		for i, instance in enumerate(self.instances):
			temp_c = instance.compute_cost(strength_fun, w, cost_fun, alpha)
			costs[i] = temp_c

		cost = np.sum(w ** 2) + np.sum(costs)
		return cost

	def num_features(self):
		return self.instances[0].graph.num_features()