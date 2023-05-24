from time import time

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_array

from functions import StrengthFunction, CostFunction
from graph import GraphData, GraphWrapper

class Instance:
	# Instance holds information on positive and negative links for a source node in a graph.
	source_node_index: int = None
	positive_link: int = None
	negative_links: ndarray = None
	graph: GraphWrapper = None

	def __init__(self, src: int, graph: GraphWrapper):
		self.source_node_index = src
		self.graph = graph
		self.positive_link = graph.get_positive_link(src)
		self.negative_links = graph.get_negative_links(src)

	def compute_cost(self, strength_fun: StrengthFunction, w: ndarray, cost_fun: CostFunction, alpha: float) -> float:
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
		cost = np.sum(cost_fun.compute_cost(page_rank[self.negative_links] - page_rank[self.positive_link]))
		return 0 + cost

	def compute_cost_and_grad(
				self,
				strength_fun: StrengthFunction,
				w: ndarray,
				cost_fun: CostFunction,
				alpha: float) -> tuple[ndarray, ndarray]:
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
		row_sums = adj_mat.sum(axis=1)
		row_sums_sq = row_sums ** 2

		# Compute the transition probability matrix with respect to a starting node s
		Q = self.compute_transition_prob_matrix(adj_mat, alpha, row_sums)
		Q_T = Q.transpose()

		# Compute PageRank
		p = np.zeros((self.graph.get_size(), 100))
		p[:, 0] = np.full(self.graph.get_size(), 1 / self.graph.get_size())

		last_iteration = 0
		i = 1
		while i < 100 and last_iteration == 0:
			p[:, i] = Q_T @ p[:, i-1]
			if np.sum((p[:, i] - p[:, i-1]) ** 2) < 1e-8:
				last_iteration = i
			if i == 99:
				last_iteration = i
				margin = np.sum((p[:, i] - p[:, i - 1]) ** 2)
				# print(f'p did not converge: margin {margin:.2E}')
			i += 1

		page_rank = p[:, last_iteration]

		# Compute the derivative for every feature
		diff_p = np.zeros((self.graph.num_features(), self.graph.get_size()))

		for k, feat in enumerate(self.graph.get_features()):
			# Initialize gradient
			diff_p_t0 = csr_array((self.graph.get_size(), 1))
			diff_p_t1 = csr_array((self.graph.get_size(), 1))

			# Compute dQ
			diff_Q = self.compute_diff_Q(strength_fun, w[k], alpha, feat, adj_mat, row_sums, row_sums_sq)
			diff_Q_T = diff_Q.transpose()
			i = 0
			conv = False
			while i < 100 and not conv:
				diff_p_t1 = Q_T @ diff_p_t0 + diff_Q_T @ csr_array(p[:, min(i, last_iteration)]).transpose()
				if ((diff_p_t0 - diff_p_t1) ** 2).sum() < 1e-8:
					conv = True
				if i == 99:
					margin = ((diff_p_t0 - diff_p_t1) ** 2).sum()
					# print(f'dp did not converge for feature {feat}: margin {margin:.2E}')
				diff_p_t0 = diff_p_t1
				i += 1

			diff_p[k] = diff_p_t1.transpose().toarray()

		# Compute	cost and gradient
		l = self.negative_links
		d = np.full(self.negative_links.size, self.positive_link)
		gradient = np.zeros(self.graph.num_features())

		costs, gradients = cost_fun.compute_cost_and_gradient(page_rank[l] - page_rank[d])
		cost = np.sum(costs)
		for k in range(self.graph.num_features()):
			gradient[k] = np.sum(gradients * (diff_p[k][l] - diff_p[k][d]))

		return cost, gradient

	def compute_page_rank(self, strength_fun: StrengthFunction, w: ndarray, alpha: float) -> ndarray:
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
		row_sums = adj_mat.sum(axis=1)

		# Compute the transition probability matrix with respect to a starting node	s
		Q = self.compute_transition_prob_matrix(adj_mat, alpha, row_sums)
		Q_T = Q.transpose()

		# Compute the	actual PageRank	using	the transition probability matrix
		page_rank_curr = np.ones((self.graph.get_size(), 1)) / self.graph.get_size()
		page_rank_prev = np.ones((self.graph.get_size(), 1)) / self.graph.get_size()

		conv = False
		i = 0
		while i < 100 and not conv:
			page_rank_curr = Q_T @ page_rank_prev
			page_rank_curr /= np.sum(page_rank_curr)

			if np.sum((page_rank_curr - page_rank_prev) ** 2) < 1e-12:
				conv = True
			if i == 99:
				print('p did not converge.')

			i += 1
			page_rank_prev = page_rank_curr

		return page_rank_curr

	def compute_diff_Q(
				self,
				strength_fun: StrengthFunction,
				w_k: float,
				alpha: float,
				feat: str,
				adj_mat: csr_array,
				row_sums: ndarray,
				row_sums_sq: ndarray) -> csr_array:

		'''
		dQ_ju/dw =
			(1 - alpha) * [df_w(psi_ju)/dw * sum_k{f_w(psi_jk)} - f_w(psi_ju) * sum_k{df_w(psi_jk)/dw}] / sum_k{f_w(psi_jk)}^2
						if (j, u) in E
			0     otherwise

		d_Q[j, u] = ( d_fw[j, u] * sum(fw[j, :]) - fw[j, u] * sum(d_fw[j, :]) ) / sum(fw[j, :])^2
		'''

		diff_f = self.graph.get_adj_matrix() * strength_fun.compute_gradient(w_k, self.graph.get_feature(feat))
		row_sums_df = diff_f.sum(axis=1)
		row_sums_2d = np.atleast_2d(row_sums).T
		row_sums_sq_2d = np.nan_to_num(np.atleast_2d(np.reciprocal(row_sums_sq)).T, posinf=0)
		row_sums_df_2d = np.atleast_2d(row_sums_df).T

		res = (1 - alpha) * ((diff_f * row_sums_2d - adj_mat * row_sums_df_2d) * row_sums_sq_2d)
		return self.graph.get_adj_matrix() * res

	def compute_transition_prob_matrix(self, adj_mat: csr_array, alpha: float, row_sums: ndarray) -> csr_array:
		n = self.graph.get_size()
		src_idx = self.source_node_index
		res = adj_mat.copy()

		# Normalize the adjacency matrix to make it stochastic
		rows, _ = res.nonzero()
		res.data /= row_sums[rows]
		# Weight the transition normalized adjacency matrix
		res *= 1 - alpha
		# Add the restart probability
		res_prob = np.zeros((n, n))
		indicator = np.array([1 if s == 0 else alpha for s in row_sums])
		res_prob[:, src_idx] += indicator
		res += csr_array(res_prob)

		return res

class Instances:
	# Instances represents a set of source/positive/negative groups with the respective graphs, which can be used as a
	# training or test dataset for link prediction.

	graph: GraphData = None
	instances: list[Instance] = None
	n: int = None

	def __init__(self, rumor_number: int, timestamps: list[int] = None, sizes: list[int] = None):
		self.graph = GraphData(rumor_number)
		self.graph.fetch_static_features(load_from_memory=True)
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

	def compute_cost_and_grad(self, strength_fun: StrengthFunction, w: ndarray, cost_fun: CostFunction, alpha: float) \
				-> tuple[float, ndarray]:
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
		gradient = 2 * w + np.sum(gradients, axis=0)
		return cost, gradient

	def compute_cost(self, strength_fun: StrengthFunction, w: ndarray, cost_fun: CostFunction, alpha: float):
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

	def compute_grad(self, strength_fun: StrengthFunction, w: ndarray, cost_fun: CostFunction, alpha: float) -> ndarray:
		gradients = np.zeros((self.n, w.size))

		for i, instance in enumerate(self.instances):
			_, temp_g = instance.compute_cost_and_grad(strength_fun, w, cost_fun, alpha)
			gradients[i, :] = temp_g

		gradient = 2 * w + np.sum(gradients, axis=0)
		return gradient

	def predict(self, strength_fun: StrengthFunction, w: ndarray, alpha: float):
		links = []
		predictions = []
		for instance in self.instances:
			scores = instance.compute_page_rank(strength_fun, w, alpha)
			predictions.append(np.argmax(scores))
			links.append(instance.positive_link)

		return links, predictions

	def num_features(self):
		return self.instances[0].graph.num_features()