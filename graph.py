import re
from datetime import datetime

import networkx as nx
import numpy as np
from numpy import ndarray
import openpyxl
import snap
from scipy.sparse import csr_array, dok_array, coo_array, save_npz, load_npz

from functions import StrengthFunction

def parse_timestamp(column: list[str]) -> list[int]:
	res = []
	for cell in column:
		time_string = ' '.join(cell[2:-2].split('\', \''))
		temp = datetime.strptime(time_string, '%Y %b %d %H:%M:%S')
		res.append(int(temp.timestamp()))
	return res

def safe_cast(column: list[str], datatype: type) -> list[str]:
	res = [str(datatype(cell)) if cell.isnumeric() else '' for cell in column]
	return res

def extract_numeric(column: list[str], datatype: type) -> list:
	res = [datatype(float(re.findall(r'\d+[.\d]*', cell)[0])) for cell in column]
	return res

def fetch_user_graph(rumor_number: int, user_id: list[str]) -> tuple[snap.TNEANet, dict]:
	if rumor_number < 1 or rumor_number > 5:
		raise ValueError('The rumor number has to be between 1 and 5.')

	path_graph = f'./in/FN{rumor_number}_DG.graph'
	path_jsonl = f'./in/FN{rumor_number}_Labels.jsonl'
	user_net = snap.ConvertGraph(snap.PNEANet, snap.TNGraph.Load(snap.TFIn(path_graph)))

	user_id_all = []
	with open(path_jsonl, 'r') as f:
		for line in f:
			x = len(line)
			user_id_all.append(line[0:x-1])
	user_id_all.pop(0)

	i = 0
	uid_to_nid = {}
	for ni in user_net.Nodes():
		uid = user_id_all[i]
		user_net.AddStrAttrDatN(ni, uid, 'user_id')
		if uid in user_id:
			uid_to_nid[uid] = ni.GetId()
		i += 1

	return user_net, uid_to_nid

def fetch_graph(rumor_number: int) -> tuple[nx.DiGraph, list[str]]:
	if rumor_number < 1 or rumor_number > 5:
		raise ValueError('The rumor number has to be between 1 and 5.')

	path_input = f'./in/FN{rumor_number}_DD.xlsx'
	tweet_data = openpyxl.load_workbook(path_input)['Sheet1']

	# Fetch attributes from .xlsx file
	user_id = [str(int(cell.value)) for cell in tweet_data['A'][1:]][::-1]
	num_tweets = [int(cell.value) for cell in tweet_data['B'][1:]][::-1]
	language = [cell.value for cell in tweet_data['C'][1:]][::-1]
	num_followers = [int(cell.value) for cell in tweet_data['D'][1:]][::-1]
	num_following = [int(cell.value) for cell in tweet_data['E'][1:]][::-1]
	timestamp = parse_timestamp([cell.value for cell in tweet_data['F'][1:]])[::-1]
	num_likes = [int(cell.value) for cell in tweet_data['G'][1:]][::-1]
	num_retweets = [int(cell.value) for cell in tweet_data['H'][1:]][::-1]
	contains_tweet = [bool(int(cell.value)) for cell in tweet_data['I'][1:]][::-1]
	tweet_id = [str(int(cell.value)) for cell in tweet_data['J'][1:]][::-1]
	retweet_id = safe_cast([cell.value for cell in tweet_data['K'][1:]], int)[::-1]
	quote_id = safe_cast([cell.value for cell in tweet_data['L'][1:]], int)[::-1]
	reply_id = safe_cast([cell.value for cell in tweet_data['M'][1:]], int)[::-1]
	frequency = extract_numeric([cell.value for cell in tweet_data['N'][1:]], int)[::-1]
	label = [cell.value for cell in tweet_data['O'][1:]][::-1]

	baseline_timestamp = timestamp[0]
	rel_timestamp = [t - baseline_timestamp for t in timestamp]

	num_original_tweets = len(tweet_id)

	# Add the missing tweets
	for i, rt, qt, rp in zip(range(num_original_tweets), retweet_id, quote_id, reply_id):
		ref_id = None

		if rt != '':
			ref_id = rt
		elif qt != '':
			ref_id = qt
		elif rp != '':
			ref_id = rp

		if ref_id is not None:
			if ref_id not in tweet_id:
				tweet_id.append(ref_id)
				user_id.append('')
				num_tweets.append(1)
				language.append('')
				num_followers.append(0)
				num_following.append(0)
				rel_timestamp.append(rel_timestamp[i] - 60)
				num_likes.append(0)
				num_retweets.append(int(ref_id == rt))
				contains_tweet.append(False)
				retweet_id.append('')
				quote_id.append('')
				reply_id.append('')
				frequency.append(2)
				label.append('g')
			else:
				original = tweet_id.index(ref_id)
				num_retweets[original] += int(ref_id == rt)
				frequency[original] += 1

	# Sort nodes in chronological order
	sorted_indices = [t[0] for t in sorted(enumerate(rel_timestamp), key=lambda t: t[1])]

	user_id = [user_id[i] for i in sorted_indices]
	num_tweets = [num_tweets[i] for i in sorted_indices]
	language = [language[i] for i in sorted_indices]
	num_followers = [num_followers[i] for i in sorted_indices]
	num_following = [num_following[i] for i in sorted_indices]
	rel_timestamp = [rel_timestamp[i] for i in sorted_indices]
	num_likes = [num_likes[i] for i in sorted_indices]
	num_retweets = [num_retweets[i] for i in sorted_indices]
	contains_tweet = [contains_tweet[i] for i in sorted_indices]
	tweet_id = [tweet_id[i] for i in sorted_indices]
	retweet_id = [retweet_id[i] for i in sorted_indices]
	quote_id = [quote_id[i] for i in sorted_indices]
	reply_id = [reply_id[i] for i in sorted_indices]
	frequency = [frequency[i] for i in sorted_indices]
	label = [label[i] for i in sorted_indices]

	# Create the directed graph
	num_tweets_final = len(tweet_id)
	tweet_net = nx.DiGraph()
	tweet_net.add_nodes_from(range(num_tweets_final))

	# Add the node attributes to the graph
	node_attrs = {
		'user_id': user_id,
		'num_tweets': num_tweets,
		'language': language,
		'num_followers': num_followers,
		'num_following': num_following,
		'rel_timestamp': rel_timestamp,
		'num_likes': num_likes,
		'num_retweets': num_retweets,
		'contains_tweet': contains_tweet,
		'tweet_id': tweet_id,
		'retweet_id': retweet_id,
		'quote_id': quote_id,
		'reply_id': reply_id,
		'frequency': frequency,
		'label': label
	}

	for name, values in node_attrs.items():
		nx.set_node_attributes(tweet_net, dict(zip(range(num_tweets_final), values)), name)

	# Create the edges
	for i, rt, qt, rp in zip(range(num_original_tweets), retweet_id, quote_id, reply_id):
		ref_id = None

		if rt != '':
			ref_id = rt
		elif qt != '':
			ref_id = qt
		elif rp != '':
			ref_id = rp

		if ref_id is not None:
			original = tweet_id.index(ref_id)
			tweet_net.add_edge(i, original)

	return tweet_net, user_id

def load_feature(feature: str, rumor_number: int) -> dok_array:
	path_prefix = f'./dump/FN{rumor_number}_'
	res = dok_array(load_npz(path_prefix + feature + '.npz'))
	return res

def save_feature(values: dok_array, feature: str, rumor_number: int):
	path_prefix = f'./dump/FN{rumor_number}_'
	save_npz(path_prefix + feature + '.npz', coo_array(values))

class GraphWrapper:
	original_graph: nx.DiGraph = None
	graph: nx.DiGraph = None
	adj_matrix: csr_array = None
	features: dict[str: csr_array] = None
	src_label: ndarray = None
	dst_label: ndarray = None

	def __init__(self, orig: nx.DiGraph, graph: nx.DiGraph, features: dict[str: csr_array], src_label: ndarray,
								dst_label: ndarray):
		self.original_graph = orig
		self.graph = graph
		self.adj_matrix = nx.to_scipy_sparse_array(self.graph, format='csr')
		self.features = features
		self.src_label = src_label
		self.dst_label = dst_label

	def get_feature(self, name: str) -> csr_array:
		"""
		Computes a matrix containing the value of the specified feature for each potential edge.

		:param name: the name of the desired feature.
		:return: an n x n matrix of floats.
		"""
		return self.features[name]

	def get_features(self) -> set[str]:
		return set(self.features.keys())

	def get_adj_matrix(self) -> csr_array:
		return self.adj_matrix

	def get_weighted_adj_matrix(self, strength_fun: StrengthFunction, w: ndarray) -> csr_array:
		"""
		Uses the strength function and its parameters to combine the features of each edge into a single double value (the
		strength).

		:param strength_fun: the strength function.
		:param w: the weight parameters.
		:return: an n x n matrix of float.
		"""

		dot_product = csr_array((self.graph.number_of_nodes(), self.graph.number_of_nodes()))
		i = 0
		for k, v in self.features.items():
			if 'label' not in k:
				dot_product += v * w[i]
				i += 1

		return self.adj_matrix * strength_fun.compute_strength(dot_product)

	def get_size(self) -> int:
		return self.graph.number_of_nodes()

	def num_features(self) -> int:
		return len(self.features)

	def get_positive_link(self, src: int) -> int:
		adj_vec = nx.to_numpy_array(self.original_graph)[src]
		dst = np.where(adj_vec > 0)[0]
		if dst.size == 0:
			return src
		return dst[0]

	def get_negative_links(self, src: int) -> ndarray:
		nodes = np.array(range(self.get_size()))
		adj_vec = np.delete(nodes, self.get_positive_link(src))
		return adj_vec

class GraphData:
	base_graph: nx.DiGraph = None
	adj_matrix: dok_array = None
	features: dict[str: dok_array] = None
	rumor_number: int = None
	user_ids: list[str] = None
	src_label: ndarray = None
	dst_label: ndarray = None

	def __init__(self, rumor_number: int):
		"""
		Generates the overall tweet graph for the specified rumor.
		:param rumor_number: the number of the desired rumor.
		"""

		self.rumor_number = rumor_number
		overall_graph, user_ids = fetch_graph(rumor_number)
		self.base_graph = overall_graph
		self.adj_matrix = nx.to_scipy_sparse_array(self.base_graph, format='dok')
		self.user_ids = user_ids

		n = self.base_graph.number_of_nodes()
		self.src_label = np.zeros((n, n), dtype=np.dtype('U1'))
		self.dst_label = np.zeros((n, n), dtype=np.dtype('U1'))

		keys = ['src_num_tweets', 'dst_num_tweets', 'src_num_followers', 'dst_num_followers', 'src_num_following',
						'dst_num_following', 'src_timestamp', 'dst_timestamp', 'timestamp_diff', 'src_dst_same',
						'src_follows_dst', 'dst_follows_src', 'shortest_path_dir', 'common_neighbors']
		self.features = {k: None for k in keys}

	def fetch_static_features(self, load_from_memory: bool = False):
		"""
		Fetches the user graph-related features of the overall tweet graph. After this method executes, these features
		will be saved in memory and loaded in the GraphData instance.

		:param load_from_memory: if True loads features that have previously been saved to memory, otherwise computes
			them and saves them to memory.
		"""

		print('Fetching static features...')
		n = self.base_graph.number_of_nodes()

		if load_from_memory:
			user_graph, uid_to_nid = None, None
			src_follows_dst = load_feature('src_follows_dst', self.rumor_number)
			dst_follows_src = load_feature('dst_follows_src', self.rumor_number)
			shortest_path_dir = load_feature('shortest_path_dir', self.rumor_number)
			jaccard_coeff = load_feature('jaccard_coeff', self.rumor_number)
			out_degree = None
		else:
			user_graph, uid_to_nid = fetch_user_graph(self.rumor_number, self.user_ids)
			src_follows_dst = dok_array((n, n), dtype=np.int32)
			dst_follows_src = dok_array((n, n), dtype=np.int32)
			shortest_path_dir = dok_array((n, n), dtype=np.int32)
			jaccard_coeff = dok_array((n, n), dtype=np.float64)
			out_degree_vec = {d.GetVal1(): d.GetVal2() for d in user_graph.GetNodeOutDegV()}
			out_degree = [out_degree_vec[uid_to_nid[self.base_graph.nodes[i]['user_id']]] for i in range(n)]

		src_num_tweets = dok_array((n, n), dtype=np.int32)
		dst_num_tweets = dok_array((n, n), dtype=np.int32)
		src_num_followers = dok_array((n, n), dtype=np.int32)
		dst_num_followers = dok_array((n, n), dtype=np.int32)
		src_num_following = dok_array((n, n), dtype=np.int32)
		dst_num_following = dok_array((n, n), dtype=np.int32)
		src_dst_same = dok_array((n, n), dtype=np.int32)
		timestamp_diff = dok_array((n, n), dtype=np.int32)

		# rows, cols = self.adj_matrix.nonzero()
		# for i in np.unique(rows):
		for i in range(n):
			if not load_from_memory and i % 50 == 0:
				print(f'Node {i}')
			# for j in cols[np.where(rows == i)]:
			for j in range(i):
				src_num_tweets[i, j] = self.base_graph.nodes[i]['num_tweets']
				dst_num_tweets[i, j] = self.base_graph.nodes[j]['num_tweets']
				src_num_followers[i, j] = self.base_graph.nodes[i]['num_followers']
				dst_num_followers[i, j] = self.base_graph.nodes[j]['num_followers']
				src_num_following[i, j] = self.base_graph.nodes[i]['num_following']
				dst_num_following[i, j] = self.base_graph.nodes[j]['num_following']

				src_uid = self.base_graph.nodes[i]['user_id']
				dst_uid = self.base_graph.nodes[j]['user_id']
				src_dst_same[i, j] = int(src_uid == dst_uid)

				src_timestamp = self.base_graph.nodes[i]['rel_timestamp']
				dst_timestamp = self.base_graph.nodes[j]['rel_timestamp']
				timestamp_diff[i, j] = src_timestamp - dst_timestamp

				self.src_label[i, j] = self.base_graph.nodes[i]['label']
				self.dst_label[i, j] = self.base_graph.nodes[j]['label']

				if not load_from_memory:
					src_uid = self.base_graph.nodes[i]['user_id']
					dst_uid = self.base_graph.nodes[j]['user_id']
					src_nid = -1 if src_uid not in uid_to_nid.keys() else uid_to_nid[src_uid]
					dst_nid = -1 if dst_uid not in uid_to_nid.keys() else uid_to_nid[dst_uid]

					if src_nid == -1 or dst_nid == -1:
						edge, edge_rev, sp_dir, cn = False, False, -1, 0
					elif src_uid == dst_uid:
						edge, edge_rev, sp_dir, cn = False, False, 0, 0
					else:
						edge = user_graph.IsEdge(src_nid, dst_nid)
						edge_rev = user_graph.IsEdge(dst_nid, src_nid)
						sp_dir = user_graph.GetShortPath(src_nid, dst_nid, True)
						cn = user_graph.GetCmnNbrs(src_nid, dst_nid, False)

					src_follows_dst[i, j] = edge
					dst_follows_src[i, j] = edge_rev
					shortest_path_dir[i, j] = 0 if sp_dir <= 0 else 1 / sp_dir
					# common_neighbors[i, j] = cn
					jaccard_coeff[i, j] = cn / (out_degree[i] + out_degree[j] - cn)

		self.features['src_num_tweets'] = src_num_tweets
		self.features['dst_num_tweets'] = dst_num_tweets
		self.features['src_num_followers'] = src_num_followers
		self.features['dst_num_followers'] = dst_num_followers
		self.features['src_num_following'] = src_num_following
		self.features['dst_num_following'] = dst_num_followingù
		self.features['src_dst_same'] = src_dst_same
		self.features['timestamp_diff'] = timestamp_diff
		self.features['src_follows_dst'] = src_follows_dst
		self.features['dst_follows_src'] = dst_follows_src
		self.features['shortest_path_dir'] = shortest_path_dir
		self.features['jaccard_coeff'] = jaccard_coeff

		if not load_from_memory:
			save_feature(src_follows_dst, 'src_follows_dst', self.rumor_number)
			save_feature(dst_follows_src, 'dst_follows_src', self.rumor_number)
			save_feature(shortest_path_dir, 'shortest_path_dir', self.rumor_number)
			save_feature(jaccard_coeff, 'jaccard_coeff', self.rumor_number)

		print('Fetching complete')

	# noinspection PyCallingNonCallable
	def compute_dynamic_features(self, tweet_graph: nx.DiGraph, mod_graph: nx.DiGraph) -> dict[str: csr_array]:
		n = len(tweet_graph)
		dst_in_degree = dok_array((n, n), dtype=np.int32)
		dst_in_degree_false = dok_array((n, n), dtype=np.int32)
		dst_in_degree_true = dok_array((n, n), dtype=np.int32)
		dst_out_degree = dok_array((n, n), dtype=np.int32)

		rows, cols = nx.to_scipy_sparse_array(mod_graph, format='dok').nonzero()
		for i in np.unique(rows):
			for j in cols[np.where(rows == i)]:
				true_graph = tweet_graph.subgraph([n for n, d in tweet_graph.nodes.items() if d['label'] == 'a' or n == j])
				false_graph = tweet_graph.subgraph([n for n, d in tweet_graph.nodes.items() if d['label'] == 'r' or n == j])
				dst_in_degree[i, j] = tweet_graph.in_degree(j) / (len(tweet_graph))
				dst_in_degree_true[i, j] = true_graph.in_degree(j) / (len(true_graph))
				dst_in_degree_false[i, j] = false_graph.in_degree(j) / (len(false_graph))
				dst_out_degree[i, j] = tweet_graph.out_degree(j)

		res = {k: csr_array(v[:n, :n]) for k, v in self.features.items()}
		res['dst_in_degree'] = csr_array(dst_in_degree)
		res['dst_in_degree_true'] = csr_array(dst_in_degree_true)
		res['dst_in_degree_false'] = csr_array(dst_in_degree_false)
		res['dst_out_degree'] = csr_array(dst_out_degree)

		src = self.src_label[:n, :n]
		dst = self.dst_label[:n, :n]

		return res, src, dst

	def get_snapshot(self, time_offset: int = -1, num_nodes: int = -1) -> GraphWrapper:
		"""
		Computes a subgraph of the overall tweet graph capturing a snapshot of the tweets at a specific instant or after
		a specific number of tweets have been posted. If no parameter is passed, the snapshot will coincide with the
		overall graph.

		:param time_offset: the length of the time interval between the first tweet and the last tweet that will be
			included in the snapshot, expressed in seconds.
		:param num_nodes: the number of tweets to include in the snapshot. If time_offset is a non-negative integer,
			this parameter is ignored.
		:return: the specified snapshot of the overall tweet graph.
		"""

		if time_offset >= 0:
			node_list = []
			for node, data in self.base_graph.nodes.data():
				if data['rel_timestamp'] <= time_offset:
					node_list.append(node)
			res_graph = self.base_graph.subgraph(node_list)

		elif num_nodes > 0:
			node_list = []
			for node in self.base_graph:
				node_list.append(node)
				if len(node_list) >= num_nodes:
					break
			res_graph = self.base_graph.subgraph(node_list)

		else:
			res_graph = self.base_graph

		# Add links to all nodes from the source and make the graph undirected
		mod_graph = nx.DiGraph(nx.DiGraph(res_graph).to_undirected())
		for n in mod_graph.nodes:
			mod_graph.add_edge(n, n)
		last_node = len(mod_graph) - 1
		for n in range(last_node):
			mod_graph.add_edge(last_node, n)

		res_features, src_labels, dst_labels = self.compute_dynamic_features(res_graph, mod_graph)

		return GraphWrapper(res_graph, mod_graph, res_features, src_labels, dst_labels)