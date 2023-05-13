import re
from datetime import datetime

import networkx as nx
import numpy as np
import openpyxl
import snap

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

	# Create the directed graph
	num_original_tweets = len(tweet_id)
	tweet_net = nx.DiGraph()
	tweet_net.add_nodes_from(range(num_original_tweets))

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
		nx.set_node_attributes(tweet_net, dict(zip(range(num_original_tweets), values)), name)

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
			if ref_id not in tweet_id:
				tweet_id.append(ref_id)
				original = tweet_id.index(ref_id)
				label.append('g')
				rel_timestamp.append(rel_timestamp[i] - 60)
				tweet_net.add_node(
							original,
							user_id='',
							num_tweets=0,
							language='',
							num_followers=0,
							num_following=0,
							rel_timestamp=rel_timestamp[i]-60,
							num_likes=0,
							num_retweets=0,
							contains_tweet=False,
							retweet_id='',
							quote_id='',
							reply_id='',
							frequency=1,
							label='g')
			else:
				original = tweet_id.index(ref_id)

			tweet_net.add_edge(i, original)

	return tweet_net, user_id

def load_feature(feature: str, rumor_number: int):
	path_prefix = f'./dump/FN{rumor_number}_'
	res = np.load(path_prefix + feature + '.npy')
	return res

def save_feature(values: np.array, feature: str, rumor_number: int):
	path_prefix = f'./dump/FN{rumor_number}_'
	np.save(path_prefix + feature + '.npy', values)

class GraphWrapper:
	graph: nx.DiGraph = None
	features: dict[str: np.array] = None

	def __init__(self, graph: nx.DiGraph, features: dict[str: np.array]):
		self.graph = graph
		self.features = features

	def get_feature(self, name: str) -> np.array:
		"""
		Computes a matrix containing the value of the specified feature for each potential edge.

		:param name: the name of the desired feature.
		:return: an n x n matrix of floats.
		"""

		if name not in self.features.keys():
			raise KeyError('No such feature.')
		return self.features[name]

	def get_features(self):
		return set(self.features.keys()) - {'src_label', 'dst_label'}

	def get_adj_matrix(self) -> np.array:
		return nx.to_numpy_array(self.graph)

	def get_weighted_adj_matrix(self, strength_fun: StrengthFunction, w: np.array):
		"""
		Uses the strength function and its parameters to combine the features of each edge into a single double value (the
		strength).

		:param strength_fun: the strength function.
		:param w: the weight parameters.
		:return: an n x n matrix of float.
		"""

		dot_product = np.zeros((self.graph.number_of_nodes(), self.graph.number_of_nodes()))

		i = 0
		for k, v in self.features.items():
			if 'label' not in k:
				dot_product += v * w[i]
				i += 1

		return self.get_adj_matrix() * strength_fun.compute_strength(dot_product)

	def get_size(self):
		return self.graph.number_of_nodes()

	def num_features(self):
		return len(self.features) - 2

	def get_positive_links(self, src: int) -> np.array:
		adj_vec = self.get_adj_matrix()[src]
		if np.sum(adj_vec) == 0:
			adj_vec[src] = 1
		return adj_vec.nonzero()[0]

	def get_negative_links(self, src: int) -> np.array:
		adj_vec = self.get_adj_matrix()[src]
		if np.sum(adj_vec) == 0:
			adj_vec[src] = 1

		return (adj_vec == 0).nonzero()[0]

class GraphData:
	base_graph: nx.DiGraph = None
	features: dict[str: np.array] = None
	rumor_number: int = None
	user_ids: list[str] = None

	def __init__(self, rumor_number: int):
		"""
		Generates the overall tweet graph for the specified rumor.
		:param rumor_number: the number of the desired rumor.
		"""

		self.rumor_number = rumor_number
		overall_graph, user_ids = fetch_graph(rumor_number)
		self.base_graph = overall_graph
		self.user_ids = user_ids

		keys = ['src_num_tweets', 'dst_num_tweets', 'src_num_followers', 'dst_num_followers', 'src_num_following',
						'dst_num_following', 'src_timestamp', 'dst_timestamp', 'timestamp_diff', 'src_label', 'dst_label',
						'src_dst_same', 'src_follows_dst', 'dst_follows_src', 'shortest_path_dir', 'common_neighbors']
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
			common_neighbors = load_feature('common_neighbors', self.rumor_number)
		else:
			user_graph, uid_to_nid = fetch_user_graph(self.rumor_number, self.user_ids)
			src_follows_dst = np.zeros((n, n))
			dst_follows_src = np.zeros((n, n))
			shortest_path_dir = np.zeros((n, n))
			common_neighbors = np.zeros((n, n))

		src_num_tweets = np.zeros((n, n))
		dst_num_tweets = np.zeros((n, n))
		src_num_followers = np.zeros((n, n))
		dst_num_followers = np.zeros((n, n))
		src_num_following = np.zeros((n, n))
		dst_num_following = np.zeros((n, n))
		src_timestamp = np.zeros((n, n))
		dst_timestamp = np.zeros((n, n))
		timestamp_diff = np.zeros((n, n))
		src_label = np.zeros((n, n), dtype=str)
		dst_label = np.zeros((n, n), dtype=str)
		src_dst_same = np.zeros((n, n))

		for i in range(n):
			if not load_from_memory and i % 50 == 0:
				print(f'Node {i}')
			for j in range(i):
				src_num_tweets[i, j] = self.base_graph.nodes[i]['num_tweets']
				dst_num_tweets[i, j] = self.base_graph.nodes[j]['num_tweets']
				src_num_followers[i, j] = self.base_graph.nodes[i]['num_followers']
				dst_num_followers[i, j] = self.base_graph.nodes[j]['num_followers']
				src_num_following[i, j] = self.base_graph.nodes[i]['num_following']
				dst_num_following[i, j] = self.base_graph.nodes[j]['num_following']
				src_timestamp[i, j] = self.base_graph.nodes[i]['rel_timestamp']
				dst_timestamp[i, j] = self.base_graph.nodes[j]['rel_timestamp']
				timestamp_diff[i, j] = src_timestamp[i, j] - dst_timestamp[i, j]
				src_label[i, j] = self.base_graph.nodes[i]['label']
				dst_label[i, j] = self.base_graph.nodes[j]['label']
				src_uid = self.base_graph.nodes[i]['user_id']
				dst_uid = self.base_graph.nodes[j]['user_id']
				src_dst_same[i, j] = int(src_uid == dst_uid)

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
					common_neighbors[i, j] = cn

		self.features['src_num_tweets'] = src_num_tweets
		self.features['dst_num_tweets'] = dst_num_tweets
		self.features['src_num_followers'] = src_num_followers
		self.features['dst_num_followers'] = dst_num_followers
		self.features['src_num_following'] = src_num_following
		self.features['dst_num_following'] = dst_num_following
		self.features['src_timestamp'] = src_timestamp
		self.features['dst_timestamp'] = dst_timestamp
		self.features['timestamp_diff'] = timestamp_diff
		self.features['src_label'] = src_label
		self.features['dst_label'] = dst_label
		self.features['src_dst_same'] = src_dst_same
		self.features['src_follows_dst'] = src_follows_dst
		self.features['dst_follows_src'] = dst_follows_src
		self.features['shortest_path_dir'] = shortest_path_dir
		self.features['common_neighbors'] = common_neighbors

		if not load_from_memory:
			save_feature(src_follows_dst, 'src_follows_dst', self.rumor_number)
			save_feature(dst_follows_src, 'dst_follows_src', self.rumor_number)
			save_feature(shortest_path_dir, 'shortest_path_dir', self.rumor_number)
			save_feature(common_neighbors, 'common_neighbors', self.rumor_number)

		print('Fetching complete')

	# noinspection PyCallingNonCallable
	def compute_dynamic_features(self, tweet_graph: nx.DiGraph) -> dict[str: np.array]:
		n = len(tweet_graph)
		dst_in_degree = np.zeros((n, n))
		dst_in_degree_false = np.zeros((n, n))
		dst_in_degree_true = np.zeros((n, n))
		dst_out_degree = np.zeros((n, n))

		for i in range(n):
			for j in range(i):
				true_graph = tweet_graph.subgraph([n for n, d in tweet_graph.nodes.items() if d['label'] == 'a' or n == j])
				false_graph = tweet_graph.subgraph([n for n, d in tweet_graph.nodes.items() if d['label'] == 'r' or n == j])
				dst_in_degree[i, j] = tweet_graph.in_degree(j) / (len(tweet_graph))
				dst_in_degree_true[i, j] = true_graph.in_degree(j) / (len(true_graph))
				dst_in_degree_false[i, j] = false_graph.in_degree(j) / (len(false_graph))
				dst_out_degree[i, j] = tweet_graph.out_degree(j)

		res = {k: v[:n, :n] for k, v in self.features.items()}
		res['dst_in_degree'] = dst_in_degree
		res['dst_in_degree_true'] = dst_in_degree_true
		res['dst_in_degree_false'] = dst_in_degree_false
		res['dst_out_degree'] = dst_out_degree

		return res

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
			i = 0
			for node in self.base_graph:
				if i >= num_nodes:
					break
				node_list.append(node)
				i += 1
			res_graph = self.base_graph.subgraph(node_list)

		else:
			res_graph = self.base_graph.copy()

		res_features = self.compute_dynamic_features(res_graph)

		return GraphWrapper(res_graph, res_features)