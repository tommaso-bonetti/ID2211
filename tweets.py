import re
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import openpyxl
import snap

def parse_timestamp(column: list[str]) -> list[int]:
	res = []
	for cell in column:
		time_string = ' '.join(cell[2:-2].split('\', \''))
		temp = datetime.strptime(time_string, '%Y %b %d %H:%M:%S')
		res.append(int(temp.timestamp()))
	return res

def safe_cast(column: list[str], datatype: type) -> list:
	res = [str(datatype(cell)) if cell.isnumeric() else '' for cell in column]
	return res

def extract_numeric(column: list[str], datatype: type) -> list:
	def get_num(s: str):
		return re.findall(r'\d+[.\d]*', s)[0]

	res = [datatype(float(get_num(cell))) for cell in column]
	return res

def plot_components(network):
	connected_components = network.GetWccSzCnt()
	sizes, counts = [], []
	for p in connected_components:
		sizes.append(p.GetVal1())
		counts.append(p.GetVal2())

	lin_bins = np.linspace(min(sizes), max(sizes) + 1, max(sizes) - min(sizes) + 2, dtype=np.int16)
	lin_counts = [counts[sizes.index(s)] if s in sizes else 0 for s in lin_bins][:-1]
	log_bins = np.geomspace(min(sizes), max(sizes) + 1, 15)
	log_counts = [
		sum(lin_counts[k] for k in range(len(lin_bins) - 1) if log_bins[i] <= lin_bins[k] < log_bins[i + 1])
		for i in range(len(log_bins) - 1)
	]

	# log_space = np.geomspace(min(sizes), max(sizes) + 1, 500)
	# fit_line = 58 * np.power(log_space, -1.7)

	# plt.stairs(lin_counts, lin_bins, fill=True, color='orange')
	plt.stairs(log_counts, log_bins, fill=True, color='orange')
	# plt.plot(log_space, fit_line)
	plt.xscale('log')
	plt.yscale('log')
	plt.title('Connected components\' sizes')
	plt.show()

def plot_degrees(network):
	in_degrees = network.GetInDegCnt()
	sizes, counts = [], []
	for p in in_degrees:
		sizes.append(p.GetVal1())
		counts.append(p.GetVal2())

	lin_bins = np.linspace(min(sizes), max(sizes) + 1, max(sizes) - min(sizes) + 2, dtype=np.int16)
	lin_counts = [counts[sizes.index(s)] if s in sizes else 0 for s in lin_bins][:-1]
	log_bins = np.geomspace(min(sizes) + 1, max(sizes) + 2, 20)
	log_counts = [
		sum(lin_counts[k] for k in range(len(lin_bins) - 1) if log_bins[i] <= lin_bins[k] + 1 < log_bins[i + 1])
		for i in range(len(log_bins) - 1)
	]

	log_space = np.geomspace(min(sizes) + 1, max(sizes) + 2, 500)
	fit_line = 58 * np.power(log_space - .9999, -1.7)

	# plt.stairs(lin_counts, lin_bins, fill=True, color='orange')
	plt.stairs(log_counts, log_bins, fill=True, color='orange')
	# plt.plot(log_space - 1, fit_line)
	plt.xscale('log')
	plt.yscale('log')
	plt.xlim(min(sizes) + .7, max(sizes) + 200)
	plt.title('In-degree distribution')
	plt.show()

def print_hubs(network: nx.DiGraph, threshold: int = 10):
	attractors = sorted(
				[(n, network.in_degree(n), network.nodes[n]['label']) for n in list(network)],
				key=lambda node: node[1],
				reverse=True)
	attractors = list(filter(lambda node: node[1] >= threshold, attractors))
	print(attractors)

def draw_network(
			network: nx.Graph,
			label: list[str],
			nodes: list[int] = None,
			spring_fac: float = 2,
			size_exp: float = .67,
			size_scale: int = 20):

	edges = list(network.edges())
	if nodes is None:
		nodes = list(network)
	else:
		edges = [e for e in edges if e[0] in nodes and e[1] in nodes]

	spring_dist = spring_fac / np.sqrt(len(nodes))
	in_degrees = [1 + sum(e[1] == n for e in edges) for n in nodes]
	hub = nodes[in_degrees.index(max(in_degrees))]

	edge_weights = {e: 15 - 3 * np.log10(in_degrees[nodes.index(e[1])]) for e in edges}
	nx.set_edge_attributes(network, edge_weights, 'weight')

	pos = nx.spring_layout(network, k=spring_dist, fixed=[hub], pos={hub: (0, 0)}, weight='weight')

	sizes = size_scale * np.power(in_degrees, size_exp)
	color_map = {
		'r': 'crimson',
		'a': 'forestgreen',
		'q': 'gold',
		'n': 'lightgrey',
		'g': 'cornflowerblue'
	}
	colors = [color_map[l] for i, l in enumerate(label) if i in nodes]

	nx.draw_networkx(
				network,
				pos,
				with_labels=False,
				nodelist=nodes,
				edgelist=edges,
				node_size=sizes,
				node_color=colors,
				width=.25,
				style=':',
				alpha=.75
	)
	plt.show()

def main():
	rumor_number = '4'

	path_input = f'./in/FN{rumor_number}_DD.xlsx'
	tweet_data = openpyxl.load_workbook(path_input)['Sheet1']

	path_jsonl = f'./in/FN{rumor_number}_Labels.jsonl'
	path_graph = f'./in/FN{rumor_number}_DG.graph'

	path_output = f'./out/FN{rumor_number}/'

	user_id = [str(int(cell.value)) for cell in tweet_data['A'][1:]]
	num_tweets = [int(cell.value) for cell in tweet_data['B'][1:]]
	language = [cell.value for cell in tweet_data['C'][1:]]
	num_followers = [int(cell.value) for cell in tweet_data['D'][1:]]
	num_following = [int(cell.value) for cell in tweet_data['E'][1:]]
	timestamp = parse_timestamp([cell.value for cell in tweet_data['F'][1:]])
	num_likes = [int(cell.value) for cell in tweet_data['G'][1:]]
	num_retweets = [int(cell.value) for cell in tweet_data['H'][1:]]
	contains_tweet = [bool(int(cell.value)) for cell in tweet_data['I'][1:]]
	tweet_id = [str(int(cell.value)) for cell in tweet_data['J'][1:]]
	retweet_id = safe_cast([cell.value for cell in tweet_data['K'][1:]], int)
	quote_id = safe_cast([cell.value for cell in tweet_data['L'][1:]], int)
	reply_id = safe_cast([cell.value for cell in tweet_data['M'][1:]], int)
	frequency = extract_numeric([cell.value for cell in tweet_data['N'][1:]], int)
	label = [cell.value for cell in tweet_data['O'][1:]]

	num_original_tweets = len(tweet_id)

	# Create the tweet network
	tweet_net = snap.TNEANet.New()
	tweet_net_nx = nx.DiGraph()
	tweet_net_nx.add_nodes_from(range(num_original_tweets))

	# Define node attributes
	int_attrs = {
		'num_tweets': num_tweets,
		'num_followers': num_followers,
		'num_following': num_following,
		'timestamp': timestamp,
		'num_likes': num_likes,
		'num_retweets': num_retweets,
		'contains_tweet': contains_tweet,
		'frequency': frequency
	}

	str_attrs = {
		'user_id': user_id,
		'language': language,
		'tweet_id': tweet_id,
		'retweet_id': retweet_id,
		'quote_id': quote_id,
		'reply_id': reply_id,
		'label': label
	}

	for attr in int_attrs.keys():
		tweet_net.AddIntAttrN(attr)
	for attr in str_attrs.keys():
		tweet_net.AddStrAttrN(attr)

	# Add the tweets as nodes along with their attributes
	for i in range(num_original_tweets):
		tweet_net.AddNode(i)
		for attr_name, attr_values in int_attrs.items():
			tweet_net.AddIntAttrDatN(i, attr_values[i], attr_name)
		for attr_name, attr_values in str_attrs.items():
			tweet_net.AddStrAttrDatN(i, attr_values[i], attr_name)

	print(tweet_net.GetNodes())

	# Define edge attributes
	tweet_net.AddStrAttrE('ref_type')

	# Create the directed edges
	for i in range(num_original_tweets):
		rt_id = tweet_net.GetStrAttrDatN(i, 'retweet_id')
		qt_id = tweet_net.GetStrAttrDatN(i, 'quote_id')
		rep_id = tweet_net.GetStrAttrDatN(i, 'reply_id')
		ref_id = None
		ref_type = None
		
		if rt_id != '':
			ref_id = rt_id
			ref_type = 'retweet'
		elif qt_id != '':
			ref_id = qt_id
			ref_type = 'quote'
		elif rep_id != '':
			ref_id = rep_id
			ref_type = 'reply'
		
		if ref_id is not None:
			if ref_id not in tweet_id:
				tweet_id.append(ref_id)
				original = tweet_id.index(ref_id)
				tweet_net.AddNode(original)
				tweet_net.AddStrAttrDatN(original, 'g', 'label')
				label.append('g')
			else:
				original = tweet_id.index(ref_id)

			edge_id = tweet_net.AddEdge(i, original)
			tweet_net.AddStrAttrDatE(edge_id, ref_type, 'ref_type')
			tweet_net_nx.add_edge(i, original)

	print(tweet_net.GetEdges())

	label_dict = {n: label[n] for n in range(len(tweet_id))}
	nx.set_node_attributes(tweet_net_nx, label_dict, 'label')

	plot_components(tweet_net)
	plot_degrees(tweet_net)
	print_hubs(tweet_net_nx)

	baseline_timestamp = timestamp[-1]
	n_hours = 128
	nodes = [i for i in range(len(timestamp)) if timestamp[i] - baseline_timestamp <= n_hours * 60 * 60]

	draw_network(tweet_net_nx, label, nodes=None, spring_fac=6)

if __name__ == '__main__':
	main()