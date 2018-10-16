import networkx as nx
from time import time


def n_nodes_samples(range_n_nodes, p_edge, n_samples=10):
    samples = []
    for n_nodes in range_n_nodes:
        samples.append([])
        for j in range(n_samples):
            graph = nx.fast_gnp_random_graph(n_nodes, p_edge)
            while not nx.is_connected(graph):
                graph = nx.fast_gnp_random_graph(n_nodes, p_edge)
            for e in graph.edges():
                graph.add_edge(e[0], e[1], weight=1)
            samples[-1].append(graph)
    return samples


def p_edge_samples(range_p_edge, n_nodes, n_samples=10):
    samples = []
    for p_edge in range_p_edge:
        samples.append([])
        for j in range(n_samples):
            graph = nx.fast_gnp_random_graph(n_nodes, p_edge)
            while not nx.is_connected(graph):
                graph = nx.fast_gnp_random_graph(n_nodes, p_edge)
            for e in graph.edges():
                graph.add_edge(e[0], e[1], weight=1)
            samples[-1].append(graph)
    return samples


def samples_evaluation(samples, algorithm):
    results_score = []
    for i, graph_list in enumerate(samples):
        results_score.append([])
        for j, graph in enumerate(graph_list):
            t_start = time()
            algorithm(graph)
            t_end = time()
            results_score[-1].append(t_end - t_start)
    return results_score