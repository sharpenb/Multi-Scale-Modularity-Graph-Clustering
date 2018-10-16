import numpy as np
import networkx as nx
from models.hsbm import HSBM
from objective_functions.hierarchical_clustering import graph2tree_cost


def n_levels_samples(range_n_levels, decay_factor=.1, division_factor=2, core_community_size=10, p_in=10, n_samples=10):
    samples = []
    for n_levels in range_n_levels:
        samples.append([])
        model = HSBM.balanced(n_levels=n_levels, decay_factor=decay_factor, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph(distribution='Poisson')
            samples[-1].append(graph)
    return samples


def decay_factor_samples(range_decay_factor, n_levels=2, division_factor=2, core_community_size=10, p_in=10, n_samples=10):
    samples = []
    for decay_factor in range_decay_factor:
        samples.append([])
        model = HSBM.balanced(n_levels=n_levels, decay_factor=decay_factor, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph(distribution='Poisson')
            samples[-1].append(graph)
    return samples


def division_factor_samples(range_division_factor, n_levels=2, decay_factor=.1, core_community_size=10, p_in=10, n_samples=10):
    samples = []
    for division_factor in range_division_factor:
        samples.append([])
        model = HSBM.balanced(n_levels=n_levels, decay_factor=decay_factor, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph(distribution='Poisson')
            samples[-1].append(graph)
    return samples


def core_community_size_samples(range_core_community_size, n_levels=2, decay_factor=.1, division_factor=2, p_in=10, n_samples=10):
    samples = []
    for core_community_size in range_core_community_size:
        samples.append([])
        model = HSBM.balanced(n_levels=n_levels, decay_factor=decay_factor, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph(distribution='Poisson')
            samples[-1].append(graph)
    return samples


def samples_evaluation(samples, algorithm, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram)):
    results_score = []
    for i, graph_list in enumerate(samples):
        results_score.append([])
        for j, graph in enumerate(graph_list):
            dendrogram = algorithm(graph)
            results_score[-1].append(score(graph, dendrogram))
    return results_score