import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score as AMI
from models.ppm import PPM
from graph_manager.graph_tools import clusters_list2clusters_dict


def n_blocks_samples(range_n_blocks, block_size=10, d_in=5., d_out=1., n_samples=10):
    samples = []
    true_clusters =[]
    for n_blocks in range_n_blocks:
        samples.append([])
        true_clusters.append([])
        model = PPM.from_degree(n_blocks, block_size, d_in, d_out)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
    return samples, true_clusters


def block_size_samples(range_block_size, n_blocks=50, d_in=5., d_out=1., n_samples=10):
    samples = []
    true_clusters =[]
    for block_size in range_block_size:
        samples.append([])
        true_clusters.append([])
        model = PPM.from_degree(n_blocks, block_size, d_in, d_out)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
    return samples, true_clusters


def degree_in_out_samples(d_in, range_d_out, n_blocks=50, block_size=10, n_samples=10):
    samples = []
    true_clusters =[]
    for d_out in range_d_out:
        samples.append([])
        true_clusters.append([])
        model = PPM.from_degree(n_blocks, block_size, d_in, d_out)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
    return samples, true_clusters


def block_size_het_samples(range_param, range_block_size, n_blocks=50, p_in=.5, p_out=.01, n_samples=10):
    samples = []
    true_clusters = []
    for param in range_param:
        samples.append([])
        true_clusters.append([])
        for j in range(n_samples):
            p = range_block_size**(-param)
            p /= sum(p)
            partition = np.random.choice(range_block_size, size=n_blocks, p=p)
            model = PPM(partition, p_in, p_out)
            graph = model.create_graph()
            while not nx.is_connected(graph):
                p = range_block_size ** (-param)
                p /= sum(p)
                partition = np.random.choice(range_block_size, size=n_blocks, p=p)
                model = PPM(partition, p_in, p_out)
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
    return samples, true_clusters


def block_size_ratio_samples(range_ratio, big_block_size=100, graph_size=600, p_in=.5, p_out=.01, n_samples=10):
    samples = []
    true_clusters = []
    n_big_bocks = int(graph_size/float(2 * big_block_size))
    for ratio in range_ratio:
        samples.append([])
        true_clusters.append([])
        small_block_size = int(big_block_size / float(ratio))
        n_small_blocks = int(graph_size / float(2 * small_block_size))
        partition = n_small_blocks * [small_block_size] + n_big_bocks * [big_block_size]
        model = PPM(partition, p_in, p_out)
        for j in range(n_samples):
            graph = model.create_graph()
            while not nx.is_connected(graph):
                graph = model.create_graph()
            samples[-1].append(graph)
            true_clusters[-1].append(model.clusters())
    return samples, true_clusters


def samples_evaluation(samples, true_clusters, algorithm, score=lambda true, pred: AMI(true, pred)):
    results_score = []
    results_blocks = []
    for i, graph_list in enumerate(samples):
        results_score.append([])
        results_blocks.append([])
        for j, graph in enumerate(graph_list):
            pred_clusters = algorithm(graph)
            true = clusters_list2clusters_dict(true_clusters[i][j]).values()
            pred = clusters_list2clusters_dict(pred_clusters).values()
            results_score[-1].append(score(true, pred))
            results_blocks[-1].append(len(pred_clusters))
    return results_score, results_blocks
