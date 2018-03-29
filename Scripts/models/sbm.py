import numpy as np
import networkx as nx
import random as rnd


class SBM:
    def __init__(self, partition, p_matrix):
        partition = np.array(partition)
        k = len(partition)
        n = sum(partition)
        index = np.cumsum(partition)
        self._clusters = [list((index[i - 1] % n) + range(partition[i])) for i in range(k)]
        self._p_matrix = np.array(p_matrix)

    def create_graph(self, distribution='Binomial'):
        G = nx.Graph()
        G.add_nodes_from(range(self._clusters[-1][-1]))
        if distribution == 'Binomial':
            edges = self._create_edges_binomial()
        elif distribution == 'Poisson':
            edges = self._create_edges_poisson()
        G.add_weighted_edges_from(edges)
        return G

    def _create_edges_binomial(self):
        edges = []
        for i, C_i in enumerate(self._clusters):
            for j, C_j in enumerate(self._clusters):
                for u in C_i:
                    for v in C_j:
                        if u != v and rnd.random() < self._p_matrix[i][j]:
                            edges.append((u, v, 1.))
        return edges

    def _create_edges_poisson(self):
        edges = []
        for i, C_i in enumerate(self._clusters):
            for j, C_j in enumerate(self._clusters):
                for u in C_i:
                    for v in C_j:
                        weight = np.random.poisson(self._p_matrix[i][j])
                        if u != v and weight > 0:
                            edges.append((u, v, weight))
        return edges

    def clusters(self):
        return self._clusters

    def info(self):
        print("\nClusters: ", self._clusters)
        print("Probability matrix: ", self._p_matrix)
