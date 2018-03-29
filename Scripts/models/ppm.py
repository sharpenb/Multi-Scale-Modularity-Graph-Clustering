import numpy as np
import networkx as nx
import random as rnd


class PPM:
    def __init__(self, partition, p_in, p_out):
        partition = np.array(partition)
        k = len(partition)
        n = sum(partition)
        index = np.cumsum(partition)
        self._clusters = [list((index[i - 1] % n) + range(partition[i])) for i in range(k)]
        self._p_in = p_in
        self._p_out = p_out

    @staticmethod
    def from_probability(partition, p_in, p_out):
        return PPM(partition, p_in, p_out)

    @staticmethod
    def from_degree(n_clusters, size_cluster, d_in, d_out):
        return PPM(n_clusters * [size_cluster], min(d_in / float(size_cluster), 1), min(d_out / float((n_clusters-1) * size_cluster), 1))

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
                if i == j:
                    for u in C_i:
                        for v in C_j:
                            if u != v and rnd.random() < self._p_in:
                                edges.append((u, v, 1.))
                else:
                    for u in C_i:
                        for v in C_j:
                            if u != v and rnd.random() < self._p_out:
                                edges.append((u, v, 1.))
        return edges

    def _create_edges_poisson(self):
        edges = []
        for i, C_i in enumerate(self._clusters):
            for j, C_j in enumerate(self._clusters):
                if i == j:
                    for u in C_i:
                        for v in C_j:
                            weight = np.random.poisson(self._p_in)
                            if u != v and weight > 0:
                                edges.append((u, v, weight))

                else:
                    for u in C_i:
                        for v in C_j:
                            weight = np.random.poisson(self._p_out)
                            if u != v and weight > 0:
                                edges.append((u, v, weight))

        return edges

    def clusters(self):
        return self._clusters

    def info(self):
        print("\nClusters: ", self._clusters)
        print("Probability in: ", self._p_in)
        print("Probability out: ", self._p_out)
