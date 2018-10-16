import networkx as nx
from sklearn.cluster import SpectralClustering


def spectral_clustering(G, n_clusters=2):
    adj_mat = nx.to_numpy_matrix(G)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    clusters = {}
    for i in range(len(sc.labels_)):
        if sc.labels_[i] not in clusters:
            clusters[sc.labels_[i]] = []
        clusters[sc.labels_[i]].append(i)
    return clusters.values()
