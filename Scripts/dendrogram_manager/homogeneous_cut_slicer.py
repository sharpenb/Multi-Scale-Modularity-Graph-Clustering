import numpy as np


def clustering_from_homogeneous_cut(D, cut):
    n_nodes = np.shape(D)[0] + 1
    cluster = {u: [u] for u in range(n_nodes)}
    for t in range(cut):
        cluster[n_nodes + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
    clusters = [cluster[c] for c in cluster]
    return clusters


def best_homogeneous_cut(D, scaling=lambda x: np.log(x)):
    score_scaled = scaling(D[1:, 2]) - scaling(D[:-1, 2])
    cut = np.argmax(score_scaled) + 1
    return cut


def ranking_homogeneous_cuts(D, scaling=lambda x: np.log(x)):
    score_scaled = scaling(D[1:, 2]) - scaling(D[:-1, 2])
    cuts = np.argsort(-score_scaled) + 1
    return cuts
