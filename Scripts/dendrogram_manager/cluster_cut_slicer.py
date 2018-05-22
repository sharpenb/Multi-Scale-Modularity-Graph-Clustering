import numpy as np


def clustering_from_cluster_cut(D, cut):
    n_nodes = np.shape(D)[0] + 1
    if cut < n_nodes:
        return [cut]
    else:
        cluster = {u: [u] for u in range(n_nodes)}
        for t in range(n_nodes - 1):
            cluster[n_nodes + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
            if cut == n_nodes + t:
                return cluster[n_nodes + t]


class ClusterTree:
    def __init__(self, cluster_label, distance, size):
        self.cluster_label = cluster_label
        self.distance = distance
        self.size = size
        self.score = 0.
        self.left = None
        self.right = None


def best_cluster_cut(D, scoring=lambda w, x, y: w * (np.log(x) - np.log(y))):
    n_nodes = np.shape(D)[0] + 1
    best_cut = -1
    best_cut_score = 0.
    cluster_trees = {t: ClusterTree(t, 0, 1) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        i = int(D[t][0])
        j = int(D[t][1])
        left_tree = cluster_trees.pop(i)
        right_tree = cluster_trees.pop(j)

        new_distance = D[t, 2]
        new_size = D[t, 3]

        if left_tree.distance > 0.:
            left_tree.score = scoring(left_tree.size, new_distance, left_tree.distance)
        else:
            left_tree.score = 0.
        if left_tree.score > best_cut_score:
            best_cut_score = left_tree.score
            best_cut = left_tree.cluster_label

        if right_tree.distance > 0.:
            right_tree.score = scoring(right_tree.size, new_distance, right_tree.distance)
        else:
            right_tree.score = 0.
        if right_tree.score > best_cut_score:
            best_cut_score = right_tree.score
            best_cut = right_tree.cluster_label

        new_tree = ClusterTree(n_nodes + t, new_distance, new_size)
        new_tree.left = left_tree
        new_tree.right = right_tree
        cluster_trees[n_nodes + t] = new_tree

    return best_cut, best_cut_score


def ranking_cluster_cuts(D, scoring=lambda w, x, y: w * (np.log(x) - np.log(y))):
    n_nodes = np.shape(D)[0] + 1
    cuts = []
    cut_scores = {}
    cluster_trees = {t: ClusterTree(t, 0, 1) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        i = int(D[t][0])
        j = int(D[t][1])
        left_tree = cluster_trees.pop(i)
        right_tree = cluster_trees.pop(j)

        new_distance = D[t, 2]
        new_size = D[t, 3]

        if left_tree.distance > 0.:
            left_tree.score = scoring(left_tree.size, new_distance, left_tree.distance)
        else:
            left_tree.score = 0.
        cuts.append(left_tree.cluster_label)
        cut_scores[left_tree.cluster_label] = left_tree.score

        if right_tree.distance > 0.:
            right_tree.score = scoring(right_tree.size, new_distance, right_tree.distance)
        else:
            right_tree.score = 0.
        cuts.append(right_tree.cluster_label)
        cut_scores[right_tree.cluster_label] = right_tree.score

        new_tree = ClusterTree(n_nodes + t, new_distance, new_size)
        new_tree.left = left_tree
        new_tree.right = right_tree
        cluster_trees[n_nodes + t] = new_tree

    ranked_cuts = sorted(cuts, key=lambda x: cut_scores[x], reverse=True)
    ranked_cut_scores = sorted(list(cut_scores.values()), reverse=True)

    return ranked_cuts, ranked_cut_scores
