import numpy as np
from heapdict import *


class ClusterTree:
    def __init__(self, cluster_label, distance, pi, p_ab):
        self.cluster_label = cluster_label
        self.d_ab = distance
        self.pi = pi
        self.p_ab = p_ab
        self.up_merge_loss = float('inf')
        self.info_distance = 0.
        self.left = None
        self.right = None


def information_profiler(graph, dendrogram):
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1
    info_dendrogram = dendrogram.copy()

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph_copy.edges():
        weight = graph_copy[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight

    # Compute sigma and build the tree
    u = n_nodes
    cluster_trees = {t: ClusterTree(t, None, w[t] / float(wtot), 0.) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        # Building of the new level
        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]

        w[u] = w.pop(a) + w.pop(b)
        pi = w[u] / float(wtot)
        d_ab = dendrogram[t, 2]
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]['weight'] / float(wtot)
        else:
            p_ab = 0
        new_tree = ClusterTree(u, d_ab, pi, p_ab)
        new_tree.left = left_tree
        new_tree.right = right_tree
        cluster_trees[u] = new_tree

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    # Compute the information loss of each possible merge
    u = 2 * n_nodes - 2
    info_dendrogram[n_nodes - 2, 2] = 0.
    for t in list(reversed(range(n_nodes - 1))):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]
        pi_a = left_tree.pi
        pi_b = right_tree.pi

        current_tree = cluster_trees[u]
        d_ab = current_tree.d_ab
        p_ab = current_tree.p_ab

        # Merge test with left level
        if left_tree.d_ab is not None:
            left_pi_a = left_tree.left.pi
            left_pi_b = left_tree.right.pi
            left_d_ab = left_tree.d_ab
            left_p_ab = left_tree.p_ab

            sigma_ = 1. - 2 * (pi_a * pi_b) / float(d_ab) - 2 * (left_pi_a * left_pi_b) / float(left_d_ab)
            new_d = 2 * (pi_a * pi_b + left_pi_a * left_pi_b) / float(p_ab + left_p_ab)

            left_tree.up_merge_loss = (p_ab + left_p_ab) * np.log(new_d) - (p_ab * np.log(d_ab) + left_p_ab * np.log(left_d_ab)) \
                                      + np.log(sigma_ * 1 / float(1 - (p_ab + left_p_ab)))

            info_dendrogram[left_tree.cluster_label - n_nodes, 2] = info_dendrogram[t, 2] - left_tree.up_merge_loss

        # Merge test with right level
        if right_tree.d_ab is not None:
            right_pi_a = right_tree.left.pi
            right_pi_b = right_tree.right.pi
            right_d_ab = right_tree.d_ab
            right_p_ab = right_tree.p_ab

            new_d = 2 * (pi_a * pi_b + right_pi_a * right_pi_b) / float((p_ab + right_p_ab))

            right_tree.up_merge_loss = (p_ab + right_p_ab) * np.log(new_d) - (p_ab * np.log(d_ab) + right_p_ab * np.log(right_d_ab))

            info_dendrogram[right_tree.cluster_label - n_nodes, 2] = info_dendrogram[t, 2] - right_tree.up_merge_loss

        u -= 1

    return reorder_dendrogram(info_dendrogram)


class ClusterMultTree:
    def __init__(self, cluster_label, distance, pi, p_ab, pi_a_pi_b):
        self.cluster_label = cluster_label
        self.merged_clusters = set([cluster_label])
        self.d_ab = distance
        self.pi = pi
        self.pi_a_pi_b = pi_a_pi_b
        self.p_ab = p_ab
        self.up_merge_loss = float('inf')
        self.up_merge_d_ab = None
        self.father = None
        self.children = set([])


def information_compresser(graph, dendrogram, n_level_merges):  # TODO: Change "for i in range(n_level_merges)" for " while percentage-of-information-diminution < x%"
    graph_copy = graph.copy()
    n_nodes = np.shape(dendrogram)[0] + 1
    info_dendrogram = dendrogram.copy()

    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in graph_copy.edges():
        weight = graph_copy[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight

    # Build the tree
    u = n_nodes
    cluster_trees = {t: ClusterMultTree(t, None, w[t]/float(wtot), 0., None) for t in range(n_nodes)}
    for t in range(n_nodes - 1):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        # Building of the new level
        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]
        pi_a = left_tree.pi
        pi_b = right_tree.pi

        w[u] = w.pop(a) + w.pop(b)
        pi = w[u] / float(wtot)
        d_ab = dendrogram[t, 2]
        if graph_copy.has_edge(a, b):
            p_ab = 2 * graph_copy[a][b]['weight'] / float(wtot)
        else:
            p_ab = 0
        new_tree = ClusterMultTree(u, d_ab, pi, p_ab, pi_a * pi_b)
        new_tree.children.add(left_tree)
        left_tree.father = new_tree
        new_tree.children.add(right_tree)
        right_tree.father = new_tree
        cluster_trees[u] = new_tree

        # Update graph
        graph_copy.add_node(u)
        neighbors_a = list(graph_copy.neighbors(a))
        neighbors_b = list(graph_copy.neighbors(b))
        for v in neighbors_a:
            graph_copy.add_edge(u, v, weight=graph_copy[a][v]['weight'])
        for v in neighbors_b:
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]['weight'] += graph_copy[b][v]['weight']
            else:
                graph_copy.add_edge(u, v, weight=graph_copy[b][v]['weight'])
        graph_copy.remove_node(a)
        graph_copy.remove_node(b)

        u += 1

    # Compute the information loss of each possible merge
    u = n_nodes
    merging_priority = heapdict()
    for t in list(reversed(range(n_nodes - 1))):
        a = int(dendrogram[t][0])
        b = int(dendrogram[t][1])

        left_tree = cluster_trees[a]
        right_tree = cluster_trees[b]

        current_tree = cluster_trees[u]
        d_ab = current_tree.d_ab
        p_ab = current_tree.p_ab
        pi_a_pi_b = current_tree.pi_a_pi_b

        # Loss computation with left level
        if left_tree.d_ab is not None:
            left_d_ab = left_tree.d_ab
            left_p_ab = left_tree.p_ab
            left_pi_a_pi_b = left_tree.pi_a_pi_b

            left_tree.up_merge_d_ab = 2 * (pi_a_pi_b + left_pi_a_pi_b) / float(p_ab + left_p_ab)
            left_tree.up_merge_loss = (p_ab + left_p_ab) * np.log(left_tree.up_merge_d_ab) - (p_ab * np.log(d_ab) + left_p_ab * np.log(left_d_ab))

            merging_priority[left_tree.cluster_label] = left_tree.up_merge_loss

        # Loss computation with right level
        if right_tree.d_ab is not None:
            right_d_ab = right_tree.d_ab
            right_p_ab = right_tree.p_ab
            right_pi_a_pi_b = right_tree.pi_a_pi_b

            right_tree.up_merge_d_ab = 2 * (pi_a_pi_b + right_pi_a_pi_b) / float((p_ab + right_p_ab))
            right_tree.up_merge_loss = (p_ab + right_p_ab) * np.log(right_tree.up_merge_d_ab) - (p_ab * np.log(d_ab) + right_p_ab * np.log(right_d_ab))

            merging_priority[right_tree.cluster_label] = right_tree.up_merge_loss

        u += 1

    # Merge n_levels times
    for n_merges in range(n_level_merges):
        cluster_label, minimum_loss = merging_priority.popitem()

        merged_tree = cluster_trees[cluster_label]
        father_merged_tree = merged_tree.father

        # Merge the two levels
        father_merged_tree.pi_a_pi_b += merged_tree.pi_a_pi_b
        father_merged_tree.p_ab += merged_tree.p_ab
        father_merged_tree.d_ab = merged_tree.up_merge_d_ab
        father_merged_tree.merged_clusters |= merged_tree.merged_clusters
        father_merged_tree.children |= merged_tree.children
        father_merged_tree.children.remove(merged_tree)

        for c in father_merged_tree.merged_clusters:
            info_dendrogram[cluster_trees[c].cluster_label - n_nodes, 2] = father_merged_tree.d_ab
        # info_dendrogram[merged_tree.cluster_label - n_nodes, 2] = info_dendrogram[father_merged_tree.cluster_label - n_nodes, 2]

        # Updates the father and the children loss
        if father_merged_tree.father is not None:
            pi_a_pi_b = father_merged_tree.pi_a_pi_b
            p_ab = father_merged_tree.p_ab
            d_ab = father_merged_tree.d_ab
            father_pi_a_pi_b = father_merged_tree.father.pi_a_pi_b
            father_p_ab = father_merged_tree.father.p_ab
            father_d_ab = father_merged_tree.father.d_ab

            father_merged_tree.up_merge_d_ab = 2 * (pi_a_pi_b + father_pi_a_pi_b) / float((p_ab + father_p_ab))
            father_merged_tree.up_merge_loss = (p_ab + father_p_ab) * np.log(father_merged_tree.up_merge_d_ab) - (p_ab * np.log(d_ab) + father_p_ab * np.log(father_d_ab))

            merging_priority[father_merged_tree.cluster_label] = father_merged_tree.up_merge_loss

        for child in father_merged_tree.children:
            if child.d_ab is not None:
                pi_a_pi_b = father_merged_tree.pi_a_pi_b
                p_ab = father_merged_tree.p_ab
                d_ab = father_merged_tree.d_ab
                child_pi_a_pi_b = child.pi_a_pi_b
                child_p_ab = child.p_ab
                child_d_ab = child.d_ab

                child.up_merge_d_ab = 2 * (pi_a_pi_b + child_pi_a_pi_b) / float((p_ab + child_p_ab))
                child.up_merge_loss = (p_ab + child_p_ab) * np.log(child.up_merge_d_ab) - (p_ab * np.log(d_ab) + child_p_ab * np.log(child_d_ab))

                child.father = father_merged_tree

                merging_priority[child.cluster_label] = child.up_merge_loss

    print(reorder_dendrogram(info_dendrogram))
    print(info_dendrogram)
    return reorder_dendrogram(info_dendrogram)
    # return info_dendrogram


def reorder_dendrogram(D):
    n = np.shape(D)[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = range(n - 1)
    order[1] = np.array(D)[:, 2]
    index = np.lexsort(order)
    nindex = {i: i for i in range(n)}
    nindex.update({n + index[t]: n + t for t in range(n - 1)})
    return np.array([[nindex[int(D[t][0])], nindex[int(D[t][1])], D[t][2], D[t][3]] for t in range(n - 1)])[index, :]
