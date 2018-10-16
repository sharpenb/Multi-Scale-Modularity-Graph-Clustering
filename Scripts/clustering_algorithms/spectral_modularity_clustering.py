import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eig
from itertools import product
from collections import deque


def spectral_modularity_clustering(G, refine=True):
    ## preprocessing
    G = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute="node_name")
    node_name = nx.get_node_attributes(G, 'node_name')

    ## only support unweighted network
    nx.set_edge_attributes(G=G, name='weight', values={edge:1 for edge in G.edges()})

    B = get_base_modularity_matrix(G)

    ## set flags for divisibility of communities
    ## initial community is divisible
    divisible_community = deque([0])

    ## add attributes: all node as one group
    community_dict = {u: 0 for u in G}

    ## overall modularity matrix

    comm_counter = 0

    while len(divisible_community) > 0:
        ## get the first divisible comm index out
        comm_index = divisible_community.popleft()
        g1_nodes, comm_nodes = _divide(G, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            ## indivisible, go to next
            continue
        ## Else divisible, obtain the other group g2
        #### Get the subgraphs (sub-communities)
        g1 = G.subgraph(g1_nodes)
        g2 = G.subgraph(set(comm_nodes).difference(set(g1_nodes)))
        parent = "%d"%comm_index

        ## add g1, g2 to tree and divisible list
        comm_counter += 1
        #community_tree.create_node(comm_counter, "%d" %comm_counter,\
        #                           parent = parent, data = g1_nodes)
        divisible_community.append(comm_counter)
        ## update community
        for u in g1:
            community_dict[u] = comm_counter

        #community_tree.create_node(comm_counter, "%d" %comm_counter,\
        #                          parent = parent, data = list(g2))
        comm_counter += 1
        divisible_community.append(comm_counter)
        ## update community
        for u in g2:
            community_dict[u] = comm_counter

    clusters_dict = {node_name[u]: community_dict[u] for u in G}
    clusters = {}
    for i in clusters_dict:
        if clusters_dict[i] not in clusters:
            clusters[clusters_dict[i]] = []
        clusters[clusters_dict[i]].append(i)
    return clusters.values()


def _divide(network, community_dict, comm_index, B, refine=False):
    comm_nodes = tuple(u for u in community_dict \
                  if community_dict[u] == comm_index)
    B_hat_g = get_mod_matrix(network, comm_nodes, B)

    if B_hat_g.shape[0] < 3:
        beta_s, u_s = largest_eig(B_hat_g)
    else:
        beta_s, u_s = sparse.linalg.eigs(B_hat_g, k=1, which='LR')
    u_1 = u_s[:, 0]
    beta_1 = beta_s[0]
    if beta_1 > 0:
        # divisible
        s = sparse.csc_matrix(np.asmatrix([[1 if u_1_i > 0 else -1] for u_1_i in u_1]))
        if refine:
            improve_modularity(network, comm_nodes, s, B)
        delta_modularity = _get_delta_Q(B_hat_g, s)
        if delta_modularity > 0:
            g1_nodes = np.array([comm_nodes[i] \
                                 for i in range(u_1.shape[0]) \
                                 if s[i,0] > 0])
            #g1 = nx.subgraph(g, g1_nodes)
            if len(g1_nodes) == len(comm_nodes) or len(g1_nodes) == 0:
                # indivisble, return None
                return None, None
            # divisible, return node list for one of the groups
            return g1_nodes, comm_nodes
    # indivisble, return None
    return None, None


def improve_modularity(G, comm_nodes, s, B):

    # iterate until no increment of Q
    B_hat_g = get_mod_matrix(G, comm_nodes, B)
    while True:
        unmoved = list(comm_nodes)
        # node indices to be moved
        node_indices = np.array([], dtype=int)
        # cumulative improvement after moving
        node_improvement = np.array([], dtype=float)
        # keep moving until none left
        while len(unmoved) > 0:
            # init Q
            Q0 = _get_delta_Q(B_hat_g, s)
            scores = np.zeros(len(unmoved))
            for k_index in range(scores.size):
                k = comm_nodes.index(unmoved[k_index])
                s[k, 0] = -s[k, 0]
                scores[k_index] = _get_delta_Q(B_hat_g, s) - Q0
                s[k, 0] = -s[k, 0]
            _j = np.argmax(scores)
            j = comm_nodes.index(unmoved[_j])
            # move j, which has the largest increase or smallest decrease
            s[j, 0] = -s[j, 0]
            node_indices = np.append(node_indices, j)
            if node_improvement.size < 1:
                node_improvement = np.append(node_improvement, scores[_j])
            else:
                node_improvement = np.append(node_improvement, \
                                        node_improvement[-1]+scores[_j])
            #print len(unmoved), 'max: ', max(scores), node_improvement[-1]
            unmoved.pop(_j)
        # the biggest improvement
        max_index = np.argmax(node_improvement)
        # change all the remaining nodes
        # which are not helping
        for i in range(max_index+1, len(comm_nodes)):
            j = node_indices[i]
            s[j,0] = -s[j, 0]
        # if we swap all the nodes, it is actually doing nothing
        if max_index == len(comm_nodes) - 1:
            delta_modularity = 0
        else:
            delta_modularity = node_improvement[max_index]
        if delta_modularity <= 0:
            break


def get_base_modularity_matrix(network):

    if type(network) == nx.Graph:
        return sparse.csc_matrix(nx.modularity_matrix(network))
    elif type(network) == nx.DiGraph:
        return sparse.csc_matrix(nx.directed_modularity_matrix(network))
    else:
        raise TypeError('Graph type not supported. Use either nx.Graph or nx.Digraph')


def _get_delta_Q(X, a):
    delta_Q = (a.T.dot(X)).dot(a)

    return delta_Q[0,0]


def get_modularity(network, community_dict):
    Q = 0
    G = network.copy()
    nx.set_edge_attributes(G, {e:1 for e in G.edges}, 'weight')
    A = nx.to_scipy_sparse_matrix(G).astype(float)

    if type(G) == nx.Graph:
        # for undirected graphs, in and out treated as the same thing
        out_degree = in_degree = dict(nx.degree(G))
        M = 2.*(G.number_of_edges())
        print("Calculating modularity for undirected graph")
    elif type(G) == nx.DiGraph:
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        M = 1.*G.number_of_edges()
        print("Calculating modularity for directed graph")
    else:
        print('Invalid graph type')
        raise TypeError

    nodes = list(G)
    Q = np.sum([A[i,j] - in_degree[nodes[i]]*\
                         out_degree[nodes[j]]/M\
                 for i, j in product(range(len(nodes)),\
                                     range(len(nodes))) \
                if community_dict[nodes[i]] == community_dict[nodes[j]]])
    return Q / M


def get_mod_matrix(G, comm_nodes=None, B=None):
    if comm_nodes is None:
        comm_nodes = list(G)
        return get_base_modularity_matrix(G)

    if B is None:
        B = get_base_modularity_matrix(G)

    # subset of mod matrix in g
    indices = [list(G).index(u) for u in comm_nodes]
    B_g = B[indices, :][:, indices]

    B_hat_g = np.zeros((len(comm_nodes), len(comm_nodes)), dtype=float)

    B_g_rowsum = np.asarray(B_g.sum(axis=1))[:, 0]
    if type(G) == nx.Graph:
        B_g_colsum = np.copy(B_g_rowsum)
    elif type(G) == nx.DiGraph:
        B_g_colsum = np.asarray(B_g.sum(axis=0))[0, :]

    for i in range(B_hat_g.shape[0]):
        for j in range(B_hat_g.shape[0]):
            if i == j:
                B_hat_g[i,j] = B_g[i,j] - 0.5 * (B_g_rowsum[i] + B_g_colsum[i])
            else:
                B_hat_g[i,j] = B_g[i,j]

    if type(G) == nx.DiGraph:
        B_hat_g = B_hat_g + B_hat_g.T

    return sparse.csc_matrix(B_hat_g)


def largest_eig(A):
    vals, vectors = eig(A.todense())
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    max_idx = np.argsort(vals)[-1]
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T
