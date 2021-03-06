import numpy as np
import networkx as nx


def paris(G):
    F = G.copy()
    nodes = list(F.nodes())
    n_nodes = len(nodes)

    # index nodes from 0 to n - 1
    F = nx.convert_node_labels_to_integers(F)

    # node weights
    w = {u: 0 for u in range(n_nodes)}
    wtot = 0
    for (u, v) in F.edges():
        weight = F[u][v]['weight']
        w[u] += weight
        w[v] += weight
        wtot += 2 * weight
        # if u != v:
        #     wtot += weight

    # cluster sizes
    s = {u: 1 for u in range(n_nodes)}

    # connected components
    cc = []

    # dendrogram as list of merges
    D = []

    # cluster index
    u = n_nodes
    while n_nodes > 0:
        # nearest-neighbor chain
        chain = [list(F.nodes())[0]]
        while chain != []:
            a = chain.pop()
            # nearest neighbor
            d_min = float("inf")
            b = -1
            neighbors_a = list(F.neighbors(a))
            for v in neighbors_a:
                if v != a:
                    d = w[v] * w[a] / float(F[a][v]['weight']) / float(wtot)
                    if d < d_min:
                        b = v
                        d_min = d
                    elif d == d_min:
                        b = min(b, v)
            d = d_min
            if chain != []:
                c = chain.pop()
                if b == c:
                    # merge a,b
                    D.append([a, b, d, s[a] + s[b]])
                    # update graph
                    F.add_node(u)
                    neighbors_a = list(F.neighbors(a))
                    neighbors_b = list(F.neighbors(b))
                    for v in neighbors_a:
                        F.add_edge(u, v, weight=F[a][v]['weight'])
                    for v in neighbors_b:
                        if F.has_edge(u, v):
                            F[u][v]['weight'] += F[b][v]['weight']
                        else:
                            F.add_edge(u, v, weight=F[b][v]['weight'])
                    F.remove_node(a)
                    F.remove_node(b)
                    n_nodes -= 1
                    # update weight and size
                    w[u] = w.pop(a) + w.pop(b)
                    s[u] = s.pop(a) + s.pop(b)
                    # change cluster index
                    u += 1
                else:
                    chain.append(c)
                    chain.append(a)
                    chain.append(b)
            elif b >= 0:
                chain.append(a)
                chain.append(b)
            else:
                # remove the connected component
                cc.append((a, s[a]))
                F.remove_node(a)
                w.pop(a)
                s.pop(a)
                n_nodes -= 1

    # add connected components to the dendrogram
    a, s = cc.pop()
    for b, t in cc:
        s += t
        D.append([a, b, float("inf"), s])
        a = u
        u += 1

    return reorder_dendrogram(np.array(D))


def reorder_dendrogram(D):
    n = np.shape(D)[0] + 1
    order = np.zeros((2, n - 1), float)
    order[0] = range(n - 1)
    order[1] = np.array(D)[:, 2]
    index = np.lexsort(order)
    nindex = {i: i for i in range(n)}
    nindex.update({n + index[t]: n + t for t in range(n - 1)})
    return np.array([[nindex[int(D[t][0])], nindex[int(D[t][1])], D[t][2], D[t][3]] for t in range(n - 1)])[index, :]
