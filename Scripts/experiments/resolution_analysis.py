import numpy as np
import matplotlib.pyplot as plt
from objective_functions.flat_clustering import modularity
from clustering_algorithms.louvain import louvain
from clustering_algorithms.paris import paris

def resolution_n_clusters(G, k=40, n_tests=100, file_name =""):
    # Paris
    D = paris(G)
    resolutions_paris = 1 / D[-k:, 2]

    plt.figure(figsize=(8, 5))
    plt.ylim(ymin=0, ymax=len(resolutions_paris))
    plt.xlabel('Resolution')
    plt.ylabel('Number of clusters')
    plt.xscale('log')

    nb_clusters_paris = range(1, len(resolutions_paris) + 1)
    plt.step(list(reversed(resolutions_paris)), nb_clusters_paris, 'b-', linewidth=2, label='Paris')
    for resolution in resolutions_paris:
        plt.axvline(x=resolution, color='k', alpha=.2)

    # Louvain
    resolutions_louvain = np.logspace(np.log10(resolutions_paris[-1]), np.log10(resolutions_paris[0]), num=n_tests)
    nb_clusters_louvain = []
    key_resolutions_louvain = []
    nb_old = 0
    for resolution in resolutions_louvain:
        cluster = louvain(G, resolution=resolution)
        nb = len(cluster)
        nb_clusters_louvain.append(nb)
        if nb > nb_old:
            nb_old = nb
            key_resolutions_louvain.append(resolution)

    plt.plot(resolutions_louvain, nb_clusters_louvain, 'ro', markersize=2, label='Louvain')

    plt.legend(loc=2)
    if file_name != "":
        plt.savefig(file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

def resolution_modularity(G, k=40, n_tests=100, file_name =""):
    # Paris
    D = paris(G)
    resolutions_paris = 1 / D[:, 2]

    plt.figure(figsize=(8, 5))
    plt.xlabel('Resolution')
    plt.ylabel('Modularity')
    plt.xlim(xmin=resolutions_paris[-1] - .1, xmax=resolutions_paris[-k] + .1)

    modularity_paris = []
    n_nodes = np.shape(D)[0] + 1
    cluster = {u: [u] for u in range(n_nodes)}
    for t in range(n_nodes - 1):
        cluster[n_nodes + t] = cluster.pop(int(D[t][0])) + cluster.pop(int(D[t][1]))
        Q = modularity(G, cluster.values(), resolutions_paris[t])
        modularity_paris.append(Q)
    plt.plot(list(reversed(resolutions_paris))[:k], list(reversed(modularity_paris))[:k], 'b-', linewidth=2, label='Paris')
    for resolution in resolutions_paris[-k:]:
        plt.axvline(x=resolution, color='k', alpha=.2)

    # Louvain
    resolutions_louvain = np.linspace(resolutions_paris[-1], resolutions_paris[-k], num=n_tests)
    modularity_louvain = []
    for resolution in resolutions_louvain:
        clusters = louvain(G, resolution=resolution)
        Q = modularity(G, clusters, resolution)
        modularity_louvain.append(Q)

    plt.plot(resolutions_louvain, modularity_louvain, 'ro', markersize=3, label='Louvain')

    plt.legend(loc=3)
    if file_name != "":
        plt.savefig(file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()
