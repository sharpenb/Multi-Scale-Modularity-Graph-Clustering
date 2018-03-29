import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(D, scaling=lambda x:np.log(x), figsize=(15, 8), filename=""):
    plt.figure(figsize=figsize)
    D_scaled = D.copy()
    D_scaled[:, 2] = scaling((D[:, 2])) - scaling((D[0, 2]))
    dendrogram(D_scaled, leaf_rotation=90.)
    plt.axis('off')

    if filename != "":
        plt.savefig(filename + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def plot_dendrogram_clustering(D, clusters, scaling=lambda x: np.log(x), figsize=(15, 8), filename=""):
    n_nodes = np.shape(D)[0] + 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] + sorted(list(mcd.XKCD_COLORS))
    clusters = sorted(clusters, key=len, reverse=True)
    default_color = "grey"
    node_colors={u: "grey" for u in range(n_nodes)}
    for i, c in enumerate(clusters):
        for u in c:
            if i < len(colors):
                node_colors[u] = colors[i]
            else:
                node_colors[u] = "grey"

    cluster_colors = {}
    for t, i12 in enumerate(D[:, :2].astype(int)):
        c1, c2 = (cluster_colors[x] if x > n_nodes - 1 else node_colors[x]
                  for x in i12)
        cluster_colors[t + n_nodes] = c1 if c1 == c2 else default_color

    plt.figure(figsize=figsize)
    D_scaled = D.copy()
    D_scaled[:, 2] = scaling((D[:, 2])) - scaling((D[0, 2]))
    dendrogram(D_scaled, leaf_rotation=90., link_color_func=lambda x: cluster_colors[x])
    plt.axis('off')

    if filename != "":
        plt.savefig(filename + ".pdf", bbox_inches='tight')
    else:
        plt.show()
