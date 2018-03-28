import warnings
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from time import time
from sklearn import cluster, mixture
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from graph_manager.graph_tools import clusters_list2clusters_dict
from graph_manager.dataset_manager import generate_dataset_from_euclidean_points


def euclidean_datasets_evaluation(algorithms, n_samples=1500, file_name=''):
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    noisy_square = np.random.rand(n_samples, 2), None
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    anisotropic_blobs = (X_aniso, y)
    varied_blobs = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    plt.figure(figsize=((9 + len(algorithms)) * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 50,
                    'n_clusters': 3}
    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied_blobs, {'eps': .18, 'n_neighbors': 2}),
        (anisotropic_blobs, {'eps': .15, 'n_neighbors': 2}),
        (blobs, {}),
        (noisy_square, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        G, pos, labels = generate_dataset_from_euclidean_points(X, similarity_measure=lambda p, q: np.exp(-np.linalg.norm(p - q)), threshold=.5)
        connectivity = nx.to_scipy_sparse_matrix(G)
        print("Dataset: ", i_dataset)

        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
        spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

        scikit_clustering_algorithms = [
            ('MiniBatchKMeans', two_means),
            ('AffinityProp', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AggloClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('Birch', birch),
            ('GaussianMixture', gmm)]

        for name, algorithm in scikit_clustering_algorithms:

            t0 = time()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)
            t1 = time()

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(scikit_clustering_algorithms) + len(algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            #          transform=plt.gca().transAxes, size=15,
            #          horizontalalignment='right')
            plot_num += 1

        for name, algorithm in algorithms:

            t0 = time()
            clusters = algorithm(G)
            t1 = time()
            y_pred = clusters_list2clusters_dict(clusters).values()

            plt.subplot(len(datasets), len(scikit_clustering_algorithms) + len(algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            #          transform=plt.gca().transAxes, size=15,
            #          horizontalalignment='right')
            plot_num += 1

    if file_name != "":
        plt.savefig(file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()
