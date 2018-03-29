import warnings
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from time import time
from sklearn import cluster, mixture
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score as AMI
from itertools import cycle, islice
from graph_manager.graph_tools import clusters_list2clusters_dict
from graph_manager.dataset_manager import *
from experiments.results_manager import *
from experiments.samples_manager import *


def make_n_blocks_experiment(algorithms, range_n_blocks=range(10, 100, 10), block_size=10, d_in=5., d_out=1., n_samples=10, score=lambda true, pred: AMI(true, pred),
                             SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}
    results_algorithms_blocks = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_n_blocks, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_blocks_" + name)
            range_n_blocks, results_algorithms_blocks[name] = load_results(directory_results, results_file_name + "_n_blocks_blocks_" + name)
    else:
        samples, true_clusters = n_blocks_samples(range_n_blocks, block_size=block_size, d_in=d_in, d_out=d_out, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name], results_algorithms_blocks[name] = samples_evaluation(samples, true_clusters, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_n_blocks, directory_results, results_file_name + "_n_blocks_" + name)
                save_results(results_algorithms_blocks[name], range_n_blocks, directory_results, results_file_name + "_n_blocks_blocks_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_n_blocks, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Number of blocks")
    plt.ylabel("AMI")
    plt.ylim(-.1, 1.1)
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms_blocks[name], range_n_blocks, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Number of blocks")
    plt.ylabel("Number of detected blocks")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_blocks_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def make_block_size_experiment(algorithms, range_block_size=range(10, 50, 5), n_blocks=50, d_in=5., d_out=1., n_samples=10, score=lambda true, pred: AMI(true, pred),
                             SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}
    results_algorithms_blocks = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_block_size, results_algorithms[name] = load_results(directory_results, results_file_name + "_block_size_" + name)
            range_block_size, results_algorithms_blocks[name] = load_results(directory_results, results_file_name + "_block_size_blocks_" + name)
    else:
        samples, true_clusters = block_size_samples(range_block_size=range_block_size, n_blocks=n_blocks, d_in=d_in, d_out=d_out, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name], results_algorithms_blocks[name] = samples_evaluation(samples, true_clusters, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_block_size, directory_results, results_file_name + "_block_size_" + name)
                save_results(results_algorithms_blocks[name], range_block_size, directory_results, results_file_name + "_block_size_blocks_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_block_size, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Block size")
    plt.ylabel("AMI")
    plt.ylim(-.1, 1.1)
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms_blocks[name], range_block_size, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Block size")
    plt.ylabel("Number of detected blocks")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def make_degree_in_out_experiment(algorithms, d_in=5., range_d_out=np.linspace(1, 20, num=10), n_blocks=50, block_size=10, n_samples=10, score=lambda true, pred: AMI(true, pred),
                             SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}
    results_algorithms_blocks = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_d_out, results_algorithms[name] = load_results(directory_results, results_file_name + "_degree_in_out_" + name)
            range_d_out, results_algorithms_blocks[name] = load_results(directory_results, results_file_name + "_degree_in_out_blocks_" + name)
    else:
        samples, true_clusters = degree_in_out_samples(d_in=d_in, range_d_out=range_d_out, n_blocks=n_blocks, block_size=block_size, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name], results_algorithms_blocks[name] = samples_evaluation(samples, true_clusters, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_d_out, directory_results, results_file_name + "_degree_in_out_" + name)
                save_results(results_algorithms_blocks[name], range_d_out, directory_results, results_file_name + "_degree_in_out_blocks_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_d_out, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Average external degree")
    plt.ylabel("AMI")
    plt.ylim(-.1, 1.1)
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("degree_in_out_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms_blocks[name], range_d_out, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Average external degree")
    plt.ylabel("Number of detected blocks")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("degree_in_out_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def make_block_size_het_experiment(algorithms, range_param=np.linspace(1., 3., num=10), range_block_size=range(10, 100), n_blocks=50, p_in=.5, p_out=.01, n_samples=10, score=lambda true, pred: AMI(true, pred),
                                   SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}
    results_algorithms_blocks = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_param, results_algorithms[name] = load_results(directory_results, results_file_name + "_block_size_het_" + name)
            range_param, results_algorithms_blocks[name] = load_results(directory_results, results_file_name + "_block_size_het_blocks_" + name)
    else:
        samples, true_clusters = block_size_het_samples(range_param=range_param, range_block_size=range_block_size, n_blocks=n_blocks, p_in=p_in, p_out=p_out, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name], results_algorithms_blocks[name] = samples_evaluation(samples, true_clusters, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_param, directory_results, results_file_name + "_block_size_het_" + name)
                save_results(results_algorithms_blocks[name], range_param, directory_results, results_file_name + "_block_size_het_blocks_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_param, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Heterogeneity parameter")
    plt.ylabel("AMI")
    plt.ylim(-.1, 1.1)
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_het_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms_blocks[name], range_param, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Heterogeneity parameter")
    plt.ylabel("Number of detected blocks")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_het_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def make_block_size_ratio_experiment(algorithms, range_ratio=range(1,10), big_block_size=100, graph_size=600, p_in=.5, p_out=.01, n_samples=10, score=lambda true, pred: AMI(true, pred),
                                   SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}
    results_algorithms_blocks = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_ratio, results_algorithms[name] = load_results(directory_results, results_file_name + "_block_size_ratio_" + name)
            range_ratio, results_algorithms_blocks[name] = load_results(directory_results, results_file_name + "_block_size_ratio_blocks_" + name)
    else:
        samples, true_clusters = block_size_ratio_samples(range_ratio=range_ratio, big_block_size=big_block_size, graph_size=graph_size, p_in=p_in, p_out=p_out, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name], results_algorithms_blocks[name] = samples_evaluation(samples, true_clusters, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_ratio, directory_results, results_file_name + "_block_size_ratio_" + name)
                save_results(results_algorithms_blocks[name], range_ratio, directory_results, results_file_name + "_block_size_ratio_blocks_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_ratio, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Block size ratio")
    plt.ylabel("AMI")
    plt.ylim(-.1, 1.1)
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_ratio_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()

    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms_blocks[name], range_ratio, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Block size ratio")
    plt.ylabel("Number of detected blocks")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("block_size_ratio_blocks_" + results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()


def make_graphical_experiments(algorithms=[], n_samples=1500, run_scikit_algorithms=True, SAVE_PLOTS=False, results_file_name=''):
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
        G, pos, labels = generate_dataset_from_euclidean_points(X, similarity_measure=lambda p, q: np.exp(-np.linalg.norm(p - q)), threshold=.6)
        G, pos, labels = connect_dataset_connected_components(G, pos, labels)
        connectivity = nx.to_scipy_sparse_matrix(G)
        print("Dataset: ", i_dataset)

        if not run_scikit_algorithms:
            scikit_algorithms = []
        else:
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

            scikit_algorithms = [
                ('MiniBatchKMeans', two_means),
                ('AffinityProp', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AggloClustering', average_linkage),
                ('DBSCAN', dbscan),
                ('Birch', birch),
                ('GaussianMixture', gmm)]

            for name, algorithm in scikit_algorithms:

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

                plt.subplot(len(datasets), len(scikit_algorithms) + len(algorithms), plot_num)
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

            plt.subplot(len(datasets), len(scikit_algorithms) + len(algorithms), plot_num)
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

    if SAVE_PLOTS:
        plt.savefig(results_file_name + ".pdf", bbox_inches='tight')
    else:
        plt.show()
