import matplotlib.pyplot as plt
import numpy as np

from objective_functions.hierarchical_clustering import *
from experiments.results_manager import *
from experiments.hierarchical_clustering_samples_manager import *
from objective_functions.hierarchical_clustering import *


def make_n_levels_experiment(algorithms, range_n_levels=range(6), decay_factor=.1, division_factor=2, core_community_size=10, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
                             SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_n_levels, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_levels_" + name)
    else:
        samples = n_levels_samples(range_n_levels=range_n_levels, decay_factor=decay_factor, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_n_levels, directory_results, results_file_name + "_n_levels_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_n_levels, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Number of levels")
    plt.ylabel("Score")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_levels_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("n_levels_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()


def make_decay_factor_experiment(algorithms, range_decay_factor=np.linspace(0.01, 1, num=5), n_levels=2, division_factor=2, core_community_size=10, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
                             SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_n_levels, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_levels_" + name)
    else:
        samples = decay_factor_samples(range_decay_factor=range_decay_factor, n_levels=n_levels, division_factor=division_factor, core_community_size=core_community_size, p_in=p_in, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_decay_factor, directory_results, results_file_name + "_n_levels_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_decay_factor, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Decay factor")
    plt.ylabel("Score")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_levels_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("n_levels_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()


def make_division_factor_experiment(algorithms, range_division_factor=range(6), n_levels=2,  decay_factor=.1, core_community_size=10, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
                                    SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_division_factor, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_levels_" + name)
    else:
        samples = division_factor_samples(range_division_factor=range_division_factor, n_levels=n_levels, decay_factor=decay_factor, core_community_size=core_community_size, p_in=p_in, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_division_factor, directory_results, results_file_name + "_n_levels_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_division_factor, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Division factor")
    plt.ylabel("Score")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_levels_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("n_levels_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()


def make_core_community_size_experiment(algorithms, range_core_community_size=range(10, 51, 5), n_levels=2, decay_factor=.1, division_factor=2, p_in=10, n_samples=10, score=lambda graph, dendrogram: graph2tree_cost(graph, dendrogram),
                                        SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_core_community_size, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_levels_" + name)
    else:
        samples = core_community_size_samples(range_core_community_size=range_core_community_size, n_levels=n_levels, decay_factor=decay_factor, division_factor=division_factor, p_in=p_in, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm, score=score)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_core_community_size, directory_results, results_file_name + "_n_levels_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', '+', 'x', '*', '^', 'v']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_core_community_size, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Size of core communities")
    plt.ylabel("Score")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_levels_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("n_levels_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()
