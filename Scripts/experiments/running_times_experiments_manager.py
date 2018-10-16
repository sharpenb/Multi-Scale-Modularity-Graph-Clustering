import warnings
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from time import time
from graph_manager.dataset_manager import *
from experiments.results_manager import *
from experiments.running_times_samples_manager import *


def make_n_nodes_running_times_experiment(algorithms, range_n_nodes=range(50, 100, 10), p_edge=.1, n_samples=10,
                                          SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_n_nodes, results_algorithms[name] = load_results(directory_results, results_file_name + "_n_nodes_time_" + name)
    else:
        samples = n_nodes_samples(range_n_nodes, p_edge, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_n_nodes, directory_results, results_file_name + "_n_nodes_time_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate']
    markers = ['o', '+', 'x', '*', '^', 'v', 's', 'h']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_n_nodes, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Number of nodes")
    plt.ylabel("Running time (seconds)")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("n_nodes_time_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("n_nodes_time_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()


def make_p_edge_running_times_experiment(algorithms, range_p_edge=np.linspace(.01, 1, 10), n_nodes=200, n_samples=10,
                                         SAVE_PLOTS=False, LOAD_RESULTS=False, SAVE_RESULTS=False, directory_results="", results_file_name=""):
    results_algorithms = {}

    if LOAD_RESULTS:
        for name, algorithm in algorithms:
            range_p_edge, results_algorithms[name] = load_results(directory_results, results_file_name + "_p_edge_time_" + name)
    else:
        samples = p_edge_samples(range_p_edge, n_nodes, n_samples=n_samples)
        for name, algorithm in algorithms:
            results_algorithms[name] = samples_evaluation(samples, algorithm)
            if SAVE_RESULTS:
                save_results(results_algorithms[name], range_p_edge, directory_results, results_file_name + "_p_edge_time_" + name)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'chocolate']
    markers = ['o', '+', 'x', '*', '^', 'v', 's', 'h']

    plt.figure()
    for i, (name, algorithm) in enumerate(algorithms):
        plot_results(results_algorithms[name], range_p_edge, label=name, color=colors[i], marker=markers[i])
    plt.xlabel("Probability of edge")
    plt.ylabel("Running time (seconds)")
    plt.legend()
    if SAVE_PLOTS:
        plt.savefig("p_edge_time_" + results_file_name + ".pdf", bbox_inches='tight')
        plt.savefig("p_edge_time_" + results_file_name + ".png", bbox_inches='tight')
    else:
        plt.show()
