"""
This module contains the methods used to run the experiments and compare various constructions of HC trees
"""

import networkx as nx
import math

from Tree_Construction import tree
import generators as gens
import graph_methods as graph
import prune_merge

# Constants
STANDARD_TREE_TYPES = ["average_linkage", "single_linkage", "complete_linkage"]
OUR_TREE_TYPES = ["Prune_Merge"]


# TESTING SYNTHETIC DATASETS
def run_experiment_SBM(tree_types, out):
    """
    The method used for comparing HC tree constructions on graphs generated according to the Stochastic Block Model.
    The values of the parameters can be changed at the beginning of the method.
    The method prints the exact and approximate costs for each HC tree construction.

    :param tree_types: The types of tree considered
    :param out: An output stream
    """

    # Initialise Parameters

    # The number of clusters
    k = 3
    # The number of vertices in each cluster
    n = 1000
    # The size of each cluster
    sizes = [n, n, n]
    # The probability of connecting a pair of vertices from different clusters
    q = 0.002
    # The range of probabilities for connecting a pair of vertices in the same cluster
    p_range = [0.06, 0.1, 0.14, 0.18, 0.2]

    # All costs are averaged over num_iterations runs
    num_iterations = 5

    print_initial_message_for_SBM('standard SBM', sizes, k, q, out)

    # Loop through the entire range of p values
    for p in p_range:
        print(f'p = {p}')

        # A dictionary mapping each tree_type to its total cost throughout the iterations
        cost = initialise_costs_to_zero(tree_types)

        for i in range(num_iterations):
            G = gens.generate_SBM_same_probs(k, sizes, inter_prob=q, intra_prob=p)

            print_iteration_message_for_SBM(i + 1, p, G.number_of_nodes(), G.number_of_edges(), out)

            # Get exact_cost of every tree_type and print it
            for tree_type in tree_types:
                exact_cost = get_cost_of_tree(tree_type, G, k)
                print_cost_of_tree_type(tree_type, exact_cost, out)
                cost[tree_type] += exact_cost

        # Print the average cost over all iterations for every algorithms' output tree
        print_average_costs(tree_types, cost, num_iterations, out)


def run_experiment_SBM_planted_clique(tree_types, out):
    """
    The method used for comparing HC tree constructions on graphs generated according to the Stochastic Block Model
    with a planted clique inside.
    The values of the parameters can be changed at the beginning of the method.
    The method prints the exact and approximate costs for each HC tree construction.

    :param tree_types: The types of tree considered
    :param out: An output stream
    """
    # Initialise Parameters

    # The number of clusters
    k = 3
    # The number of vertices in each cluster
    n1 = 1000
    n2 = 1000
    n3 = 1000
    # The size of each cluster
    sizes = [n1, n2, n3]
    # The probability of connecting a pair of vertices from different clusters
    q = 0.002
    # The probability of connecting a pair of vertices from the same clusters
    p = 0.06

    # The range of values of c_p, where c_p is the fraction of vertices in each cluster that form a complete subgraph
    c_p_range = [0.05, 0.1, 0.2, 0.3, 0.4]

    # All costs are averaged over no_iterations runs
    num_iterations = 5

    print_initial_message_for_SBM_planted_clique('SBM with planted clique', sizes, k, q, p, out)

    # Loop through the entire range of c_p values
    for c_p in c_p_range:
        print(f'clique_percentage = {c_p}')
        # A dictionary mapping each construction to the exact cost of its corresponding tree
        cost = initialise_costs_to_zero(tree_types)

        # Loop through all iterations
        for i in range(num_iterations):
            probs = [[p, q, q], [q, p, q], [q, q, p]]
            # The probabilities for the case of different sized clusters
            # probs = [[p, q, q], [q, p, q], [q, q, 5 * p]]

            G = gens.generate_SBM_with_planted_clique(sizes, probs, c_p)

            print_iteration_message_for_SBM_planted_clique(i + 1, c_p, G.number_of_nodes(), G.number_of_edges(), out)

            # Get exact_cost of every tree_type and print it
            for tree_type in tree_types:
                exact_cost = get_cost_of_tree(tree_type, G, k)
                print_cost_of_tree_type(tree_type, exact_cost, out)
                cost[tree_type] += exact_cost

        # Print the average cost over all iterations for every algorithms' output tree
        print_average_costs(tree_types, cost, num_iterations, out)


def run_experiment_HSBM(tree_types, out):
    """
    The method used for comparing HC tree constructions on graphs generated according to the
    Hierarchical Stochastic Block Model.
    The values of the parameters can be changed at the beginning of the method.
    The method prints the exact and approximate costs for each HC tree construction.

    :param tree_types: The types of tree considered
    :param out: An output stream
    """

    # Initialise Parameters

    # The number of clusters
    k = 5
    # The number of vertices in each cluster
    n = 600
    # The size of each cluster
    sizes = [n, n, n, n, n]
    # The minimum probability of connecting a pair of vertices
    q = 0.0005
    # The range of probabilities for connecting a pair of vertices in the same cluster
    p_range = [0.06, 0.1, 0.14, 0.18, 0.2]

    # All costs are averaged over num_iterations runs
    num_iterations = 5

    print_initial_message_for_SBM('HSBM', sizes, k, q, out)

    # Loop through the entire range of p values
    for p in p_range:
        print(f'p = {p}')
        # A dictionary mapping each construction to the exact cost of its corresponding tree
        cost = initialise_costs_to_zero(tree_types)

        for i in range(num_iterations):
            prob_matrix = get_prob_matrix_for_HSBM(p, q)
            G = gens.generate_HSBM(k, sizes, prob_matrix)

            print_iteration_message_for_SBM(i + 1, p, G.number_of_nodes(), G.number_of_edges(), out)

            # Get exact_cost of every tree_type and print it
            for tree_type in tree_types:
                exact_cost = get_cost_of_tree(tree_type, G, k)
                print_cost_of_tree_type(tree_type, exact_cost, out)
                cost[tree_type] += exact_cost

        # Print the average cost over all iterations for every algorithms' output tree
        print_average_costs(tree_types, cost, num_iterations, out)


def run_experiment_complete_graph(tree_types, out):
    """
    The method used for comparing HC tree constructions on complete graphs.
    This method is used as a sanity check since we know all costs should be the same.
    The value of the parameter n can be changed at the beginning of the method.

    :param tree_types: The types of tree considered
    :param out: An output stream
    """

    # The number of vertices in the graph
    n = 300
    G = gens.generate_complete_graph(n)

    print(f'Testing begins for the complete graph; fixed parameters: n = {n}\n')
    out.write(f'Testing begins for the complete graph; fixed parameters: n = {n}\n\n')

    # Test every tree type construction
    for tree_type in tree_types:
        exact_cost = get_cost_of_tree(tree_type, G, k=1)
        print_cost_of_tree_type(tree_type, exact_cost, out)


# TESTING REAL-WORLD DATASETS
def run_experiment_real_data(tree_types, out):
    """
    The method used for comparing HC tree constructions on graphs generated from real-world datasets.

    :param tree_types: The types of tree considered
    :param out: An output stream
    """

    # The datasets considered
    data_tuples = [('IRIS', 3), ('WINE', 5), ('CANCER', 5), ('BOSTON', 5), ('NEWSGROUP', 2)]

    for data_tuple in data_tuples:
        if data_tuple[0] == 'IRIS':
            # check_eigen_gap_data_tuple(data_tuple, out)
            gamma = 5

            run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out)
        elif data_tuple[0] == 'WINE':
            # check_eigen_gap_data_tuple(data_tuple, out)
            gamma = 0.65

            run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out)
        elif data_tuple[0] == 'CANCER':
            # check_eigen_gap_data_tuple(data_tuple, out)
            gamma = 0.65

            run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out)
        elif data_tuple[0] == 'BOSTON':
            gamma = 0.65

            run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out)
        elif data_tuple[0] == 'NEWSGROUP':
            gamma = 0.00003

            run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out)


def run_experiment_real_data_tuple(tree_types, data_tuple, gamma, out):
    """
    This method generates the tree given by the parameter tree_type for the resulting dataset.
    The method prints the exact and approximate costs for each HC tree construction.

    :param tree_types: The types of tree considered
    :param data_tuple: The dataset considered
    :param gamma: The parameter used to control the weight of the edges in the similarity graph
    :param out: An output stream
    """

    data_type, k = data_tuple

    print(f'\nNow we test: {data_type}\n')
    out.write(f'\nNow we test: {data_type}\n')

    G = gens.generate_dataset_graph(data_type, gamma)

    # print(f'Gamma = {gamma}, n = {G.number_of_nodes()}, m = {G.number_of_edges()} \n')
    # print(f'Eigenvalues: {graph.get_smallest_eigs(G, 10)}')

    out.write(f'Sigma = {round(1 / math.sqrt(2 * gamma), 2)}, Gamma = {gamma}, n = {G.number_of_nodes()}, '
              f'edges = {G.number_of_edges()}\n\n')

    for tree_type in tree_types:
        exact_cost = get_cost_of_tree(tree_type, G, k)

        print_cost_of_tree_type(tree_type, exact_cost, out)
    out.write('\n')


# Helping methods
def initialise_costs_to_zero(tree_types):
    """
    This method returns a dictionary cost, where each key is an element in the list tree_types and each value is 0

    :param tree_types: A list of tree types that represent the keys of the dictionary
    :return: The dictionary cost where each value is set to 0
    """

    cost = {}
    for tree_type in tree_types:
        cost[tree_type] = 0
    return cost


def get_cost_of_tree(tree_type, G, k):
    """
    This method generates the tree of type tree_type, of the graph G with (supposedly) k clusters.
    The method then computes and returns the cost of that tree according to Dasgupta's cost function.

    :param tree_type: The type of the tree considered
    :param G: The underlying networkx graph
    :param k: The (supposed) number of clusters
    :return: The cost of the generate tree for the graph G
    """

    # Initialise the cost of the tree to 0
    exact_cost = 0

    print(f'Now we test {tree_type}')
    T = tree.Tree()

    # Generate the HC tree and compute its cost
    if tree_type in STANDARD_TREE_TYPES:
        T.make_tree(G, tree_type)
        exact_cost = T.get_tree_cost()
    elif tree_type in OUR_TREE_TYPES:
        T = prune_merge.prune_merge(G, k)
        exact_cost = T.get_tree_cost()
    else:
        raise Exception('Unknown tree type')

    return exact_cost


def get_prob_matrix_for_HSBM(p, q):
    prob_matrix = [[p, 3 * q, 2 * q, q, q],
                   [3 * q, p, 2 * q, q, q],
                   [2 * q, 2 * q, p, q, q],
                   [q, q, q, p, 2 * q],
                   [q, q, q, 2 * q, p]]

    return prob_matrix


def print_initial_message_for_SBM(message, sizes, k, q, out):
    print('Testing begins for the ' + message + f'; fixed parameters: sizes = {sizes}, k = {k}, q = {q}\n')
    out.write('Testing begins for the ' + message + f'; fixed parameters: sizes = {sizes}, k = {k}, q = {q}\n\n')


def print_initial_message_for_SBM_planted_clique(message, sizes, k, q, p, out):
    print(f'Testing begins for the ' + message + f'; fixed parameters: sizes = {sizes}, k = {k}, q = {q}, p = {p}\n')
    out.write(f'Testing begins for the ' + message + f'; fixed parameters: sizes = {sizes}, k = {k}, q = {q}, p = {p}\n')


def print_iteration_message_for_SBM(iteration, p, n, m, out):
    print(f'Iteration: {iteration}')
    out.write('\n')
    out.write(f'#### Iteration {iteration},  p = {p}, n = {n}, m = {m} ###:\n')


def print_iteration_message_for_SBM_planted_clique(iteration, c_p, n, m, out):
    print(f'Iteration: {iteration}')
    out.write('\n')
    out.write(f'#### Iteration {iteration},  clique_percentage = {c_p}, n = {n}, m = {m} ###: \n\n')


def print_cost_of_tree_type(tree_type, exact_cost, out):
    out.write(f"{tree_type} approx cost : {adjust_print_format(exact_cost)}, exact cost: {exact_cost}\n")
    out.flush()


def print_average_costs(tree_types, cost, num_iterations, out):
    out.write(f"\n########### Averages ###########\n\n")
    for tree_type in tree_types:
        avg_cost = cost[tree_type] / num_iterations
        print_cost_of_tree_type(tree_type, avg_cost, out)
    out.write('\n')


def adjust_print_format(cost_value):
    """
    Given a cost_value we return the string reflecting an approximation, up to two decimal places,
    of its order of magnitude

    :param cost_value: A cost value
    :return: A string reflecting an approximation up to an order of magnitude
    """

    order_of_mag = math.floor(math.log10(cost_value))
    new_val = round(cost_value / 10 ** order_of_mag, 2)
    return str(new_val) + " * 10^" + str(order_of_mag)


def check_eigen_gap_data_tuple(data_tuple, out):
    """
    The method used for finding the best value for parameter gamma that achieves a large eigengap lambda_(k+1)/lambda_k

    :param data_tuple: The dataset considered
    :param out: An output stream
    """

    data_type, k = data_tuple

    out.write(f'Now we test for the eigengap for {data_type}\n\n')

    # Testing with gamma
    gamma_range = [0.03, 0.1, 0.3, 1, 3, 10]

    for gamma in gamma_range:
        print(f'gamma = {gamma}')
        G = gens.generate_dataset_graph(data_type, gamma)
        out.write(f'gamma = {gamma}, n = {G.number_of_nodes()}, edges = {G.number_of_edges()}, connected? : {nx.is_connected(G)}\n\n')
        eigs = graph.get_smallest_eigs(G, k + 1)
        out.write(f'Eigs : {eigs}\n\n')
        out.write(f'Lambda {k} = {eigs[k - 1]}\n')
        out.write(f'Lambda {k + 1} = {eigs[k]}\n')
        out.write(f'Lambda {k + 1} / Lambda {k} = {eigs[k] / eigs[k - 1]}')
        out.write('\n\n')
        out.flush()
