"""
An implementation of the Partitioning algorithm
"""

from __future__ import division
import math
from Cheeger_Cut import cheeger_cut
from Tree_Construction import tree
import graph_methods as graph


def compute_improved_partition(G, k):
    """
    This method takes as input a networkx graph G and a (supposed) number of clusters k
    and returns a pair (clusters, critical_nodes). The first element is a list of partitioning sets {P_i} and the second
    is a list of critical nodes grouped corresponding to the trees T_i for each induced graph G[P_i].

    :param G: A networkx graph
    :param k: A (supposed) number of clusters

    :return: A pair of lists (clusters, critical_nodes) that contain the partition sets
    and a list in which every element critical_nodes[i] consists of the critical nodes for clusters[i]
    """

    # Initialise the parameters for the algorithm
    # We compute the smallest k+1 eigenvalues of the normalised Laplacian matrix of G
    eigs = graph.get_smallest_eigs(G, k + 1)
    # We assign the k'th and (k+1)'th eigenvalues to the corresponding variables
    lambda_k_plus_1 = eigs[k]
    lambda_k = eigs[k - 1]
    # A corner case to overcome numerical issues when computing lambda_1
    if k == 1:
        lambda_k = 0
    # In our implementation we set the constant c_0 = 1
    c_0 = 1
    # The theoretical choices of the parameters rho_star and phi_in
    rho_star = min(lambda_k_plus_1 / 10, 30 * c_0 * ((k + 1) ** 5) * math.sqrt(lambda_k))
    phi_in = lambda_k_plus_1 / (140 * (k + 1) * (k + 1))
    # The practical choices for the aforementioned parameters
    phi_in = max(phi_in, 2 * lambda_k)
    rho_star = max(phi_in, rho_star)

    # print_lambda_k_and_lambda_k_plus_1(lambda_k, lambda_k_plus_1)

    clusters, grouped_critical_nodes = get_clusters_and_critical_nodes(G, k, rho_star, phi_in)

    # print_clusters(clusters)

    return clusters, grouped_critical_nodes


def get_clusters_and_critical_nodes(G, k, rho_star, phi_in):
    """
    The implementation of the main body of the partitioning Algorithm.
    The main while-loop of the algorithm is executed as long as a refinement is still possible.

    :param phi_in: An algorithm parameter used to lower bound the inner conductance of each cluster
    :param rho_star: A technical parameter of the algorithm
    :param G: A networkx graph
    :param k: The (supposed) number of clusters

    :return: a list containing an l-wise partitioning of the nodes of G, for some l <= k
    """

    # A list of vertices in the graph G
    vertices = list(G.nodes())
    # Initially the graph contains one cluster P_1 = V with core set core_1 = P_1.
    P_1 = vertices[:]
    core_1 = P_1[:]
    # num_clusters is the variable denoting the current number of clusters
    num_clusters = 1
    # clusters is a list storing the current cluster structure of G (i.e. P_1, ..., P_l)
    clusters = [P_1]
    # core_sets is a list containing the current core_subsets of each cluster.
    # (i.e. core_1, ..., core_(num_clusters) with core_i being a subset of P_i)
    core_sets = [core_1]
    # A list of lists, where each element grouped_critical_nodes[i] is a list of critical nodes from the tree T_i of
    # cluster clusters[i]
    grouped_critical_nodes = []
    # The main loop of the algorithm. We continue as long as an update is possible
    overall_update_is_found = True
    while overall_update_is_found:
        # At the beginning of the loop there is no update found
        overall_update_is_found = False

        # The main loop of the Partition Algorithm. We continue as long as a GT_update is possible
        GT_update_is_found = True
        while GT_update_is_found:
            # First we check if a GT_update is possible
            GT_update_is_found, index_cluster_to_update = check_if_GT_update_is_possible(G, clusters, core_sets,
                                                                                         phi_in)

            if GT_update_is_found:
                GT_update_is_done = False
                # Notation of the corresponding sets of vertices
                P_i = clusters[index_cluster_to_update]
                core_i = core_sets[index_cluster_to_update]

                S = cheeger_cut.cheeger_cut(G.subgraph(P_i))
                S_complement = diff(vertices, S)

                S_plus = intersect(S, core_i)
                S_plus_bar = intersect(S_complement, core_i)
                S_minus = intersect(diff(P_i, core_i), S)
                S_minus_bar = intersect(diff(P_i, core_i), S_complement)

                # Without loss of generality we assume vol(S_plus) < vol(core_i) / 2
                if vol(G, S_plus) > vol(G, S_plus_bar):
                    S_plus, S_plus_bar = S_plus_bar, S_plus
                    S_minus, S_minus_bar = S_minus_bar, S_minus

                # First "if" in the algorithm
                if is_first_if_condition_satisfied(G, S_plus, S_plus_bar, k, num_clusters, rho_star):
                    make_new_cluster_with_subset_T_bar_of_core_i(
                        S_plus, S_plus_bar, clusters, core_sets, index_cluster_to_update)

                    num_clusters += 1
                    # A sanity check update
                    num_clusters = min(num_clusters, k)

                    GT_update_is_done = True

                # Second "if" in the algorithm
                if not GT_update_is_done and is_second_if_condition_satisfied(G, S_plus, S_plus_bar, core_i, k):
                    update_core_to_subset_T_or_T_bar(G, S_plus, S_plus_bar, core_sets, index_cluster_to_update)
                    GT_update_is_done = True

                # Third "if" in the algorithm
                if not GT_update_is_done and is_third_if_condition_satisfied(G, S_minus, k, num_clusters, rho_star):
                    make_new_cluster_with_subset_T_of_P_i(S_minus, clusters, core_sets, index_cluster_to_update)

                    num_clusters += 1
                    # A sanity check update
                    num_clusters = min(num_clusters, k)

                    GT_update_is_done = True

                # At this point only a refinement of the partition is possible
                if not GT_update_is_done:
                    # If there is a cluster P_j s.t. w(P_i - core_i -> P_i)  < w(P_i - core_i -> P_j),
                    # then merge (P_i - core_i) with argmax_(P_j){w(P_i - core_i -> P_j)}

                    P_i_minus_core_i = diff(P_i, core_i)

                    # Find the index j of argmax_(P_j){w(P_i - core_i -> P_j)}.
                    best_cluster_index = find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, P_i_minus_core_i,
                                                                                              clusters)

                    # Forth "if" in the algorithm.
                    if best_cluster_index != index_cluster_to_update:
                        move_subset_T_from_P_i_to_P_j(P_i_minus_core_i, clusters, index_cluster_to_update,
                                                      best_cluster_index)

                        GT_update_is_done = True

                    if not GT_update_is_done:
                        # If there is a cluster P_j s.t. w(S_minus -> P_i)  < w(S_minus -> P_j),
                        # then merge S_minus with argmax_(P_j){w(S_minus -> P_j)}

                        # Find the index j of argmax_(P_j){w(S_minus -> P_j)}.
                        best_cluster_index = find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, S_minus, clusters)

                        # Fifth "if" in the algorithm
                        if best_cluster_index != index_cluster_to_update:
                            move_subset_T_from_P_i_to_P_j(S_minus, clusters, index_cluster_to_update,
                                                          best_cluster_index)

                            GT_update_is_done = True
                if not GT_update_is_done:
                    raise Exception('No GT_update performed in iteration')

        grouped_critical_nodes = []

        # Check if critical nodes need refinements

        for i in range(len(clusters)):
            # Get the list of critical nodes in the degree based construction of the graph G_i = G[P_i]
            P_i = clusters[i]
            core_i = core_sets[i]
            G_i = G.subgraph(P_i)
            T_i = tree.Tree()
            T_i.make_tree(G_i, "degree")
            critical_nodes_of_T_i = T_i.get_critical_nodes()

            grouped_critical_nodes = grouped_critical_nodes + [critical_nodes_of_T_i]
            for node in critical_nodes_of_T_i:
                # Notation
                N = node.vertices
                N_complement = diff(vertices, N)
                N_plus = intersect(N, core_i)
                N_plus_bar = intersect(N_complement, core_i)
                N_minus = intersect(diff(P_i, core_i), N)
                N_minus_bar = intersect(diff(P_i, core_i), N_complement)

                # Sixth "if" of the algorithm, first "if" of the refinement of the nodes,
                if is_sixth_if_condition_satisfied(G, N_plus, N_plus_bar, k, num_clusters, rho_star):
                    make_new_cluster_with_subset_T_bar_of_core_i(
                        N_plus, N_plus_bar, clusters, core_sets, i)

                    num_clusters += 1

                    # A sanity check update
                    num_clusters = min(num_clusters, k)

                    overall_update_is_found = True
                    break

                # Seventh "if" of the algorithm, second if of the refinement of the nodes
                if not overall_update_is_found and is_seventh_if_condition_satisfied(G, N_plus, core_i, k):
                    update_core_to_subset_T_or_T_bar(G, N_plus, N_plus_bar, core_sets, i)

                    overall_update_is_found = True
                    break

                # We attempt to move N_minus to the cluster P_j that maximises w(N_minus -> P_j)
                if not overall_update_is_found and vol(G, N_minus) <= vol(G, P_i) / 2:

                    # Find the index j of argmax_(P_j){w(N_minus -> P_j)}.
                    # If best_cluster_index = i, then the eighth "if" is not satisfied
                    best_cluster_index = find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, N_minus, clusters)

                    # Eighth "if" of the algorithm, third if of the refinement of the nodes.
                    if weight(G, N_minus, P_i) < weight(G, N_minus, clusters[best_cluster_index]):
                        move_subset_T_from_P_i_to_P_j(N_minus, clusters, i,
                                                      best_cluster_index)

                        overall_update_is_found = True
                        break

            if overall_update_is_found:
                break

    return clusters, grouped_critical_nodes


def check_if_GT_update_is_possible(G, clusters, core_sets, phi_in):
    """
    This method checks if the condition of the while-loop is satisfied,
    i.e. if a refinement of the current partition is possible

    :param G: A networkx graph
    :param clusters: A list of clusters corresponding to the current partitioning
    :param core_sets: A list of core_sets corresponding to each of the clusters
    :param phi_in: A parameter corresponding to the inner conductance

    :return: A pair (GT_update_is_found, index_of_cluster_to_update). The first
    element is assigned a boolean value and is True if the while condition is satisfied.
    The second element is the index i of the cluster to be updated
    """

    # Initialise the two returned values
    GT_update_is_found = False
    index_of_cluster_to_update = -1

    # Check if there exists a sparse cut cheeger cut. That is, for each cluster P_i with cheeger cut S check if
    # phi_G[P_i](S) < phi_in
    for i in range(len(clusters)):
        H = G.subgraph(clusters[i])
        S = cheeger_cut.cheeger_cut(H)
        if max(graph.conductance(H, S), graph.conductance(H, graph.complement(G, S))) < phi_in:
            GT_update_is_found, index_of_cluster_to_update = True, i
            break

    # Check if we can refine the current partition. That is, check if there are i and j with
    # w(P_i - core(P_i) -> core(P_i)) < w(P_i - core(P_i) -> P_j)
    if not GT_update_is_found:
        for i in range(len(clusters)):
            P_i, core_i = clusters[i], core_sets[i]
            core_i_complement = diff(P_i, core_i)
            for j in range(len(clusters)):
                if i != j:
                    P_j = clusters[j]
                    if weight(G, core_i_complement, core_i) < weight(G, core_i_complement, P_j):
                        GT_update_is_found, index_of_cluster_to_update = True, i
                        break
            # Once an update is found we break the loop
            if GT_update_is_found:
                break
    return GT_update_is_found, index_of_cluster_to_update


def is_first_if_condition_satisfied(G, S_plus, S_plus_bar, k, num_clusters, rho_star):
    """
    This method checks if the first "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param S_plus: A list containing the vertices in the set S_plus defined as S_plus = intersect(core(P_i), S)
    :param S_plus_bar: A list containing the vertices in the set S_plus_bar defined as S_plus_bar = core(P_i) - S_plus
    :param k: The target number of clusters
    :param num_clusters: The current number of clusters in the algorithm. This is the parameter r in the paper
    :param rho_star: A parameter rho_star as defined in the paper and in the method compute_improved_partition()

    :return: True if the first "if"-condition of the algorithm is satisfied and False otherwise
    """
    Phi_S_plus = graph.conductance(G, S_plus)
    Phi_S_plus_bar = graph.conductance(G, S_plus_bar)
    # Compute the expression on the right hand side of the "if"-condition
    right_hand_side = (1 + 1.0 / (k + 1)) ** (num_clusters + 1) * rho_star
    if max(Phi_S_plus, Phi_S_plus_bar) <= right_hand_side:
        return True
    else:
        return False


def is_second_if_condition_satisfied(G, S_plus, S_plus_bar, core_i, k):
    """
    This method checks if the second "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param S_plus: A list containing the vertices in the set S_plus defined as S_plus = intersect(core(P_i), S)
    :param S_plus_bar: A list containing the vertices in the set S_plus_bar defined as S_plus_bar = core(P_i) - S_plus
    :param core_i: The core set core(P_i) of cluster P_i
    :param k: The target number of clusters

    :return: True if the second "if"-condition of the algorithm is satisfied and False otherwise
    """
    varphi_S_plus = varphi_conductance(G, S_plus, core_i)
    varphi_S_plus_bar = varphi_conductance(G, S_plus_bar, core_i)
    if min(varphi_S_plus, varphi_S_plus_bar) <= 1.0 / (3 * (k + 1)):
        return True
    else:
        return False


def is_third_if_condition_satisfied(G, S_minus, k, num_clusters, rho_star):
    """
    This method checks if the third "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param S_minus: A list containing the vertices in the set S_minus defined as S_minus = S - core(P_i)
    :param k: The target number of clusters
    :param num_clusters: The current number of clusters in the algorithm. This is the parameter r in the paper
    :param rho_star: A parameter rho_star as defined in the paper and in the method compute_improved_partition()

    :return: True if the third "if"-condition of the algorithm is satisfied and False otherwise
    """
    Phi_S_minus = graph.conductance(G, S_minus)
    # Compute the expression on the right hand side of the "if"-condition
    right_hand_side = ((1 + 1.0 / (k + 1)) ** (num_clusters + 1)) * rho_star
    if Phi_S_minus <= right_hand_side:
        return True
    else:
        return False


def is_sixth_if_condition_satisfied(G, N_plus, N_plus_bar, k, num_clusters, rho_star):
    """
    This method checks if the sixth "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param N_plus: A list containing the vertices in the set N_plus defined as N_plus = intersect(core(P_i), N)
    :param N_plus_bar: A list containing the vertices in the set S_plus_bar defined as N_plus_bar = core(P_i) - N_plus
    :param k: The target number of clusters
    :param num_clusters: The current number of clusters in the algorithm. This is the parameter r in the paper
    :param rho_star: A parameter rho_star as defined in the paper and in the method compute_improved_partition()

    :return: True if the sixth "if"-condition of the algorithm is satisfied and False otherwise
    """
    Phi_N_plus = graph.conductance(G, N_plus)
    Phi_N_plus_bar = graph.conductance(G, N_plus_bar)
    # Compute the expression on the right hand side of the "if"-condition
    right_hand_side = (1 + 1 / (k + 1)) ** (num_clusters + 1) * rho_star
    if max(Phi_N_plus, Phi_N_plus_bar) <= right_hand_side:
        return True
    else:
        return False


def is_seventh_if_condition_satisfied(G, N_plus, core_i, k):
    """
    This method checks if the seventh "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param N_plus: A list containing the vertices in the set N_plus defined as N_plus = intersect(core(P_i), N)
    :param core_i: The core set core(P_i) of cluster P_i
    :param k: The target number of clusters

    :return: True if the seventh "if"-condition of the algorithm is satisfied and False otherwise
    """
    varphi_N_plus = varphi_conductance(G, N_plus, core_i)
    if vol(G, N_plus) <= vol(G, core_i) / 2 and varphi_N_plus <= 1.0 / (3 * (k + 1)):
        return True
    else:
        return False


def find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, T, clusters):
    """
    Given a subset of vertices T, this method finds and returns the index j of cluster P_j that maximises w(T -> P_j)

    :param G: A networkx graph
    :param T: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices

    :return: The index j of cluster P_j that maximises w(T -> P_j)
    """
    best_cluster_index = 0
    best_weight_so_far = weight(G, T, clusters[0])
    for j in range(len(clusters)):
        P_j = clusters[j]
        weight_to_P_j = weight(G, T, P_j)
        if j != best_cluster_index and best_weight_so_far < weight_to_P_j:
            best_cluster_index = j
            best_weight_so_far = weight_to_P_j

    return best_cluster_index


def make_new_cluster_with_subset_T_bar_of_core_i(T, T_bar, clusters, core_sets, i):
    """
    This method corresponds to the updates performed should the "if"-conditions #1 or #6 be satisfied.
    Given a partition T, T_bar of core(P_i), the method updates core(P_i) <- T and creates a new cluster with the
    vertices in T_bar

    :param T: A list of vertices such that T is a subset of core(P_i)
    :param T_bar: A list of vertices such that T_bar = core(P_i) - T
    :param clusters: A list of clusters, stored as lists of vertices
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update core_i = T
    core_sets[i] = T[:]
    # Update P_i = P_i - T_bar
    clusters[i] = diff(clusters[i], T_bar)
    # Update P_(num_clusters + 1) = core_(num_clusters + 1) = T_bar
    clusters.append(T_bar[:])
    core_sets.append(T_bar[:])


def update_core_to_subset_T_or_T_bar(G, T, T_bar, core_sets, i):
    """
    This method corresponds to the updates performed should the "if"-conditions #2 or #7 be satisfied.
    Given a partition T, T_bar of core(P_i), the method updates core(P_i) to either T or T_bar of lower conductance

    :param G: A networkx graph
    :param T: A list of vertices such that T is a subset of core(P_i)
    :param T_bar: A list of vertices such that T_bar = core(P_i) - T
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update core_i to either T or T_bar of minimum conductance
    if graph.conductance(G, T) > graph.conductance(G, T_bar):
        T_bar, T = T, T_bar
    core_sets[i] = T[:]


def make_new_cluster_with_subset_T_of_P_i(T, clusters, core_sets, i):
    """
    This method corresponds to the update performed should the "if"-condition #3 be satisfied.
    Given a subset T of P_i, the method creates a new cluster with the vertices in T

    :param T: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update P_i = P_i - T
    clusters[i] = diff(clusters[i], T)

    # Update P_(num_clusters + 1) = core_(num_clusters + 1) = T
    core_sets.append(T[:])
    clusters.append(T[:])


def move_subset_T_from_P_i_to_P_j(T, clusters, i, j):
    """
    This method corresponds to the updates performed should the "if"-conditions #4, #5 of #8 be satisfied.
    Given a subset T of P_i the method moves the set T to cluster P_j

    :param T: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices
    :param i: The index of the original cluster P_i
    :param j: The index of the target cluster P_j
    """
    # Update P_i = P_i - T
    clusters[i] = diff(clusters[i], T)

    # Merge T with P_j, where P_j is argmax_(P_j){w(T -> P_j)}
    clusters[j] = clusters[j] + T


def varphi_conductance(G, S, P):
    """
    This method computes the relative conductance varphi(S, P) for a subset of nodes S of P,
    defined as the ratio (weight(S, P - S)) / (coeff * w(S, V - P)), where coeff = vol(P - S) / vol(P).

    :param G: A networkx graph
    :param S: A subset of nodes included in P
    :param P: An arbitrary subset of nodes in G
    :return: The relative conductance of S with respect to P
    """

    # The relative volume coefficient
    coeff = graph.volume(G, diff(P, S)) / graph.volume(G, P)

    # The complement of P
    P_complement = diff(list(G.nodes()), P)

    if coeff == 0 or weight(G, S, P_complement) == 0:
        return 1
    return weight(G, S, P) / (coeff * weight(G, S, P_complement))


# The weight function w(S->T) defined to be sum of the weights of the edges with one endpoint in S and the other in T/S
def weight(G, S, T):
    return graph.cut_value(G, S, diff(T, S))


# The volume function that returns the volume of a set S in the graph G
def vol(G, S):
    return graph.volume(G, S)


# The diff function that takes as input two lists (of vertices) A and B, and returns the list consisting of elements
# in A that are not in B
def diff(A, B):
    return list(set(A) - set(B))


# The intersect function that takes as input two lists (of vertices) A and B, and returns the list consisting of
# elements that are both in A and in B
def intersect(A, B):
    return list(set(A) & set(B))


def print_lambda_k_and_lambda_k_plus_1(lambda_k, lambda_k_plus_1):
    print(f'Lambda_k : {lambda_k} \nLambda_k+1: {lambda_k_plus_1} \n')


def print_clusters(clusters):
    print(f'Number of clusters found: {len(clusters)}')
    for i in range(len(clusters)):
        print(f'Cluster {i + 1} size: {len(clusters[i])}')
