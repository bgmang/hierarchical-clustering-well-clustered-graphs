"""
An implementation of the cheeger cut for networkx graphs.
"""
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import networkx as nx


def sweep_set(A, v_2, degrees):
    """
    Given the adjacency matrix of a graph, and the second eigenvalue of the laplacian matrix, use the sweep set
    algorithm to find a sparse cut.

    :param A: The adjacency matrix of the graph to use.
    :param v_2: The second eigenvector of the laplacian matrix of the graph
    :param degrees: a list with the degrees of each vertex in the graph
    :return: The set of vertices corresponding to the optimal cut
    """

    # Calculate n here once
    n = A.shape[0]
    # Keep track of the best cut so far
    best_cut_index = None
    best_conductance = None

    # Keep track of the size of the set and the cut weight to make computing the conductance
    # straightforward
    total_volume = np.sum(degrees)
    set_volume = 0
    set_size = 0
    cut_weight = 0

    # Normalise v_2 with the degrees of each vertex
    D = sp.sparse.diags(degrees, 0)
    v_2 = D.power(-(1/2)).dot(v_2)

    # First, sort the vertices based on their value in the second eigenvector
    sorted_vertices = [i for i, v in sorted(enumerate(v_2), key=(lambda x: x[1]))]

    # Keep track of which edges to add/subtract from the cut each time
    x = np.ones(n)

    # Loop through the vertices in the graph
    for (i, v) in enumerate(sorted_vertices[:-1]):
        # Update the set size and cut weight
        set_volume += degrees[v]
        set_size += 1

        # From now on, edges to this vertex will be removed from the cut at each iteration.
        x[v] = -1

        additional_weight = A[v, :].dot(x)
        cut_weight += additional_weight

        if cut_weight < 0:
            raise Exception('Something went wrong in sweep set: conducatance negative!')
        # Calculate the conductance
        if min(set_volume, total_volume - set_volume) == 0:
            this_conductance = 1
        else:
            this_conductance = cut_weight / min(set_volume, total_volume - set_volume)

        # Check whether this conductance is the best
        if best_conductance is None or this_conductance < best_conductance:
            best_cut_index = i
            best_conductance = this_conductance

    # Return the best cut
    return sorted_vertices[:best_cut_index+1]


def cheeger_cut(G):
    """
    Given a networkx graph G, find the cheeger cut.

    :param G: The graph on which to operate
    :return: A set containing the vertices on one side of the cheeger cut
    """
    if G.number_of_nodes() == 0:
        raise Exception(f'Cheeger cut: Graph should not be empty!')

    if nx.is_connected(G) is False:
        return list(list(nx.connected_components(G))[0])

    if G.number_of_nodes() == 1:
        return list(G.nodes())

    if G.number_of_nodes() == 2:
        return [list(G.nodes)[0]]

    # Compute the key graph matrices
    adjacency_matrix = nx.adjacency_matrix(G, weight='weight')
    laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight')
    graph_degrees = [t[1] for t in nx.degree(G, weight='weight')]

    # Compute the second smallest eigenvalue of the laplacian matrix
    eig_vals, eig_vecs = sp.sparse.linalg.eigsh(laplacian_matrix, which="SM", k=2)
    v_2 = eig_vecs[:, 1]

    # Perform the sweep set operation to find the sparsest cut
    S = sweep_set(adjacency_matrix, v_2, graph_degrees)
    nodes = list(G.nodes())
    return [nodes[i] for i in S]
