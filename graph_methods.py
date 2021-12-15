"""
An implementation of various graph methods for networkx graphs.
"""
import networkx as nx
import scipy as sp


def complement(G, S):
    """
    Given a set of vertices S, return its complement in G,
    i.e. the list of vertices that are *not* in S.

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The complement of S in G
    """

    return [node for node in list(G.nodes()) if node not in S]


def cut_value(G, S, T):
    """
    Given two sets of vertices S and T, we compute and return the cut value between S and T,
    i.e. the sum of the weights of edges with one endpoint in S and the other in T,
    sometimes denoted as w(S, T).

    :param G: A networkx graph
    :param S: A list of vertices in G
    :param T: A list of vertices in G
    :return: The cut value between S and T in G
    """

    # Deal with the corner cases
    if nx.is_empty(G) or len(S) == 0 or len(T) == 0:
        return 0

    # cut_val stores the overall cut value
    cut_val = 0

    # If S and T have small sizes, we compute the cut by looping through S and T
    if G.number_of_nodes() ** 1.5 > len(S) * len(T):
        for u in S:
            for v in T:
                if G.has_edge(u, v) and 'weight' in G.edges[u, v]:
                    cut_val += G[u][v]['weight']
                elif G.has_edge(u, v):
                    cut_val += 1.0
    # If S or T has large size, we compute the cut by looping through the edges in G
    else:
        vertices_in_S_or_T = {}
        for vertex in S:
            vertices_in_S_or_T[vertex] = 'S'
        for vertex in T:
            vertices_in_S_or_T[vertex] = 'T'
        for edge in list(G.edges()):
            if edge[0] in vertices_in_S_or_T and edge[1] in vertices_in_S_or_T:
                if vertices_in_S_or_T[edge[0]] != vertices_in_S_or_T[edge[1]]:
                    cut_val += G[edge[0]][edge[1]]['weight']
    return cut_val


def volume(G, S):
    """
    Given a set of vertices S in G, we compute and return its volume in G,
    i.e. the sum of the degrees of vertices in S.

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The volume of S in G
    """

    if nx.is_empty(G) or len(S) == 0:
        return 0
    return nx.volume(G, S, weight='weight')


def inner_volume(G, S):
    """
    Given a set of vertices S in G, we compute and return its inner volume,
    i.e. twice the sum of weights of edges with both endpoints in S

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The volume of S in G, i.e. the sum of the degrees of vertices in S
    """

    # in_vol stores the overall inner volume of S
    in_vol = 0

    # Loop through all pairs of vertices u, v in S and add the weight of the edge (u, v)
    for u in S:
        for v in S:
            if u != v and G.has_edge(u, v):
                in_vol += G[u][v]['weight']
    return in_vol


def conductance(G, S):
    """
    Given a set of vertices S in G, we compute and return its conductance in G.
    The conductance of S in G is defined as the ratio w(S, Sc)/vol(S), where
    w(S, Sc) is the cut value between S and its complement Sc,
    and vol(S) is the volume of S in G

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The conductance of S in G
    """

    vol = volume(G, S)

    # Corner cases if S is empty or has zero volume, by convention the conductance is 1.
    if len(S) == 0 or vol == 0:
        return 1

    S_complement = complement(G, S)
    return cut_value(G, S, S_complement) / vol


def get_smallest_eigs(G, k):
    """
    Given a parameter k, we compute and return the k smallest eigenvalues
    of the normalised Laplacian matrix of G

    :param G: A networkx graph
    :param k: A number of desired eigenvalues
    :return: A sorted list of the k smallest eigenvalues of the normalised Laplacian of G
    """

    # Corner case if G is empty all eigenvalues are 0.
    if nx.is_empty(G):
        return [0] * k
    # Corner case k should be smaller than the number of vertices in G.
    elif k > G.number_of_nodes():
        raise Exception(f'{k} eigenvalues requested, but G has only {G.number_of_nodes()} nodes')

    laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight')
    eigs = sp.sparse.linalg.eigsh(laplacian_matrix, k=k, which='SM', return_eigenvectors=False)
    eigs.sort()

    return eigs
