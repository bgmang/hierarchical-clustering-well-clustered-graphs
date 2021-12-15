"""
This module implements various classical bottom-up linkage algorithms for constructing Hierarchical Clustering Trees
"""

from Tree_Construction import tree
import graph_methods as graph

from queue import PriorityQueue

AGGLOMERATIVE_TREES_TYPES = ["average_linkage", "single_linkage", "complete_linkage"]


def get_id_pair_from_nodes(node1, node2):
    """
    This method returns a sorted pair of ids corresponding to two different nodes
    :param node1: The first Node object
    :param node2: The second Node object
    :return: A sorted pair (id1, id2) of the corresponding ids such that id1 < id2
    """

    id1, id2 = node1.id, node2.id
    if id1 < id2:
        return id1, id2
    return id2, id1


def get_minimum_weight(G, S, T):
    """
    Given a networkx graph G and two subsets of vertices S and T, this method returns the minimum weight of an edge e,
    having one endpoint in S and the other in T

    :param G: A networkx graph
    :param S: A subset of vertices in G
    :param T: A subset of vertices in G
    :return: The minimum weight of an edge crossing the cut (S, T)
    """

    min_weight = float('inf')
    for u in S:
        for v in T:
            if G.has_edge(u, v):
                if "weight" in G[u][v].keys():
                    min_weight = min(min_weight, G[u][v]["weight"])
                else:
                    min_weight = min(min_weight, 1)
    return min_weight


def get_maximum_weight(G, S, T):
    """
    Given a networkx graph G and two subsets of vertices S and T, this method returns the maximum weight of an edge e,
    having one endpoint in S and the other in T

    :param G: A networkx graph
    :param S: A subset of vertices in G
    :param T: A subset of vertices in G
    :return: The maximum weight of an edge crossing the cut (S, T)
    """

    max_weight = -1
    for u in S:
        for v in T:
            if G.has_edge(u, v):
                if "weight" in G[u][v].keys():
                    max_weight = max(max_weight, G[u][v]["weight"])
                else:
                    max_weight = max(max_weight, 1)
    return max_weight


def get_nodes_distance(G, node1, node2, merge_type):
    """
    Given two intermediate Nodes and a linkage type, this method returns the distance between the two nodes
    :param G: A networkx graph
    :param node1: The first Node considered
    :param node2: The second Node considered
    :param merge_type: The linkage type
    :return: The distance between the two nodes according to the linkage type
    """

    if merge_type == "average_linkage":
        return graph.cut_value(G, node1.vertices, node2.vertices) / (node1.number_of_vertices * node2.number_of_vertices)
    if merge_type == "single_linkage":
        return get_minimum_weight(G, node1.vertices, node2.vertices)
    if merge_type == "complete_linkage":
        return get_maximum_weight(G, node1.vertices, node2.vertices)


def initialise_dists(G, nodes, tree_type):
    """
    Given a list of nodes and a tree_type , this method initialises the pairwise distances between every pair of nodes
    in the given list and adds them to a PriorityQueue pq

    :param G: A networkx graph
    :param nodes: A list of initial nodes
    :param tree_type: The hierarchical clustering tree type considered
    :return: A pair (dists, pq), where dists is a dictionary mapping a pair of ids (id1, id2) to the distance
    between the nodes of the corresponding ids; and pq is a PriorityQueue containing all the pairwise distances
    """

    dists = {}
    pq = PriorityQueue()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Get the distance between nodes i and j and the pair of their ids
            dist_ij = get_nodes_distance(G, nodes[i], nodes[j], tree_type)
            ids = get_id_pair_from_nodes(nodes[i], nodes[j])

            # Update the dictionary dists and the Priority Queue pq
            dists[ids] = dist_ij
            pq.put((-dist_ij, ids))

    return dists, pq


def update_dists(nodes, A, B, C, dists, pq, tree_type):
    """
    This method updates the dictionary of pairwise distances dists as well as the Priority Queue of all distances pq.
    The update accommodates for the merge of the two nodes A and B in the resulting node C.

    :param nodes: A list of Nodes
    :param A: The first merged Node
    :param B: The second merged Node
    :param C: The resulting Node from the merge of A and B
    :param dists: A dictionary containing the pairwise distances between of the nodes
    :param pq: A Priority Queue containing the pairwise distances between the nodes
    :param tree_type: The linkage tree type
    :return: An updated pair (dists, pq)
    """

    # We loop through all remaining nodes and update the distances with respect to the newly created node C
    for i in range(len(nodes)):
        if tree_type == "average_linkage":
            new_dist = (A.number_of_vertices * dists[get_id_pair_from_nodes(A, nodes[i])] + B.number_of_vertices * dists[
                get_id_pair_from_nodes(B, nodes[i])]) / C.number_of_vertices
        elif tree_type == "single_linkage":
            new_dist = min(dists[get_id_pair_from_nodes(A, nodes[i])], dists[get_id_pair_from_nodes(B, nodes[i])])
        elif tree_type == "complete_linkage":
            new_dist = max(dists[get_id_pair_from_nodes(A, nodes[i])], dists[get_id_pair_from_nodes(B, nodes[i])])

        new_ids = get_id_pair_from_nodes(C, nodes[i])

        dists[new_ids] = new_dist
        pq.put((-new_dist, new_ids))

        del dists[get_id_pair_from_nodes(A, nodes[i])]
        del dists[get_id_pair_from_nodes(B, nodes[i])]
    return dists, pq


# Receives a list of nodes
def build_agg_tree_from_nodes(G, nodes, tree_type):
    """
    Given parameters G, nodes, tree_type, the method constructs and returns an HC tree of the graph G starting from the
    set of Nodes nodes.
    :param G: A networkx graph
    :param nodes: An initial list of nodes
    :param tree_type: The type of the constructed tree
    :return: A linkage HC tree of the underlying graph G
    """

    if tree_type not in AGGLOMERATIVE_TREES_TYPES:
        raise Exception('Agglomerative tree type not found!')

    # A map connecting each id to the corresponding node
    ids_to_nodes = {}

    # A map determining if a certain node has already been merged. This maps ids to booleans
    is_node_merged = {}

    # A variable keeping track of the current_id
    current_id = 0

    # Label the initial nodes
    for node in nodes:
        node.set_id(current_id)
        ids_to_nodes[current_id] = node
        is_node_merged[current_id] = False

        current_id += 1

    dists, pq = initialise_dists(G, nodes, tree_type)
    while len(nodes) > 1:
        # Get the two nodes to merge
        if pq.empty():
            raise Exception('priority queue of distances should not be empty!')
        (dist, ids) = pq.get()
        while (is_node_merged[ids[0]] is True) or (is_node_merged[ids[1]] is True):
            if pq.empty():
                raise Exception('priority queue of distances should not be empty!')
            (dist, ids) = pq.get()

        node1_to_merge, node2_to_merge = ids_to_nodes[ids[0]], ids_to_nodes[ids[1]]

        # Mark the two old nodes as being merged and the new node as being unmerged
        is_node_merged[ids[0]] = True
        is_node_merged[ids[1]] = True
        is_node_merged[current_id] = False

        # Merge the two nodes
        merged_node = node1_to_merge.merge_node(node2_to_merge, current_id, G)

        # Add the link from the id to the new node
        ids_to_nodes[current_id] = merged_node

        # Remove the merging nodes from the list of the current nodes
        nodes.remove(node1_to_merge)
        nodes.remove(node2_to_merge)

        # First Update the distances, then append the new node to the current list of nodes
        dists, pq = update_dists(nodes, node1_to_merge, node2_to_merge, merged_node, dists, pq, tree_type)

        nodes.append(merged_node)

        current_id += 1

    final_tree = tree.Tree()
    final_tree.set_graph(G)
    final_tree.set_root(nodes[0])
    final_tree.set_tree_type(tree_type)
    return final_tree


def build_agg_tree(G, tree_type):
    """
    Given a networkx graph G the method constructs and returns a linkage HC tree of type tree_type.

    :param G: A networkx graph
    :param tree_type: The type of the constructed tree
    :return: A linkage HC tree of the underlying graph G
    """

    nodes = []
    for vertex in list(G.nodes()):
        node = tree.Node()
        node.number_of_vertices = 1
        node.vertices = [vertex]
        nodes.append(node)

    return build_agg_tree_from_nodes(G, nodes, tree_type)
