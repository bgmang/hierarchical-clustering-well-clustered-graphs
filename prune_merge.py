"""
An implementation of the Algorithm PruneMerge computing an HC tree for a well-clustered graph
"""

import partition as partition
import graph_methods as graph
from Tree_Construction import tree


def prune_merge(G, k):
    """
    The implementation of the PruneMerge Algorithm on input parameters G and k.
    The Algorithm consists in three phases: Partition, Prune and Merge.
    In the Partition phase, we compute a partition of the graph G into subgraphs of high inner conductance
    In the Prune phase we prune (if necessary) the trees T_i corresponding to each induced subgraph G[P_i]
    and add the pruned nodes to the global collection of trees
    In the Merge phase we join together the collection of trees obtained from the Prune phase

    :param G: A networkx graph
    :param k: A target number of clusters
    :return: A Hierarchical Clustering tree T_PM of the graph G
    """

    # The Partition phase
    # We apply the partitioning algorithm to obtain the partitioning sets P_i with critical nodes S_i.
    clusters, critical_nodes = partition.compute_improved_partition(G, k)

    # The Prune Phase
    nodes_to_merge = prune(G, clusters, critical_nodes, k)

    # The Merge Phase
    T = merge(G, nodes_to_merge)

    return T


def prune(G, clusters, grouped_critical_nodes, k):
    """
    This method takes as input the list of clusters {P_i} out from the Partition Phase and prunes (if necessary)
    the corresponding trees T_i. Each pruned node, is added to a global list nodes_to_merge that will be merged in the
    next phase of the algorithm. Moreover, the root of each pruned tree T_i is also added to the list of nodes_to_merge


    :param G: A networkx graph
    :param clusters: A list containing all partition sets {P_i} obtained from the Partition phase
    :param grouped_critical_nodes: A list containing the critical nodes for each T_i grouped for every cluster
    :param k: The target number of clusters
    :return: A list of nodes_to_merge that will be used in the next Phase of the algorithm
    """

    # Initialise the parameters
    current_id = 1
    cluster_sizes = [0] * len(grouped_critical_nodes)

    # This dictionary maps a node id with the weight between that node and the other clusters.
    # Formally if N is a critical node in cluster P_i, the outer weight of N is exactly weight(N, V - P_i)
    outer_weight_critical_node = {}

    # Set ids and parent_size of nodes
    for i in range(len(grouped_critical_nodes)):
        cluster = grouped_critical_nodes[i]
        cluster_sizes[i] = sum(node.number_of_vertices for node in cluster)
        for node in cluster:
            # Set id
            node.set_id(current_id)
            current_id += 1

            # Set parent_size either to 2 * node.number_of_vertices or to the cluster size
            nb_vertices = node.number_of_vertices
            if (nb_vertices & (nb_vertices - 1) == 0) and nb_vertices != 0:
                node.set_parent_size(2 * nb_vertices)
            else:
                node.set_parent_size(cluster_sizes[i])

    # Get outer weight of critical nodes
    for i in range(len(grouped_critical_nodes)):
        nodes = grouped_critical_nodes[i]
        for node in nodes:
            outer_weight = 0
            for j in range(len(grouped_critical_nodes)):
                if j != i:
                    outer_weight += graph.cut_value(G, node.vertices, clusters[j])
            outer_weight_critical_node[node.id] = outer_weight

    # The actual pruning step

    # Initialise the list of nodes_to_merge as the empty list
    nodes_to_merge = []

    # Loop through all clusters
    for i in range(len(grouped_critical_nodes)):
        # Get the critical nodes of cluster P_i
        nodes = grouped_critical_nodes[i]

        # Compute the inner volumes of each node, i.e. vol_(G[P_i])(N)
        inner_volume_of_node = {}
        for node in nodes:
            inner_volume = graph.volume(G.subgraph(clusters[i]), node.vertices)
            inner_volume_of_node[node.id] = inner_volume

        # Get total outer weight, i.e. sum (weight(N, V - P_i))
        total_outer_weight = 0
        for node in nodes:
            total_outer_weight += outer_weight_critical_node[node.id]

        # Get total inner cost of nodes i.e. sum(|parent(N)| * vol(G[P_i], N))
        total_inner_cost = 0
        for node in nodes:
            total_inner_cost += node.parent_size * inner_volume_of_node[node.id]

        # Start Pruning!
        for node in nodes:
            unpruned_nodes = nodes[:]
            # Check if the "if"-condition in Line~7 of Algorithm PruneMerge is satisfied
            if G.number_of_nodes() * total_outer_weight <= 6 * (k+1) * total_inner_cost:
                # Reconstruct the pruned tree T_i from the remaining unpruned nodes
                T = merge(G, unpruned_nodes)
                # Add the root of the (pruned) tree T_i to the list of nodes_to_merge
                nodes_to_merge.append(T.root)
                break
            else:
                # Add the pruned node to the list of nodes_to_merge
                nodes_to_merge.append(node)

                # Update the total outer weight and total inner cost to accommodate for the pruned node
                total_outer_weight -= outer_weight_critical_node[node.id]
                total_inner_cost -= node.parent_size * inner_volume_of_node[node.id]

                # Prune the node
                unpruned_nodes.remove(node)

    return nodes_to_merge


def merge(G, nodes_to_merge):
    """
    This method merges the list of nodes_to_merge in a 'Caterpillar' fashion, having the node of largest size closest
    to the root. The method sorts the nodes in decreasing order of their sizes and constructs the tree recursively

    :param G: A networkx graph
    :param nodes_to_merge: A list of nodes to be merged in a 'Caterpillar' fashion
    :return: The Hierarchical Clustering 'Caterpillar' tree on the list of nodes_to_merge
    """

    # If there is only one node return the tree having this node as a root
    if len(nodes_to_merge) == 1:
        one_node_tree = tree.Tree()
        one_node_tree.make_tree(G.subgraph(nodes_to_merge[0].vertices), "degree")
        return one_node_tree
    else:
        # Sort the nodes in decreasing order of their sizes
        nodes_to_merge.sort(key=lambda x: x.number_of_vertices, reverse=True)

        # At the root level, the left child will be the node with the largest size
        left_child = merge(G, nodes_to_merge[:1])

        # The right child will be the subtree obtained by applying the construction recursively to all the other nodes
        right_child = merge(G, nodes_to_merge[1:])

        # The method returns the union of the previous two trees
        return left_child.merge_tree(right_child, G)
