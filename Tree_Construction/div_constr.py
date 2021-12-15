"""
This module implements various top-down algorithms for constructing Hierarchical Clustering Trees
"""

import math
import random
from Cheeger_Cut import cheeger_cut
from Tree_Construction import tree
import graph_methods as graph

DIVISIVE_TREE_TYPES = ["degree", "random", "cheeger", "local_search"]


def get_largest_power_of_two_less_than(n):
    r = int(math.log(n, 2))
    if 2 ** r == n:
        return 2 ** (r - 1)
    return 2 ** r


def refine_cut_locally(G, cut):
    A = cut[0]
    B = cut[1]

    # get left child inner volume, right child inner volume
    A_volume = graph.inner_volume(G, A)
    B_volume = graph.inner_volume(G, B)

    refinement_is_found = True
    while refinement_is_found:
        refinement_is_found = False

        # Check refinement from A to B
        for v in A:
            weight_to_A = graph.cut_value(G, [v], A)
            weight_to_B = graph.cut_value(G, [v], B)

            if A_volume + (len(A) - 1) * weight_to_B > B_volume + (len(B) + 1) * weight_to_A:
                refinement_is_found = True
                A.remove(v)
                B.append(v)
                A_volume -= weight_to_A
                B_volume += weight_to_B
                break

        if not refinement_is_found:
            # Check refinement from B to A
            for u in B:
                weight_to_A = graph.cut_value(G, [u], A)
                weight_to_B = graph.cut_value(G, [u], B)

                if B_volume + (len(B) - 1) * weight_to_A > A_volume + (len(A) + 1) * weight_to_B:
                    refinement_is_found = True
                    A.append(u)
                    B.remove(u)
                    A_volume += weight_to_A
                    B_volume -= weight_to_B
                    break

    return A, B


def get_cut(G, cut_type):
    left_child_vertices = []
    right_child_vertices = []
    if G.number_of_nodes() == 2:
        left_child_vertices = [list(G.nodes())[0]]
        right_child_vertices = [list(G.nodes())[1]]

    elif cut_type == "random":
        for node in list(G.nodes()):
            if random.random() <= 0.5:
                left_child_vertices.append(node)
            else:
                right_child_vertices.append(node)

    elif cut_type == "degree":
        sorted_vertices = [node[0] for node in sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)]
        r = get_largest_power_of_two_less_than(G.number_of_nodes())
        left_child_vertices = sorted_vertices[:r]
        right_child_vertices = sorted_vertices[r:]

    elif cut_type == "cheeger":
        all_vertices = list(G.nodes())
        left_child_vertices = cheeger_cut.cheeger_cut(G)
        right_child_vertices = list(set(all_vertices) - set(left_child_vertices))

    elif cut_type == "local_search":
        random_cut = get_cut(G, "random")
        left_child_vertices, right_child_vertices = refine_cut_locally(G, random_cut)

    else:
        raise Exception("Cut type not found!")

    if len(left_child_vertices) == 0 or len(right_child_vertices) == 0:
        return get_cut(G, cut_type)

    return left_child_vertices, right_child_vertices


def build_div_tree(G, tree_type):
    if tree_type not in DIVISIVE_TREE_TYPES:
        raise Exception(f"The type of divisive tree is not found!")

    new_tree = tree.Tree()
    if G.number_of_nodes() == 0:
        raise Exception(f"The underlying graph should not be empty!")
    elif G.number_of_nodes() == 1:
        new_tree.set_graph(G)
        new_tree.root = tree.Node()
        new_tree.root.number_of_vertices = 1
        new_tree.root.set_vertices(list(G.nodes()))
        new_tree.set_tree_type(tree_type)
        return new_tree
    else:
        induced_cut = get_cut(G, tree_type)
        left_child = build_div_tree(G.subgraph(induced_cut[0]), tree_type)
        right_child = build_div_tree(G.subgraph(induced_cut[1]), tree_type)

        new_tree = left_child.merge_tree(right_child, G)

    return new_tree
