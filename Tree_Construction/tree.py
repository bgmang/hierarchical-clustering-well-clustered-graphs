"""
This module contains the implementation of the Node and Tree base classes
"""

from Tree_Construction import div_constr
from Tree_Construction import agg_constr
import graph_methods as graph

# Certain types of trees considered
DIVISIVE_TREE_TYPES = ["degree", "random", "cheeger", "local_search"]
AGGLOMERATIVE_TREES_TYPES = ["average_linkage", "single_linkage", "complete_linkage"]


class Node:
    """
    Base class for Node objects.

    Attributes:
        number_of_vertices: The number of vertices in the subtree rooted at this node
        vertices: The list of vertices in the subtree rooted at this node
        cost: Dasgupta's cost value induced by the cut at this node
        left_child: The Node corresponding to the left child
        right_child: The Node corresponding to the right child
        id: The identifier of this node
        parent_size: The number of vertices in the subtree rooted at the parent of this node
    """

    def __init__(self):
        self.number_of_vertices = 0
        self.vertices = []
        self.cost = 0
        self.left_child = None
        self.right_child = None
        self.id = -1
        self.parent_size = 0

    def set_number_of_vertices(self, n):
        self.number_of_vertices = n

    def set_vertices(self, vertices):
        self.vertices = vertices

    def set_cost(self, cost):
        self.cost = cost

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def set_id(self, index):
        self.id = index

    def set_parent_size(self, parent_size):
        self.parent_size = parent_size

    def merge_node(self, node, new_id, G):
        """
        This method returns a new Node that is the union of the self node and another node given as parameter.

        :param node: The second node to be merged
        :param new_id: The id of the resulting merged node
        :param G: The underlying networkx graph, needed for computing the cost induced at the new node
        :return: A new_node that results by merging the two nodes
        """
        new_node = Node()
        new_node.set_number_of_vertices(self.number_of_vertices + node.number_of_vertices)
        new_node.set_vertices(self.vertices + node.vertices)
        new_node.set_cost(new_node.number_of_vertices * graph.cut_value(G, self.vertices, node.vertices))
        new_node.left_child = self
        new_node.right_child = node
        new_node.id = new_id
        return new_node


class Tree:
    """
    Base class for Tree objects.

    Attributes:
        graph: The underlying networkx graph for the tree
        root: The root Node of the tree
        tree_type: The type of the tree constructed
    """

    def __init__(self):
        self.graph = None
        self.root = None
        self.tree_type = ""

    def make_tree(self, G, tree_type):
        self.set_graph(G)
        self.root = Node()
        self.set_tree_type(tree_type)
        if tree_type in DIVISIVE_TREE_TYPES:
            T = div_constr.build_div_tree(G, tree_type)
            self.set_tree(T)
        elif tree_type in AGGLOMERATIVE_TREES_TYPES:
            T = agg_constr.build_agg_tree(G, tree_type)
            self.set_tree(T)
        else:
            raise Exception('Tree type not found')

    def get_tree_cost(self):
        return self.get_subtree_cost(self.root)

    def get_subtree_cost(self, node):
        if node.number_of_vertices <= 1:
            return 0
        else:
            return self.get_subtree_cost(node.left_child) \
                   + self.get_subtree_cost(node.right_child) + node.cost

    def get_critical_nodes(self):
        """
        This method returns the list of critical nodes associated to the tree

        :return: The list of critical nodes
        """
        critical_nodes = []
        total_volume = graph.volume(self.graph, self.graph.nodes())

        current_node = self.root
        current_node_volume = graph.volume(self.graph, current_node.vertices)

        # Travel down the tree as long as the the volume of the current_node is at least half the total volume of G
        while 2 * current_node_volume > total_volume:
            left_child, right_child = current_node.left_child, current_node.right_child

            # Ensure the left child has has larger volume
            if graph.volume(self.graph, left_child.vertices) < graph.volume(self.graph, right_child.vertices):
                left_child, right_child = right_child, left_child

            # Append the child of lower volume to the set of critical nodes
            critical_nodes.append(right_child)

            # Travel down to the child of larger volume
            current_node = left_child
            current_node_volume = graph.volume(self.graph, current_node.vertices)

        # Finally append the last node to the list of critical nodes
        critical_nodes.append(current_node)
        return critical_nodes

    def set_graph(self, G):
        self.graph = G

    def set_root(self, root):
        self.root = root

    def set_tree_type(self, tree_type):
        self.tree_type = tree_type

    def set_tree(self, T):
        self.set_graph(T.graph)
        self.set_root(T.root)
        self.set_tree_type(T.tree_type)

    def merge_tree(self, T, G):
        """
        This method returns a new Tree that is the union of the self tree and another tree T passed as an argument.
        :param T: The tree with which we join the current self tree
        :param G: The underlying graph G
        :return: A new tree resulting from the union of the self tree with T
        """

        new_root = self.root.merge_node(T.root, -1, G)

        new_tree = Tree()
        new_tree.set_graph(G)
        new_tree.set_root(new_root)
        new_tree.set_tree_type(self.tree_type)
        return new_tree
