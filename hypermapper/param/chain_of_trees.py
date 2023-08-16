import random
from os import makedirs
import graphviz as gz
import numpy as np
import pickle
from typing import Optional, Dict, List, Any
import torch
from torch import Tensor

from hypermapper.param.parameters import Parameter
import itertools


class Node:
    """
    A node holds a value for a parameter. A path from the root to a leaf makes up a unique partial feasible configuration.
    """

    def __init__(
            self,
            parent: Any,
            value: Any,
            parameter_name: str,
            val_idx: int = None,
            probability: float = 1,
            prior_weighted_probability: float = 1,
    ):
        """

        Input:
            parent: parent node
            value: parameter value
            parameter_name: name of the parameter
            val_idx: the index of the value in the parameter's list of values
            probability: probability of the node without the prior
            prior_weighted_probability: probability of the node with the priors
        """
        self.parent = parent
        self.children = []
        self.children_dict = {}
        self.value = value  # "original" representation
        self.parameter_name = parameter_name
        self.val_idx = val_idx
        self.probability = probability
        self.prior_weighted_probability = prior_weighted_probability
        self.id = None

    def get_partial_configuration(self) -> torch.Tensor:
        """
        Get the partial configuration corresponding to the node.
        Returns it in the tree order, so will need to be reshuffled.
        """
        node = self
        config = []
        while (node is not None) and (node.parameter_name is not None):
            config.append(node.value)
            node = node.parent
        return torch.flip(torch.tensor(config), (0,))

    def propagate_probabilities(self, param_space):
        """
        Update the probabilities for the node's child parameters
        """
        if self.children:
            child_parameter = param_space.parameters[param_space.parameter_names.index(
                self.children[0].parameter_name
            )
            ]
            child_val_idxs = [child.val_idx for child in self.children]
            child_probability = self.probability / len(self.children)
            prior_weighted_probability_sum = np.sum(
                [child_parameter.distribution[val_idx] for val_idx in child_val_idxs]
            )
            child_prior_weighted_probabilities = [
                self.prior_weighted_probability
                * child_parameter.distribution[val_idx]
                / prior_weighted_probability_sum
                for val_idx in child_val_idxs
            ]
            for idx, child in enumerate(self.children):
                child.probability = child_probability
                child.prior_weighted_probability = child_prior_weighted_probabilities[
                    idx
                ]
                child.propagate_probabilities(param_space)

    def set_children_dict(self):
        for child in self.children:
            self.children_dict[child.value] = child
            child.set_children_dict()

    def get_probability(self):
        return self.probability

    def set_probability(self, probability):
        self.probability = probability

    def get_prior_weighted_probability(self):
        return self.prior_weighted_probability

    def set_prior_weighted_probability(self, probability):
        self.prior_weighted_probability = probability

    def add_child(self, child):
        self.children.append(child)

    def remove_child(self, child):
        self.children.remove(child)

    def get_children(self):
        return self.children

    def get_parent(self):
        return self.parent

    def get_value(self):
        return self.value

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id


class ChainOfTrees:
    """
    A chain of trees is a data structure for enumerating all feasible configurations. Each tree enumerates a group of co-dependent
    variables. The complete set of feasible configurations is then the Cartesian product of the feasible partial configurations
    of each tree.
    """

    def __init__(
        self,
        cot_order: List[int],
        all_parameters: bool
    ):
        """
        Input:
            - cot_order: order in which the variables come in the trees
            - all_parameters: whether the trees cover all the parameters or not
        """
        self.trees = []
        self.gaussian_means = None
        self.size = None
        self.dimension = len(cot_order)
        self.cot_order = cot_order  # order in which the variables come in the trees
        self.all_parameters = all_parameters
        if all_parameters:
            self.reverse_order = [cot_order.index(i) for i in range(self.dimension)]  # indices to reverse to the original parameter order

    def to_cot_order(self, configuration: torch.Tensor) -> torch.Tensor:
        """
        Transforms a tensor to the order used by the tree.
        """
        if configuration.dim() == 1:
            return configuration[self.cot_order]
        else:
            return configuration[:, self.cot_order]

    def to_original_order(self, configuration: torch.Tensor) -> torch.Tensor:
        """
        Transforms a tensor the original order.
        """
        if not self.all_parameters:
            raise Exception("ChainOfTrees.to_original_order doesn't work for incomplete trees. See over the code.")
        if configuration.dim() == 1:
            return configuration[self.reverse_order]
        else:
            return configuration[:, self.reverse_order]

    def add_tree(self, tree):
        self.trees.append(tree)

    def get_trees(self):
        return self.trees

    def get_random_configuration(self) -> Dict[str, Any]:
        """
        Samples a random configuration from the chain of trees.
        Returns:
            - config: a dict with the configuration

        """
        config = {}
        for tree in self.trees:
            randomNumber = random.randint(0, len(tree.leaves) - 1)
            node = tree.leaves[randomNumber]
            while (node is not None) and (node.parameter_name is not None):
                config[node.parameter_name] = node.value
                node = node.parent
        return config

    def sample(
            self,
            n_samples: int,
            sample_type: str,
            parameter_names: List[str],
            allow_repetitions: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Sampling feasible random configurations using the chain of trees.
        Input:
            - n_samples: number of configurations requested
            - sample_type: "uniform", "embedding", "using priors" decides the probability distribution for different configurations
            - parameter_names: an ordered list of the names of the parameters
            - allow_repetitions: whether to allow multiple identical configurations
        Returns:
            - tensor with the sampled configurations

        This is fast for allow_repetition = True, but significantly slower for allow_repetition = False
        This is technically also only pseudo random if it contains multiple trees and we don't allow repetitions

        """
        if not self.get_trees():
            return torch.Tensor()

        output = [[] for _ in range(n_samples)]
        ordered_names = []
        tree_lengths = [len(t.get_leaves()) for t in self.get_trees()]
        max_length_index = np.argmax(tree_lengths)

        # if n_samples <= the maximum number of leaves in any tree, then we cannot get any repetitions, and we don't need to ugly sample
        if (not allow_repetitions) and n_samples > tree_lengths[max_length_index]:
            return self._partitioned_sample(n_samples, sample_type, parameter_names)

        else:
            for tree_idx, tree in enumerate(self.trees):
                allow_repetitions_for_this_tree = (
                        allow_repetitions or tree_idx != max_length_index
                )
                if sample_type == "uniform":
                    pr = None
                elif sample_type == "embedding":
                    pr = [leaf.probability for leaf in tree.get_leaves()]
                elif sample_type == "using_priors":
                    pr = [leaf.prior_weighted_probability for leaf in tree.get_leaves()]
                else:
                    print(
                        f"incorrect sample type: {sample_type}. Expected on of uniform, embedding and using_priors."
                    )
                    exit(1)
                leaf_indices = np.random.choice(
                    np.arange(len(tree.get_leaves())),
                    size=n_samples,
                    replace=allow_repetitions_for_this_tree,
                    p=pr,
                )

                nodes = [tree.get_leaves()[idx] for idx in leaf_indices]
                while nodes[0].parent:
                    for i in range(n_samples):
                        output[i].append(nodes[i].value)
                    ordered_names.append(nodes[0].parameter_name)
                    nodes = [node.parent for node in nodes]
            output = torch.tensor(output)[:, torch.tensor([ordered_names.index(name) for name in parameter_names])]
            return output

    def _partitioned_sample(
            self,
            n_samples: int,
            sample_type: str,
            parameter_names: List[str],
    ) -> Tensor:
        """
        this method is a hack to deal with random sampling from a CoT, when we want many samples and no repetitions
        and have multiple trees.
        what we want to do is sample from the cartesian product of range(tree1.leaves)Xrange(tree2.leaves)X... without repetition,
        but I don't know how to do that efficiently
        Instead, we just sample the maximum number of samples that we can guarantee are unique by choosing one sample for
        each of the leaves in the tree with maximum number of leaves. Then we repeat until we have sufficient samples.
        """
        all_samples = torch.Tensor()
        iter = 0
        max_samples = np.max([len(t.get_leaves()) for t in self.get_trees()])
        while len(all_samples) < n_samples and iter < 1000:
            new_samples = torch.cat([x for x in self.sample(max_samples, sample_type, parameter_names, False).unsqueeze(0)])
            if not len(all_samples) == 0:
                bools = [not (all_samples == x).all(1).any() for x in new_samples]
                new_samples = new_samples[bools, :]
            all_samples = torch.cat((all_samples, new_samples), 0)
            iter += 1
        if iter == 999:
            print("Warning: found less than the required number of random samples.")
        return all_samples

    def write_to_pickle(self, file):
        with open(file, "wb") as f:
            pickle.dump((self.trees, self.gaussian_means), f)

    def read_from_pickle(self, file):
        with open(file, "rb") as f:
            self.trees, self.gaussian_means = pickle.load(f)

    def plot_trees(self, filepath=None):
        """
        Quick and dirty plotting of the chain of trees.
        Input:
            - filepath: path to save the plots to
        """
        if filepath is None:
            filepath = "chain_of_trees/"
        elif not filepath[-1] == "/":
            filepath += "/"
        makedirs(filepath, exist_ok=True)

        for idx, tree in enumerate(self.get_trees()):
            tree.set_probabilities()
            nodes, edges = tree.get_nodes_and_edges()
            tree_printer = gz.Graph()
            for node in reversed(nodes):
                id = str(node[0])
                label = f"{node[1]}\n{node[2]:.3f}" if node[1] is not None else ""
                tree_printer.node(id, label)
            for edge in reversed(edges):
                nodeA = str(edge[0])
                nodeB = str(edge[1])
                tree_printer.edge(nodeA, nodeB)
            filename = filepath + f"tree{idx}"
            tree_printer.render(filename, format="pdf", cleanup=True)

    def get_size(self) -> int:
        """
        Get the number of configurations in the chain of trees.
        """

        if self.size is None:
            size = 1
            for tree in self.trees:
                size *= len(tree.get_leaves())
            self.size = size
        return self.size

    def get_all_configurations(self) -> torch.Tensor:
        """
        Get a tensor with all feasible solutions
        """
        configurations = torch.zeros((self.get_size(), self.dimension))
        tree_partial_configurations = []
        for tree_idx, tree in enumerate(self.trees):
            tree_partial_configurations.append([leaf.get_partial_configuration() for leaf in tree.get_leaves()])
        cartesian_product = list(itertools.product(*tree_partial_configurations))
        for i, c in enumerate(cartesian_product):
            configurations[i, :] = torch.cat(c)
        return self.to_original_order(configurations)


class Tree:
    """
    A single tree for enumerating a group of co-dependent parameters.
    """

    def __init__(self):
        """
        Initialize an empty tree. Will be filled by create_tree in space.py.
        """
        self.leaves = []
        self.root = Node(None, None, None)
        self.nodes = None
        self.edges = None
        self.sparsity = 0
        self.depth = 0

    def add_leaf(self, leaf: Node):
        self.leaves.append(leaf)

    def get_root(self):
        return self.root

    def get_leaves(self):
        return self.leaves

    def remove_leaf(self, leaf: Node):
        """
        Removes a single feasible configuration and prunes the tree accordingly.
        Input:
            - leaf: the corresponding leaf to remove.
        """
        self.leaves.remove(leaf)
        done = False
        while not done:
            parent = leaf.parent
            parent.children.remove(leaf)
            self.nodes.remove(leaf)
            if len(parent.children) > 0:
                done = True
            else:
                leaf = parent

    def set_probabilities(self):
        """
        Sets the probability for embedding sampling.
        """
        stack = [self.root]
        while len(stack) > 0:
            current_node = stack.pop()
            children = current_node.get_children()
            stack += children
            for child in children:
                child.set_probability(current_node.get_probability() / len(children))

    def set_prior_weighted_probabilities(self, parameter_dict: Dict[str, Parameter]):
        """
        Sets the probability for "using priors" sampling
        """
        stack = [self.root]
        while len(stack) > 0:
            current_node = stack.pop()
            children = current_node.get_children()
            stack += children
            for child in children:
                child.set_prior_weighted_probability(
                    current_node.get_probability()
                    * parameter_dict[child.parameter_name].pdf(child.value)
                    / sum([parameter_dict[c.parameter_name].pdf(c.value) for c in children])
                )

    def get_nodes_and_edges(self):
        """
        Ranks the nodes and edges for the plotting function.
        """

        if (self.nodes is not None) and (self.edges is not None):
            return self.nodes, self.edges
        self.nodes = []
        self.edges = []
        node_id = 0
        stack = [self.root]
        while len(stack) > 0:
            current_node = stack.pop()
            current_node.set_id(node_id)
            self.nodes += [
                (
                    node_id,
                    current_node.get_value(),
                    current_node.get_probability(),
                    current_node,
                )
            ]
            stack += current_node.get_children()
            parent = current_node.get_parent()
            if parent is not None:
                self.edges += [(parent.get_id(), node_id)]
            node_id += 1
        return self.nodes, self.edges
