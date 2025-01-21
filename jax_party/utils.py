import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Sequence, TypeVar

PyTree = TypeVar("PyTree")


def tree_slice(tree: PyTree, i: chex.Numeric) -> PyTree:
    """
    Returns a slice of the tree where all leaves are mapped by x: x[i].
    """
    return jax.tree_util.tree_map(lambda x: x[i], tree)


def tree_add_element(tree: PyTree, i: chex.Numeric, element: PyTree) -> PyTree:
    """
    Sets one value of a tree along the batch axis. It is equivalent to
    ```Python
    for array_leaf in tree:
        array_leaf[i] = element[i]
    ```
    """
    new_tree: PyTree = jax.tree_util.tree_map(
        lambda array, value: array.at[i].set(value), tree, element
    )
    return new_tree
