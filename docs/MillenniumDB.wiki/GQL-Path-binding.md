
# Path binding

The path binding is a list that stores the variables bound to a graph element, like a node or an edge, and their values.

Each element in the path binding is called a **path step**. A path step contains:

* A variable that binds to an edge
* A set of variables that bind to a node
* An edge
* A node

Since a path is sequence of graph elements, alternating between nodes and edges, a path step contains an edge and the node it reaches. A path starts and ends with a node, so the first path step in a path binding will contain only the node binding.

The set of variables stores variables that are contained in consecutive node patterns, so they are bound to the same value.

For example, consider this graph pattern

```txt
(a)((b)-[e]->(c)){2}(d)
```

If we expand the repeated pattern, we have this pattern

```txt
(a)((b)-[e]->(c))((b)-[e]->(c))(d)
```

Then, the variables in a path binding will look like this. Each element of this list represents a path step.

<p align="center">
  <img alt="path_binding" src="https://raw.githubusercontent.com/wiki/gustavo-toro/MillenniumDB/assets/path_binding.svg" width="500" />
</p>

Here, `n0`, `n1` and `n2` are nodes and `e0` and `e1` are edges in the database.

Also, path bindings can be concatenated. when concatenating two bindings, the last step of the left path binding is merged with the first step of the right path binding. The node values in these steps must be equal. The node variables in this step are merged, while the edge variable and the edge value keep the values of the left path binding.

For example, if we concatenate these 2 path bindings

<p align="center">
  <img alt="path_binding_concat_a" src="https://raw.githubusercontent.com/wiki/gustavo-toro/MillenniumDB/assets/path_binding_concat_a.svg" width="350" />
</p>

We get a path binding that looks like this.

<p align="center">
  <img alt="path_binding_concat_b" src="https://raw.githubusercontent.com/wiki/gustavo-toro/MillenniumDB/assets/path_binding_concat_b.svg" width="500" />
</p>

## Path binding iterator

The path binding iterator is an interface that allows us to get path bindings. Classes that inherit from the the path binding iterator must implement these functions:

* `begin`: Prepares the path binding iterator.
* `set_left_boundary`: Sets the leftmost variable.
* `set_right_boundary`: Sets the rightmost variable.
* `next`: Returns a unique pointer to the next path binding.
* `reset`: Resets the iterator.
* `assign_nulls`: Assigns null to the variables set by the iterator.
* `assign_empty`: Assigns the value null to the variables set by the iterator.
* `print`: Writes a string representation of the iterator in the stream.

The `set_left_boundary` and `set_right_boundary` functions are used for assigning boundary variables, when a query has two consecutive node patterns. For example, in this pattern:

```txt
(a)->(b)(c)->(d)
```

We first find a binding for the variables in `(a)->(b)` and then we use the `set_left_boundary` function to assign the variable `c`, so that `b` and `c` are bound to the same node.

## Main path binding iterators

In this section, we explain the main path binding iterators used in the evaluation of a query.

### LinearPatternPath

The LinearPatternPath **generates the path binding from a binding**.

This iterator stores the variables in order they appear in the query. Its child iterator does not inherit from the path binding iterator, but from the binding iterator instead. When `next` is called, the LinearPatternPath takes the binding written by its child and generates a path binding to return.

### ExtendRight

The ExtendRight iterator generates a path binding from the concatenation of two path bindings from its left and right children.

In the `begin` function, this iterator calls its left child and stores its path binding. When `next` is called, then it will call the right iterator and it will concatenate the results.

Also, initially when the left path binding is stored, this iterator calls `set_left_boundary` on its right iterator, so that the last node value in the left path binding is the same as the first node value in the right path binding. This is needed for the concatenation to be valid.

There is another iterator, ExtendLeft, which is analogous to this iterator, but it first calls its right child and concatenates the result with the result of the left child.

### Repetition

The Repetition iterator generates a path binding by concatenating the path binding obtained from its child iterator with itself.

This iterator has the value of a node which is the start of the path. This value was passed through the `set_left_boundary` or the `set_right_boundary` function. When `begin` is called, this value is passed to its child and also it is stored as a path binding of size 1. This path binding is added to a queue in order to perform BFS.

When calling `next`, this iterator performs BFS over the graph using the path bindings of its child iterator. Before calling its child, it needs to set its left boundary or right boundary, so that the bindings can be concatenated.

### ListToBinding

This ListToBinding iterator adds the values of the path binding to the binding.

This iterator does not inherit from the path binding iterator, but from the binding iterator. It takes the values of the path and add them as is or as a list, depending on the degree of reference of the variables.

Also, this iterator sets the path variable in the binding if it is present in the query.
