# Basic Algorithms and Data Structures

## Singly Linked List Operations (`linked_list_operations.py`)

This module provides a Python implementation of a singly linked list along with several advanced operations, including reversal, merge sort, and merging two sorted lists. It includes a test suite with assertions and performance measurements.

### Features:

*   **Singly Linked List Implementation:** Basic structure with `ListNode` and `LinkedList` classes.
*   **Operations:**
    *   **Reversal:** Reverses the linked list in-place.
    *   **Merge Sort:** Sorts the linked list using the merge sort algorithm.
    *   **Merge Two Sorted Lists:** Merges two already sorted linked lists into a single sorted list.
*   **Performance Measurement:** Includes a decorator (`measure_performance`) to time the execution of functions.
*   **Comprehensive Test Suite:** Contains tests for each operation with assertions to verify correctness and measures performance on various dataset sizes.

### Key Components:

*   **Classes:**
    *   `ListNode`: Data class for linked list nodes with value and next pointer.
    *   `LinkedList`: Main class with initialization and utility methods.

*   **Core Methods:**
    *   `_build_from_list(values)`: Constructs linked list from Python list.
    *   `to_list()`: Converts linked list back to Python list.
    *   `reverse_linked_list(head)`: Reverses the linked list starting from `head`.
    *   `merge_sort_linked_list(head)`: Sorts the linked list using merge sort.
    *   `merge_two_sorted_lists(list1, list2)`: Merges two sorted lists.

*   **Utility Functions:**
    *   `measure_performance(func, *args)`: Times function execution.
    *   `run_comprehensive_tests()`: Runs the test suite.

### Complexity Analysis Summary:

*   **Reverse Linked List:**
    *   Time Complexity: O(n)
    *   Space Complexity: O(1)
*   **Merge Sort:**
    *   Time Complexity: O(n log n)
    *   Space Complexity: O(log n) (due to recursion stack)
*   **Merge Two Sorted Lists:**
    *   Time Complexity: O(n + m)
    *   Space Complexity: O(1)

## Pythagorean Tree Fractal Generator (`pythagorean_tree.py`)

This script generates and visualizes Pythagorean tree fractals with various styling and coloring options. It includes a command-line interface for easy use, performance profiling, and internal tests.

### Features:

*   **Correct Geometric Construction:** Implements the proper method for generating child squares based on a right triangle. The branching angle is configurable.
*   **Command-Line Interface (CLI):** Use `argparse` to control fractal depth, branching angle, visualization style, color, output file, and headless mode.
*   **Visualization Styles:** Supports 'default' (filled squares with optional triangles), 'skeleton' (square outlines and twigs), and 'centerline' (connected apex-to-apex lines).
*   **Coloring Options:** Allows using Matplotlib colormaps for gradients (by depth or generation order) or a single color.
*   **Performance Analysis:** Includes a decorator for timing function execution and a separate mode (`--profile`) to run a complexity analysis.
*   **Internal Tests:** Contains built-in tests (`--test`) to verify the correctness of geometric calculations and fractal generation counts.

### CLI Usage Examples:

```bash
# Draw a level-5 tree with default settings
python pythagorean_tree.py --depth 5

# Draw a level-4 skeleton tree with maroon color
python pythagorean_tree.py --depth 4 --style skeleton --color maroon

# Draw a level-8 centerline tree with viridis colormap and larger figure size
python pythagorean_tree.py --depth 8 --style centerline --color viridis --figure-size 12.0

# Run internal tests
python pythagorean_tree.py --test

# Run complexity analysis up to a certain depth
python pythagorean_tree.py --profile --depth 8
```

### Key Components:

*   `Point`: Represents a 2D point.
*   `Square`: Represents a square defined by its vertices.
*   `PythagoreanTree`: The main class for generating and visualizing the fractal.
    *   `build_fractal()`: Generates the square elements and centerline segments recursively.
    *   `visualize()`: Handles plotting using Matplotlib based on selected style and colors.

### Complexity:

*   **Time Complexity:** O(2^n), where n is the recursion depth (due to the branching nature).
*   **Space Complexity:** O(2^n) for storing all square elements in memory (O(n) if using lazy rendering, but current implementation stores all). Additional space for centerline segments and twig bases.

## Dijkstra's Algorithm Implementation (`dijkstra_algorithm.py`)

This script provides an implementation of Dijkstra's algorithm for finding the shortest paths in a weighted graph. It utilizes a binary heap for optimization and includes functionalities for graph creation, shortest path computation (single-source and all-pairs), visualization, and performance analysis.

### Features:

*   **Graph Representation:** Uses `networkx.DiGraph` to represent the directed weighted graph.
*   **Binary Heap Optimization:** Implements Dijkstra's algorithm using `heapq` for efficient priority queue operations, resulting in improved time complexity.
*   **Shortest Path Computation:**
    *   Finds the shortest path between a single source and destination vertex (`dijkstra_heap`).
    *   Computes shortest paths from a single source to all other reachable vertices (`dijkstra_all_pairs`).
*   **Graph Visualization:** Plots the graph using `matplotlib` and `networkx`, with an option to highlight the shortest path.
*   **Performance Analysis:** Includes a function to measure the execution time of the algorithm on random graphs of varying sizes.
*   **Testing:** Contains assertion-based tests to verify the correctness of the algorithm's core functionalities.

### Key Components:

*   `DijkstraGraph` class:
    *   `__init__()`: Initializes an empty directed graph.
    *   `add_edge(source, target, weight)`: Adds a weighted edge to the graph, validating for non-negative weights.
    *   `create_sample_graph()`: Creates a predefined sample graph for demonstration and testing.
    *   `dijkstra_heap(src, dst, weight)`: Implements Dijkstra's algorithm to find the shortest path between two specific vertices.
    *   `dijkstra_all_pairs(src, weight)`: Computes shortest paths from a source to all reachable vertices.
    *   `plot_graph(highlighted_path, title, figsize)`: Visualizes the graph, optionally highlighting a given path.
    *   `measure_performance(src, dst, iterations)`: Measures the execution time of `dijkstra_heap` over multiple iterations.
*   `create_random_graph(num_vertices, edge_probability, max_weight)`: Generates a random weighted graph for performance testing.
*   Test functions (`test_dijkstra_basic`, `test_dijkstra_no_path`, `test_dijkstra_single_vertex`, `test_negative_weight`, `test_all_pairs`): Verify different aspects of the Dijkstra implementation.
*   `performance_analysis()`: Orchestrates performance testing on different graph sizes.
*   `main()`: Demonstrates the usage of the `DijkstraGraph` class, runs tests, performs analysis, and visualizes the sample graph.

### Complexity Analysis Summary:

*   **Dijkstra's Algorithm with Binary Heap:**
    *   Time Complexity: O((V + E) log V), where V is the number of vertices and E is the number of edges.
    *   Space Complexity: O(V) for distance and predecessor tables in the single-source algorithm. O(V^2) in the worst case for storing all-pairs paths in the result dictionary.

## Tree Visualization Modules

These modules provide functionality for visualizing binary trees, heaps, and tree traversal algorithms using NetworkX and Matplotlib.

### Common Dependencies:
*   `networkx`: Graph representation and visualization
*   `matplotlib`: Advanced plotting and visualization
*   `typing`: Type hints and sequence handling
*   `numpy`: Array operations and color value generation
*   `uuid`: Unique node identification
*   `collections.deque`: Efficient queue implementation
*   `time`: Performance measurements

### Binary Heap Visualization (`heap_visualization.py`)

This module provides the foundation for tree visualization using NetworkX and Matplotlib. It includes type-safe functions for heap operations, tree visualization, and testing capabilities.

### Features:

*   **Type-Safe Implementation:** Uses Python's type hints with proper handling of mutable and immutable sequences.
*   **Heap Operations:**
    *   **Build Heap:** Constructs a binary heap from an input array.
    *   **Heapify:** Maintains the heap property by recursively adjusting node positions.
    *   **Visualization:** Generates clear visual representations of the heap structure.
*   **Tree Operations:**
    *   **Node Creation:** Safe node instantiation with value checking.
    *   **Tree Building:** Converts arrays into binary tree structures.
*   **Error Handling:** Robust handling of edge cases including:
    *   Empty arrays
    *   Null node values
    *   Type validation for input sequences

### Key Functions:

*   `build_heap_from_array(arr)`: Builds a heap from an input sequence, with proper type handling.
*   `heapify(arr, length, root_idx)`: Maintains heap property by recursively adjusting nodes.
*   `create_node(val)`: Creates a binary tree node with type checking.
*   `build_tree(arr)`: Constructs a binary tree from an array with proper error handling.
*   `add_edges(graph, node, pos)`: Recursively adds edges to the NetworkX graph.
*   `visualize_heap(arr)`: Generates a visual representation of the heap using NetworkX.

### Complexity Analysis:

*   **Build Heap:**
    *   Time Complexity: O(n)
    *   Space Complexity: O(n)
*   **Heapify:**
    *   Time Complexity: O(log n)
    *   Space Complexity: O(1)
*   **Tree Building:**
    *   Time Complexity: O(n)
    *   Space Complexity: O(n)

### Usage Example:

```python
from heap_visualization import visualize_heap

# Create and visualize a heap
array = [4, 10, 3, 5, 1]
visualize_heap(array)  # Generates a visual representation of the heap
```

### Type Safety and Error Handling:

*   Input validation for array parameters
*   Proper handling of empty sequences
*   Null checks for node values
*   Clear type hints for function parameters and return values

### Tree Traversal Extension (`tree_traversal_viz.py`)

This extension of the heap visualization module adds support for visualizing tree traversal algorithms (DFS and BFS). It implements iterative approaches using stack and queue data structures, with color-coded visualization of traversal order.

### Features:

*   **Traversal Algorithms:**
    *   **Depth-First Search (DFS):**
        * Stack-based LIFO implementation
        * Explores depth-first: root → left subtree → right subtree
        * Memory efficient with stack size proportional to tree height
    *   **Breadth-First Search (BFS):**
        * Queue-based FIFO implementation
        * Explores level-by-level
        * Queue size proportional to tree width
*   **Visual Enhancements:**
    *   Color-coded traversal order using matplotlib's 'plasma' colormap
    *   Dark-to-light color progression showing visit sequence
    *   Interactive visualization with colorbar legend
*   **Comprehensive Testing:**
    *   Functional tests for traversal correctness
    *   Edge case handling (empty trees, single nodes)
    *   Performance analysis across different tree sizes
*   **Flexible Tree Creation:**
    *   Support for manual tree construction
    *   Conversion from heap array representation
    *   Sample tree generation for testing

### Key Functions:

*   `dfs(root)`: Performs iterative depth-first search traversal
*   `bfs(root)`: Performs iterative breadth-first search traversal
*   `visualize_traversal(root, traversal_type)`: Creates color-coded visualization
*   `array_to_heap_tree(heap_array)`: Converts array to tree structure
*   `generate_traversal_colors(num_nodes)`: Generates color scheme for visualization
*   `performance_analysis()`: Measures and compares algorithm performance

### Complexity Analysis:

*   **Depth-First Search (DFS):**
    *   Time Complexity: O(n) for visiting all nodes
    *   Space Complexity: O(h) where h is tree height
*   **Breadth-First Search (BFS):**
    *   Time Complexity: O(n) for visiting all nodes
    *   Space Complexity: O(w) where w is maximum tree width
*   **Visualization:**
    *   Time Complexity: O(n) for tree construction and rendering
    *   Space Complexity: O(n) for storing graph structure

### Usage Example:

```python
from tree_traversal_viz import create_sample_tree, visualize_traversal

# Create a sample binary tree
root = create_sample_tree()

# Visualize DFS traversal
dfs_nodes, dfs_time = visualize_traversal(root, "dfs")
print(f"DFS traversal order: {[node.val for node in dfs_nodes]}")

# Visualize BFS traversal
bfs_nodes, bfs_time = visualize_traversal(root, "bfs")
print(f"BFS traversal order: {[node.val for node in bfs_nodes]}")
```

### Implementation Details:

*   **Iterative Approach:** Uses explicit stack/queue instead of recursion
*   **Color Visualization:** Scientific colormap for intuitive traversal order
*   **Performance Monitoring:** Built-in execution time measurement
*   **Error Handling:** Robust handling of edge cases and invalid inputs
*   **Testing Framework:** Test suite with assertions

### Notes:

*   Memory efficiency through iterative implementations
*   Clear visualization of traversal patterns through color progression
*   Test coverage and performance analysis
*   Type-safe implementation with proper error handling
*   Flexible support for different tree structures
