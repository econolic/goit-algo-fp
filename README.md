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

### Key Functions:

*   `reverse_linked_list(head)`: Reverses the linked list starting from `head`.
*   `merge_sort_linked_list(head)`: Sorts the linked list starting from `head` using merge sort.
*   `merge_two_sorted_lists(list1, list2)`: Public interface for merging two sorted lists.
*   `measure_performance(func, *args)`: Decorator/function to measure execution time.
*   `run_comprehensive_tests()`: Executes the test suite.

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
