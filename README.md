# Basic Algorithms and Data Structures

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

A Python implementation of fundamental algorithms and data structures, featuring practical applications in business optimization and visualization.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation & Quick Start](#installation--quick-start)
4. [Core Modules](#core-modules)
   - [Singly Linked List Operations](#singly-linked-list-operations-linked_list_operationspy)
      - [Features](#features)
      - [Key Components](#key-components)
      - [Complexity Analysis](#complexity-analysis-summary)
   - [Pythagorean Tree Fractal Generator](#pythagorean-tree-fractal-generator-pythagorean_treepy)
      - [Features](#features-1)
      - [CLI Usage Examples](#cli-usage-examples)
      - [Key Components](#key-components-1)
      - [Complexity](#complexity)
   - [Dijkstra's Algorithm Implementation](#dijkstras-algorithm-implementation-dijkstra_algorithmpy)
      - [Features](#features-2)
      - [Key Components](#key-components-2)
      - [Complexity Analysis](#complexity-analysis-summary-1)
   - [Tree Visualization Modules](#tree-visualization-modules)
      - [Common Dependencies](#common-dependencies)
      - [Binary Heap Visualization](#1-binary-heap-visualization-heap_visualizationpy)
      - [Tree Traversal Extension](#2-tree-traversal-extension-tree_traversal_vizpy)
   - [Food Selection Optimization](#food-selection-optimization-greed_dp_algorithmspy)
      - [Features](#features-3)
      - [Key Components](#key-components-3)
      - [Complexity Analysis](#complexity-analysis-summary-2)
   - [Monte Carlo Dice Simulation](#monte-carlo-dice-simulation-monte_carlo_dicepy)
      - [Features](#features-4)
      - [Key Components](#key-components-4)
      - [Complexity Analysis](#complexity-analysis-summary-3)
   - [ADDITIONAL: Restaurant Supply Chain Optimizer](#additional-restaurant-supply-chain-optimizer-restaurant_optimizerpy)
      - [Features](#features-5)
      - [Project Structure](#project-structure-and-file-organization)
      - [Core Components](#core-components)
      - [Interactive Notebook Guide](#interactive-notebook-guide-restaurant_optimizer_notebookipynb)
         - [Key Features](#key-features)
         - [Notebook Sections](#notebook-sections)
         - [Usage](#usage)
         - [Tips for Interactive Learning](#tips-for-interactive-learning)

## Project Overview

This project implements various algorithms and data structures with real-world applications, from basic data structures to complex business optimization systems. Each module is thoroughly documented, tested, and includes performance analysis.

## Project Structure

The project is organized with clear separation of concerns:

1. **Core Implementation Files:**
   * Implementation code for each algorithm/data structure
   * Focused purely on functionality
   * Can be imported and used by other projects
   * Maintains clean, modular code structure

2. **Test Files:**
   * Separate test suites for each module
   * Enable test-driven development
   * Can run tests independently
   * Make it easier to maintain test coverage

3. **Demo Scripts:**
   * Provide real-world usage examples
   * Serve as documentation through examples
   * Can be run independently to showcase features
   * Help new users understand the system

4. **Jupyter Notebook (`restaurant_optimizer_notebook.ipynb`):**
    * An interactive environment for step-by-step execution, experimentation, and visualization.

This separation provides several advantages:

*   **Modularity:** Clear boundaries between different concerns (implementation, testing, demo, notebook)
*   **Maintainability:** Easier to update, test, and document each component
*   **Usability:** Users can quickly start with demos or notebooks, and refer to core files as needed
*   **Collaboration:** Multiple contributors can work on different aspects simultaneously

## Installation & Quick Start

### Prerequisites
* Python 3.12 or higher
* pip package manager

### Installation

1. Clone the repository:
```powershell
git clone https://github.com/econolic/goit-algo-fp.git
cd goit-algo-fp
```

2. Install required dependencies:
```powershell
pip install -r requirements.txt
```

### Quick Start Examples

1. Run the restaurant optimization demo:
```powershell
python demo_restaurant_optimizer.py
```

2. Generate a Pythagorean tree fractal:
```powershell
python pythagorean_tree.py --depth 5
```

3. Run Monte Carlo dice simulation:
```powershell
python monte_carlo_dice.py
```

### Dependencies

This project requires the following key packages:

* **Visualization:**
  * `matplotlib>=3.8.2` - Advanced plotting for fractals, graphs, and statistical visualizations
  * `networkx>=3.2.1` - Graph operations and visualization
* **Numerical Computing:**
  * `numpy>=1.26.3` - Array operations and random number generation
* **Development:**
  * `typing-extensions>=4.9.0` - Enhanced type hints
  * `pytest>=8.0.0` - Testing framework

All dependencies can be installed using:
```powershell
pip install -r requirements.txt
```

[Back to Top](#basic-algorithms-and-data-structures)

## Core Modules

### Singly Linked List Operations (`linked_list_operations.py`)

This module provides a Python implementation of a singly linked list along with several advanced operations, including reversal, merge sort, and merging two sorted lists. It includes a test suite with assertions and performance measurements.

#### Features:

*   **Singly Linked List Implementation:** Basic structure with `ListNode` and `LinkedList` classes.
*   **Operations:**
    *   **Reversal:** Reverses the linked list in-place.
    *   **Merge Sort:** Sorts the linked list using the merge sort algorithm.
    *   **Merge Two Sorted Lists:** Merges two already sorted linked lists into a single sorted list.
*   **Performance Measurement:** Includes a decorator (`measure_performance`) to time the execution of functions.
*   **Test Suite:** Contains tests for each operation with assertions to verify correctness and measures performance on various dataset sizes.

#### Key Components:

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

#### Complexity Analysis Summary:

*   **Reverse Linked List:**
    *   Time Complexity: $O(n)$
    *   Space Complexity: $O(1)$
*   **Merge Sort:**
    *   Time Complexity: $O(n log n)$
    *   Space Complexity: $O(log n)$ (due to recursion stack)
*   **Merge Two Sorted Lists:**
    *   Time Complexity: $O(n + m)$
    *   Space Complexity: $O(1)$

### Pythagorean Tree Fractal Generator (`pythagorean_tree.py`)

This script generates and visualizes Pythagorean tree fractals with various styling and coloring options. It includes a command-line interface for easy use, performance profiling, and internal tests.

#### Features:

*   **Correct Geometric Construction:** Implements the proper method for generating child squares based on a right triangle. The branching angle is configurable.
*   **Command-Line Interface (CLI):** Use `argparse` to control fractal depth, branching angle, visualization style, color, output file, and headless mode.
*   **Visualization Styles:** Supports 'default' (filled squares with optional triangles), 'skeleton' (square outlines and twigs), and 'centerline' (connected apex-to-apex lines).
*   **Coloring Options:** Allows using Matplotlib colormaps for gradients (by depth or generation order) or a single color.
*   **Performance Analysis:** Includes a decorator for timing function execution and a separate mode (`--profile`) to run a complexity analysis.
*   **Internal Tests:** Contains built-in tests (`--test`) to verify the correctness of geometric calculations and fractal generation counts.

#### CLI Usage Examples:

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

#### Key Components:

*   `Point`: Represents a 2D point.
*   `Square`: Represents a square defined by its vertices.
*   `PythagoreanTree`: The main class for generating and visualizing the fractal.
    *   `build_fractal()`: Generates the square elements and centerline segments recursively.
    *   `visualize()`: Handles plotting using Matplotlib based on selected style and colors.

#### Complexity:

*   **Time Complexity:** $O(2^n)$, where n is the recursion depth (due to the branching nature).
*   **Space Complexity:** $O(2^n)$ for storing all square elements in memory ($O(n)$ if using lazy rendering, but current implementation stores all). Additional space for centerline segments and twig bases.

### Dijkstra's Algorithm Implementation (`dijkstra_algorithm.py`)

This script provides an implementation of Dijkstra's algorithm for finding the shortest paths in a weighted graph. It utilizes a binary heap for optimization and includes functionalities for graph creation, shortest path computation (single-source and all-pairs), visualization, and performance analysis.

#### Features:

*   **Graph Representation:** Uses `networkx.DiGraph` to represent the directed weighted graph.
*   **Binary Heap Optimization:** Implements Dijkstra's algorithm using `heapq` for efficient priority queue operations, resulting in improved time complexity.
*   **Shortest Path Computation:**
    *   Finds the shortest path between a single source and destination vertex (`dijkstra_heap`).
    *   Computes shortest paths from a single source to all other reachable vertices (`dijkstra_all_pairs`).
*   **Graph Visualization:** Plots the graph using `matplotlib` and `networkx`, with an option to highlight the shortest path.
*   **Performance Analysis:** Includes a function to measure the execution time of the algorithm on random graphs of varying sizes.
*   **Testing:** Contains assertion-based tests to verify the correctness of the algorithm's core functionalities.

#### Key Components:

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

#### Complexity Analysis Summary:

*   **Dijkstra's Algorithm with Binary Heap:**
    *   Time Complexity: $O((V + E) log V)$, where V is the number of vertices and E is the number of edges.
    *   Space Complexity: $O(V)$ for distance and predecessor tables in the single-source algorithm. $O(V^2)$ in the worst case for storing all-pairs paths in the result dictionary.

### Tree Visualization Modules

These modules provide functionality for visualizing binary trees, heaps, and tree traversal algorithms using NetworkX and Matplotlib.

#### Common Dependencies:
*   `networkx`: Graph representation and visualization
*   `matplotlib`: Advanced plotting and visualization
*   `typing`: Type hints and sequence handling
*   `numpy`: Array operations and color value generation
*   `uuid`: Unique node identification
*   `collections.deque`: Efficient queue implementation
*   `time`: Performance measurements

## 1. Binary Heap Visualization (`heap_visualization.py`)

This module provides the foundation for tree visualization using NetworkX and Matplotlib. It includes type-safe functions for heap operations, tree visualization, and testing capabilities.

#### Features:

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

#### Key Functions:

*   `build_heap_from_array(arr)`: Builds a heap from an input sequence, with proper type handling.
*   `heapify(arr, length, root_idx)`: Maintains heap property by recursively adjusting nodes.
*   `create_node(val)`: Creates a binary tree node with type checking.
*   `build_tree(arr)`: Constructs a binary tree from an array with proper error handling.
*   `add_edges(graph, node, pos)`: Recursively adds edges to the NetworkX graph.
*   `visualize_heap(arr)`: Generates a visual representation of the heap using NetworkX.

#### Complexity Analysis:

*   **Build Heap:**
    *   Time Complexity: $O(n)$
    *   Space Complexity: $O(n)$
*   **Heapify:**
    *   Time Complexity: $O(log n)$
    *   Space Complexity: $O(1)$
*   **Tree Building:**
    *   Time Complexity: $O(n)$
    *   Space Complexity: $O(n)$

#### Usage Example:

```python
from heap_visualization import visualize_heap

# Create and visualize a heap
array = [4, 10, 3, 5, 1]
visualize_heap(array)  # Generates a visual representation of the heap
```

#### Type Safety and Error Handling:

*   Input validation for array parameters
*   Proper handling of empty sequences
*   Null checks for node values
*   Clear type hints for function parameters and return values

## 2. Tree Traversal Extension (`tree_traversal_viz.py`)

This extension of the heap visualization module adds support for visualizing tree traversal algorithms (DFS and BFS). It implements iterative approaches using stack and queue data structures, with color-coded visualization of traversal order.

#### Features:

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
*   **Testing:**
    *   Functional tests for traversal correctness
    *   Edge case handling (empty trees, single nodes)
    *   Performance analysis across different tree sizes
*   **Flexible Tree Creation:**
    *   Support for manual tree construction
    *   Conversion from heap array representation
    *   Sample tree generation for testing

#### Key Functions:

*   `dfs(root)`: Performs iterative depth-first search traversal
*   `bfs(root)`: Performs iterative breadth-first search traversal
*   `visualize_traversal(root, traversal_type)`: Creates color-coded visualization
*   `array_to_heap_tree(heap_array)`: Converts array to tree structure
*   `generate_traversal_colors(num_nodes)`: Generates color scheme for visualization
*   `performance_analysis()`: Measures and compares algorithm performance

#### Complexity Analysis:

*   **Depth-First Search (DFS):**
    *   Time Complexity: $O(n)$ for visiting all nodes
    *   Space Complexity: $O(h)$ where $h$ is tree height
*   **Breadth-First Search (BFS):**
    *   Time Complexity: $O(n)$ for visiting all nodes
    *   Space Complexity: $O(w)$ where $w$ is maximum tree width
*   **Visualization:**
    *   Time Complexity: $O(n)$ for tree construction and rendering
    *   Space Complexity: $O(n)$ for storing graph structure

#### Usage Example:

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

#### Implementation Details:

*   **Iterative Approach:** Uses explicit stack/queue instead of recursion
*   **Color Visualization:** Scientific colormap for intuitive traversal order
*   **Performance Monitoring:** Built-in execution time measurement
*   **Error Handling:** Robust handling of edge cases and invalid inputs
*   **Testing Framework:** Test suite with assertions

#### Notes:

*   Memory efficiency through iterative implementations
*   Clear visualization of traversal patterns through color progression
*   Test coverage and performance analysis
*   Type-safe implementation with proper error handling
*   Flexible support for different tree structures

### Food Selection Optimization (`greed_dp_algorithms.py`)

This module implements and compares two algorithmic approaches to solve the knapsack problem variant for food selection optimization: **a greedy algorithm** based on efficiency ratios and **a dynamic programming solution** for finding globally optimal results. It includes performance analysis and testing capabilities.

#### Features:

*   **Type-Safe Implementation:**
    *   Uses `TypedDict` for food item properties
    *   Implements `dataclass` for optimization results
    *   Robust input validation and error handling
*   **Dual Algorithm Implementation:**
    *   Greedy algorithm using calorie-to-cost efficiency
    *   Dynamic programming for optimal solution
*   **Performance Analysis:**
    *   Execution time measurement using `time.perf_counter()`
    *   Comparative analysis across different budget scenarios
    *   Memory and time complexity documentation
*   **Testing:**
    *   Test suite with edge case coverage
    *   Assertions for correctness verification
    *   Budget constraint validation

#### Key Components:

*   **Data Structures:**
    *   `FoodItem`: TypedDict for food properties (cost, calories)
    *   `OptimizationResult`: Dataclass for algorithm results
*   **Core Functions:**
    *   `greedy_algorithm(items, budget)`: Implements efficiency-based selection
    *   `dynamic_programming(items, budget)`: Implements optimal solution search
    *   `measure_performance(func, *args)`: Times function execution
    *   `compare_algorithms(items, budget)`: Analyzes both approaches
*   **Utility Functions:**
    *   `validate_inputs(items, budget)`: Ensures valid input parameters
    *   `test_algorithms()`: Runs test suite

#### Complexity Analysis Summary:

*   **Greedy Algorithm:**
    *   Time Complexity: $O(n log n)$ due to sorting
    *   Space Complexity: $O(n)$ for efficiency calculations
    *   Advantage: Better performance for large datasets
*   **Dynamic Programming:**
    *   Time Complexity: $O(n × budget)$ where $n$ is number of items
    *   Space Complexity: $O(n × budget)$ for DP table
    *   Advantage: Guarantees optimal solution

### Monte Carlo Dice Simulation (`monte_carlo_dice.py`)

This module implements a Monte Carlo simulation for estimating dice rolling probabilities and comparing them with analytical results. It provides extensive visualization, statistical analysis, and performance measurement capabilities, making it suitable for both educational purposes and practical probability estimation tasks.

#### Features:

*   **Monte Carlo Simulation:**
    *   Configurable number of dice and sides
    *   Millions of iterations per second
    *   Convergence analysis showing $O(\frac{1}{\sqrt{n}})$ error reduction
*   **Statistical Analysis:**
    *   Detailed probability comparisons
    *   Absolute and relative error calculations
    *   Mean error analysis across all outcomes
*   **Visualization:**
    *   Comparison bar charts for Monte Carlo vs analytical results
    *   Convergence analysis plots
    *   Interactive probability distribution displays
*   **Theoretical Foundation:**
    *   Based on classical probability theory
    *   Uses Cartesian product for outcome generation
    *   Applies equiprobability principle
*   **Testing Framework:**
    *   Test suite
    *   Edge case coverage
    *   Statistical validation tests

#### Key Components:

*   **Simulation Functions:**
    *   `monte_carlo_simulation(num_simulations)`: Performs basic two-dice simulation
    *   `simulate_dice_rolls(n_dice, n_sides, iterations)`: Performs generalized Monte Carlo simulation
    *   `dice_sum_pmf(n, m)`: Computes theoretical probabilities for n m-sided dice
    *   `analyze_convergence(num_simulations_list)`: Studies error reduction with sample size
*   **Visualization Tools:**
    *   `plot_probability_comparison(monte_carlo_probs, analytical_probs)`: Creates comparative bar charts
    *   `create_comparison_table(monte_carlo_probs, analytical_probs)`: Generates detailed probability tables
*   **Core Functions:**
    *   `roll_single_die()`: Basic die roll implementation
    *   `roll_two_dice()`: Rolls two dice simultaneously
    *   `calculate_dice_sum(die1, die2)`: Computes sum of dice values
    *   `calculate_probabilities(sum_counts, total_simulations)`: Converts counts to probabilities

#### Complexity Analysis Summary:

*   **Monte Carlo Simulation:**
    *   Time Complexity: $O(n)$ for $n$ iterations
    *   Space Complexity: $O(1)$ for fixed number of possible sums
    *   Performance: ~1 million iterations per second
*   **Analytical Calculation:**
    *   Time Complexity: $O(m^n)$ for $m$ sides and n dice
    *   Space Complexity: $O(m^n)$ for storing all combinations
    *   Optimization: $O(1)$ for standard two-dice case
*   **Convergence Properties:**
    *   Error Reduction: $O(\frac{1}{\sqrt{n}})$ with number of iterations
    *   Typical Accuracy: < 0.001 mean absolute error at 1M iterations
    *   Validation: Matches theoretical probabilities within 0.01%

#### Usage Example:

```python
from monte_carlo_dice import simulate_dice_rolls, plot_results

# Run simulation with 2 dice, 6 sides each, 1 million iterations
results = simulate_dice_rolls(n_dice=2, n_sides=6, iterations=1_000_000)

# Plot comparison with theoretical probabilities
plot_results(results)

# Analyze convergence
convergence = analyze_convergence([1000, 10000, 100000, 1000000])
print(f"Mean absolute error at 1M iterations: {convergence[-1]:.6f}")
```

### ADDITIONAL: Restaurant Supply Chain Optimizer (`restaurant_optimizer.py`)

**Quick Links:**
- [Features](#features)
- [Project Structure](#project-structure-and-file-organization)
- [Demo Script](#demo-script-demo_restaurant_optimizerpy)
- [Test Suite](#test-suite-test_restaurant_optimizerpy)
- [Complexity Analysis](#complexity-analysis-summary)
- [Real-World Benefits](#real-world-benefits)
- [Back to Top](#basic-algorithms-and-data-structures)

This module demonstrates the practical application of various algorithms in a real-world business context. It implements a complete restaurant chain optimization system that combines route planning, menu optimization, inventory management, demand forecasting, and supply chain visualization.

#### Features:

*   **Integrated Solution:**
    *   Combines multiple algorithms for business optimization
    *   Real-world application of theoretical concepts
    *   Practical visualization and reporting tools
*   **Route Optimization:**
    *   Uses Dijkstra's algorithm for optimal delivery route planning
    *   Achieves efficient routes with demonstrated cost savings
    *   Example: Downtown → Harbor View route optimized to 2.13 units vs 5.67 units indirect route
*   **Menu and Inventory:**
    *   Applies knapsack optimization for budget-constrained menu planning
    *   Uses min-heap structure for priority-based inventory management
    *   Supports multiple budget levels ($800-$1200) with optimal item selection
*   **Demand Analysis:**
    *   Monte Carlo simulation with location-specific modifiers
    *   Statistical analysis showing item popularity patterns
    *   Example: Pizza demand varies from 127 units (Downtown) to 82 units (Harbor)
*   **Visualization:**
    *   Interactive supply chain network visualization
    *   Priority-based inventory queue representation
    *   Performance metrics and reports

#### Project Structure and File Organization

The restaurant optimization system is organized into three separate files, following software development best practices:

1. **Core Implementation (`restaurant_optimizer.py`):**
   * Contains the actual implementation code
   * Focused purely on functionality
   * Can be imported and used by other projects
   * Maintains clean, modular code structure

2. **Test Suite (`test_restaurant_optimizer.py`):**
   * Separates testing from implementation
   * Enables test-driven development
   * Can run tests independently
   * Makes it easier to maintain test coverage
   * Follows testing best practices

3. **Demo Script (`demo_restaurant_optimizer.py`):**
   * Provides a walkthrough of the optimization process
   * Demonstrates loading data, running algorithms, and visualizing results
   * Serves as a practical example for users

4. **Jupyter Notebook (`restaurant_optimizer_notebook.ipynb`):**
    * An interactive environment for step-by-step execution, experimentation, and visualization.

This separation provides several advantages:

*   **Modularity:** Clear boundaries between different concerns (implementation, testing, demo, notebook)
*   **Maintainability:** Easier to update, test, and document each component
*   **Usability:** Users can quickly start with demos or notebooks, and refer to core files as needed
*   **Collaboration:** Multiple contributors can work on different aspects simultaneously

#### Core Components:

*   **Data Classes:**
    *   `Restaurant`: Represents restaurant locations with inventory and operational details.
    *   `Order`: Represents customer order details with tracking information.
    *   `RestaurantOptimizer`: Main class for managing the optimization process with methods:
        *   `setup_sample_data()`: Initializes sample restaurants and delivery routes.
        *   `optimize_delivery_routes(start_restaurant_id)`: Finds optimal delivery routes using Dijkstra's algorithm.
        *   `plan_daily_menu(restaurant_id)`: Optimizes menu selection using dynamic programming.
        *   `simulate_daily_demand(restaurant_id, days)`: Performs Monte Carlo simulation for demand forecasting.
        *   `update_inventory(restaurant_id, demand)`: Manages inventory with min-heap priority queue.
        *   `process_order(restaurant_id, items_ordered)`: Handles order processing and validation.
        *   `visualize_supply_chain()`: Creates interactive network visualizations.

*   **Helper Methods:**
    *   `rebuild_delivery_graph()`: Reconstructs the delivery network from current restaurants.
    *   `get_inventory(restaurant_id)`: Retrieves current inventory levels.
    *   `record_order(order)`: Adds order to history using linked list.
    *   `generate_report()`: Creates optimization analysis.

*   **Testing and Validation:**
    *   Order processing validation with inventory checks
    *   Menu optimization constraints verification
    *   Route feasibility validation
    *   Integration with core algorithm test suites:
        * Monte Carlo simulation tests
        * Dijkstra's algorithm validation
        * Dynamic programming verification
        * Min-heap structure tests

*   **Visualization Functions:**
    *   Network graph visualization with demand-based node sizing
    *   Min-heap visualization for inventory management
    *   Integration with specialized visualization modules:
        * NetworkX for supply chain mapping
        * Matplotlib for demand patterns
        * Heap visualization for inventory structure

#### Interactive Notebook Guide (`restaurant_optimizer_notebook.ipynb`)

A Jupyter notebook that provides an interactive, educational walkthrough of the restaurant supply chain optimization system. The notebook is designed to be both a learning tool and a practical demonstration platform.

##### Key Features:

* **Interactive Workflow:**
  * Step-by-step execution of optimization processes
  * Real-time visualization of results
  * Ability to modify parameters and see immediate effects
  * Built-in error handling and validation

* **Educational Structure:**
  * Clear explanations for each optimization step
  * Visual representations of algorithms in action
  * Performance measurements and analysis
  * Practical examples and use cases

##### Notebook Sections:

1. **Initialization and Setup**
   * Importing required libraries
   * Setting up the optimizer
   * Loading sample data
   * Configuring visualization settings

2. **Route Optimization**
   * Visualizing the restaurant network
   * Computing optimal delivery routes
   * Analyzing route efficiencies
   * Performance timing of route calculations

3. **Menu Planning**
   * Budget-based menu optimization
   * Comparing greedy vs dynamic programming approaches
   * Visualizing menu selection results
   * Cost-benefit analysis

4. **Demand Simulation**
   * Monte Carlo simulation of customer demand
   * Location-specific demand patterns
   * Statistical analysis of results
   * Visualization of demand distribution

5. **Inventory Management**
   * Real-time inventory tracking
   * Priority-based restocking analysis
   * Min-heap visualization of stock levels
   * Order processing simulation

##### Usage:

1. Start the notebook:
```powershell
jupyter notebook restaurant_optimizer_notebook.ipynb
```

2. Run cells sequentially to follow the optimization process
3. Modify parameters to experiment with different scenarios
4. Observe real-time visualizations and performance metrics

##### Tips for Interactive Learning:

* Execute cells in order to maintain proper state
* Experiment with different parameter values
* Pay attention to performance measurements
* Try the suggested exercises and variations
* Use the visualization tools to understand the algorithms

[Back to Top](#basic-algorithms-and-data-structures)
