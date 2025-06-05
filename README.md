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
*   `_get_middle(head)`: Helper function to find the middle node of a linked list.
*   `_merge_sorted_lists(list1, list2)`: Helper function to merge two sorted linked lists.
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
    *   `_calculate_triangle_apex()`: Calculates the position of the triangle apex for branching.

### Complexity:

*   **Time Complexity:** O(2^n), where n is the recursion depth (due to the branching nature).
*   **Space Complexity:** O(2^n) for storing all square elements in memory (O(n) if using lazy rendering, but current implementation stores all). Additional space for centerline segments and twig bases.
