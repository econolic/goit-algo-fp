"""
Binary Heap Visualization Module

This module provides functionality to visualize binary heaps as tree structures
using NetworkX and Matplotlib. It converts array-based heap representation
to a tree structure for visualization purposes.
"""

import uuid
import time
from typing import List, Optional, Union, Tuple, Sequence
import networkx as nx
import matplotlib.pyplot as plt


class Node:
    """
    Node class for binary tree representation.
    
    Attributes:
        val: The value stored in the node
        color: Color for visualization (default: skyblue)
        left: Reference to left child node
        right: Reference to right child node
        id: Unique identifier for the node
    """
    
    def __init__(self, key: Union[int, float], color: str = "skyblue") -> None:
        """
        Initialize a new node.
        
        Args:
            key: The value to store in the node
            color: Color for visualization purposes
        """
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None
        self.val: Union[int, float] = key
        self.color: str = color
        self.id: str = str(uuid.uuid4())


def add_edges(graph: nx.DiGraph, node: Optional[Node], pos: dict,
              x: float = 0, y: float = 0, layer: int = 1) -> nx.DiGraph:
    """
    Recursively add edges to the graph for tree visualization.
    
    Args:
        graph: NetworkX directed graph
        node: Current node being processed
        pos: Dictionary to store node positions
        x: X coordinate for current node
        y: Y coordinate for current node
        layer: Current layer depth in the tree
        
    Returns:
        Updated NetworkX graph
        
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree (recursion stack)
    """
    if node is not None:
        graph.add_node(node.id, color=node.color, label=node.val)
        pos[node.id] = (x, y)
        
        if node.left:
            graph.add_edge(node.id, node.left.id)
            l = x - 1 / 2 ** layer
            pos[node.left.id] = (l, y - 1)
            add_edges(graph, node.left, pos, x=l, y=y - 1, layer=layer + 1)
            
        if node.right:
            graph.add_edge(node.id, node.right.id)
            r = x + 1 / 2 ** layer
            pos[node.right.id] = (r, y - 1)
            add_edges(graph, node.right, pos, x=r, y=y - 1, layer=layer + 1)
            
    return graph


def draw_tree(tree_root: Node, title: str = "Binary Tree Visualization") -> None:
    """
    Draw the binary tree using matplotlib.
    
    Args:
        tree_root: Root node of the tree
        title: Title for the plot
        
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) for storing graph and positions
    """
    tree = nx.DiGraph()
    pos = {tree_root.id: (0, 0)}
    tree = add_edges(tree, tree_root, pos)
    
    colors = [node[1]['color'] for node in tree.nodes(data=True)]
    labels = {node[0]: node[1]['label'] for node in tree.nodes(data=True)}
    
    plt.figure(figsize=(12, 8))
    plt.title(title, fontsize=16, fontweight='bold')
    nx.draw(tree, pos=pos, labels=labels, arrows=False, 
            node_size=2500, node_color=colors, font_size=12, font_weight='bold')
    plt.show()


def array_to_heap_tree(heap_array: Sequence[Union[int, float]],
                      heap_type: str = "min") -> Optional[Node]:
    """
    Convert array representation of binary heap to tree structure.
    
    Args:
        heap_array: Array representation of the heap
        heap_type: Type of heap ("min" or "max")
        
    Returns:
        Root node of the constructed tree or None if array is empty
        
    Raises:
        ValueError: If heap_array is empty or heap_type is invalid
        
    Time Complexity: O(n) where n is length of heap_array
    Space Complexity: O(n) for storing nodes
    """
    if not heap_array:
        raise ValueError("Heap array cannot be empty")
    
    if heap_type not in ["min", "max"]:
        raise ValueError("heap_type must be 'min' or 'max'")
    
    # Color coding: green for min-heap root, red for max-heap root, skyblue for others
    root_color = "lightgreen" if heap_type == "min" else "lightcoral"
    
    # Create nodes for all elements
    nodes = []
    for i, val in enumerate(heap_array):
        color = root_color if i == 0 else "skyblue"
        nodes.append(Node(val, color))
    
    # Build tree structure using heap property
    # For element at index i:
    # - Left child at index 2*i + 1
    # - Right child at index 2*i + 2
    for i in range(len(heap_array)):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        
        if left_idx < len(heap_array):
            nodes[i].left = nodes[left_idx]
        if right_idx < len(heap_array):
            nodes[i].right = nodes[right_idx]
    
    return nodes[0] if nodes else None


def is_valid_heap(heap_array: Sequence[Union[int, float]],
                 heap_type: str = "min") -> bool:
    """
    Validate if array represents a valid heap.
    
    Args:
        heap_array: Array to validate
        heap_type: Type of heap to validate against
        
    Returns:
        True if valid heap, False otherwise
        
    Time Complexity: O(n) where n is length of heap_array
    Space Complexity: O(1)
    """
    if not heap_array:
        return True
    
    n = len(heap_array)
    
    # Check heap property for all non-leaf nodes
    for i in range(n // 2):
        left_idx = 2 * i + 1
        right_idx = 2 * i + 2
        
        if heap_type == "min":
            # Min-heap: parent <= children
            if left_idx < n and heap_array[i] > heap_array[left_idx]:
                return False
            if right_idx < n and heap_array[i] > heap_array[right_idx]:
                return False
        else:  # max heap
            # Max-heap: parent >= children
            if left_idx < n and heap_array[i] < heap_array[left_idx]:
                return False
            if right_idx < n and heap_array[i] < heap_array[right_idx]:
                return False
    
    return True


def visualize_heap(heap_array: Sequence[Union[int, float]],
                  heap_type: str = "min",
                  validate: bool = True) -> Tuple[Optional[Node], float]:
    """
    Main function to visualize binary heap from array representation.
    
    Args:
        heap_array: Array representation of binary heap
        heap_type: Type of heap ("min" or "max")
        validate: Whether to validate heap property
        
    Returns:
        Tuple of (root_node, execution_time)
        
    Raises:
        ValueError: If heap is invalid and validate=True
        
    Time Complexity: O(n) where n is length of heap_array
    Space Complexity: O(n) for tree construction
    """
    start_time = time.perf_counter()
    
    if validate and not is_valid_heap(heap_array, heap_type):
        raise ValueError(f"Invalid {heap_type}-heap structure")
    
    root = array_to_heap_tree(heap_array, heap_type)
    
    if root:
        title = f"Binary {heap_type.capitalize()}-Heap Visualization"
        draw_tree(root, title)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    return root, execution_time


def build_heap_from_array(arr: Sequence[Union[int, float]],
                         heap_type: str = "min") -> List[Union[int, float]]:
    """
    Build a valid heap from unsorted array using heapify algorithm.
    
    Args:
        arr: Input array to heapify
        heap_type: Type of heap to build
        
    Returns:
        Array representing valid heap
        
    Time Complexity: O(n) - heapify algorithm
    Space Complexity: O(1) - in-place modification
    """
    if not arr:
        return [] # Return an empty list for empty input
    
    heap = list(arr) # Create a mutable list copy
    n = len(heap)
    
    def heapify(heap_arr: List[Union[int, float]], n: int, i: int) -> None: # Keep as List for mutation
        """Helper function to maintain heap property."""
        if heap_type == "min":
            extreme = i  # smallest
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and heap_arr[left] < heap_arr[extreme]:
                extreme = left
            if right < n and heap_arr[right] < heap_arr[extreme]:
                extreme = right
        else:  # max heap
            extreme = i  # largest
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and heap_arr[left] > heap_arr[extreme]:
                extreme = left
            if right < n and heap_arr[right] > heap_arr[extreme]:
                extreme = right
        
        if extreme != i:
            heap_arr[i], heap_arr[extreme] = heap_arr[extreme], heap_arr[i]
            heapify(heap_arr, n, extreme)
    
    # Build heap (rearrange array)
    for i in range(n // 2 - 1, -1, -1):
        heapify(heap, n, i)
    
    return heap # Return the modified heap list


# Test functions and assertions
def test_heap_functionality() -> None:
    """
    Comprehensive test suite for heap visualization functionality.
    """
    print("Running comprehensive tests...")
    
    # Test 1: Valid min-heap
    min_heap = [1, 3, 6, 5, 9, 8]
    assert is_valid_heap(min_heap, "min"), "Min-heap validation failed"
    root, exec_time = visualize_heap(min_heap, "min")
    assert root is not None, "Root should not be None for valid heap"
    print(f"✓ Min-heap test passed (execution time: {exec_time:.6f}s)")
    
    # Test 2: Valid max-heap
    max_heap = [9, 8, 6, 5, 3, 1]
    assert is_valid_heap(max_heap, "max"), "Max-heap validation failed"
    root, exec_time = visualize_heap(max_heap, "max")
    assert root is not None, "Root should not be None for valid heap"
    print(f"✓ Max-heap test passed (execution time: {exec_time:.6f}s)")
    
    # Test 3: Invalid heap should raise error
    invalid_heap = [1, 9, 6, 5, 3, 8]  # 9 > 1 violates min-heap
    assert not is_valid_heap(invalid_heap, "min"), "Should detect invalid min-heap"
    
    try:
        visualize_heap(invalid_heap, "min")
        assert False, "Should raise ValueError for invalid heap"
    except ValueError:
        print("✓ Invalid heap detection test passed")
    
    # Test 4: Empty heap
    try:
        visualize_heap([], "min")
        assert False, "Should raise ValueError for empty heap"
    except ValueError:
        print("✓ Empty heap test passed")
    
    # Test 5: Single element heap
    single_heap = [42]
    assert is_valid_heap(single_heap, "min"), "Single element should be valid heap"
    root, exec_time = visualize_heap(single_heap, "min")
    assert root is not None, "Root should not be None for single element heap"
    assert root.val == 42, "Single element value should match"
    print(f"✓ Single element test passed (execution time: {exec_time:.6f}s)")
    
    # Test 6: Build heap from unsorted array
    unsorted = [4, 10, 3, 5, 1, 15, 2, 7, 6, 12]
    min_heapified = build_heap_from_array(unsorted, "min")
    max_heapified = build_heap_from_array(unsorted, "max")
    
    assert is_valid_heap(min_heapified, "min"), "Heapified array should be valid min-heap"
    assert is_valid_heap(max_heapified, "max"), "Heapified array should be valid max-heap"
    
    print("✓ Heapify algorithm test passed")
    print(f"Original: {unsorted}")
    print(f"Min-heap: {min_heapified}")
    print(f"Max-heap: {max_heapified}")
    
    # Visualize the heapified arrays
    print("\nVisualizing min-heap from unsorted array:")
    visualize_heap(min_heapified, "min")
    
    print("\nVisualizing max-heap from unsorted array:")
    visualize_heap(max_heapified, "max")


if __name__ == "__main__":
    # Performance benchmarking
    print("Binary Heap Visualization Module")
    print("=" * 50)
    
    # Run tests
    test_heap_functionality()
    
    # Performance analysis
    print("\nPerformance Analysis:")
    print("-" * 30)
    
    sizes = [10, 50, 100, 500]
    for size in sizes:
        # Generate random heap
        import random
        arr = list(range(1, size + 1))
        random.shuffle(arr)
        heap_arr = build_heap_from_array(arr, "min")
        
        start = time.perf_counter()
        root, _ = visualize_heap(heap_arr, "min", validate=False)
        end = time.perf_counter()
        
        theoretical_complexity = size  # O(n)
        print(f"Size: {size:4d} | Time: {(end-start)*1000:8.3f}ms | "
              f"Theoretical O(n): {theoretical_complexity:4d}")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
