"""
Binary Tree Traversal Visualization Module

This module extends the heap visualization functionality to provide
visual representation of tree traversal algorithms (DFS and BFS).
It demonstrates iterative implementations using stack and queue data structures.
"""

import uuid
import time
from typing import List, Optional, Union, Tuple, Sequence, Set, Dict
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import numpy as np


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


def add_edges(graph: nx.DiGraph, node: Optional[Node], pos: Dict[str, Tuple[float, float]],
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


def generate_traversal_colors(num_nodes: int, colormap: str = 'plasma') -> List[str]:
    """
    Generate colors for tree traversal visualization using colormap.
    Colors transition from dark to light based on traversal order.
    
    Args:
        num_nodes: Number of nodes to generate colors for
        colormap: Matplotlib colormap name (default: 'plasma')
        
    Returns:
        List of hex color strings
        
    Time Complexity: O(n) where n is num_nodes
    Space Complexity: O(n) for color list
    """
    if num_nodes <= 0:
        return []
    
    # Generate color values from 0 to 1
    color_values = np.linspace(0, 1, num_nodes)
    cmap = plt.cm.get_cmap(colormap)
    
    # Convert to hex colors
    colors = []
    for val in color_values:
        rgba = cmap(val)
        hex_color = mcolors.rgb2hex(rgba[:3])
        colors.append(hex_color)
    
    return colors


def dfs(root: Optional[Node]) -> Tuple[List[Node], float]:
    """
    Perform depth-first search traversal using iterative approach with stack.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        Tuple of (visited_nodes_list, execution_time)
        
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(h) where h is height of tree (stack size)
    """
    start_time = time.perf_counter()
    
    if not root:
        return [], time.perf_counter() - start_time
    
    visited: List[Node] = []
    stack: List[Node] = [root]
    visited_set: Set[str] = set()
    
    while stack:
        node = stack.pop()  # LIFO - Last In, First Out
        
        if node.id not in visited_set:
            visited.append(node)
            visited_set.add(node.id)
            
            # Add children to stack (right first, then left)
            # This ensures left subtree is processed before right subtree
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
    
    end_time = time.perf_counter()
    return visited, end_time - start_time


def bfs(root: Optional[Node]) -> Tuple[List[Node], float]:
    """
    Perform breadth-first search traversal using iterative approach with queue.
    
    Args:
        root: Root node of the binary tree
        
    Returns:
        Tuple of (visited_nodes_list, execution_time)
        
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(w) where w is maximum width of tree (queue size)
    """
    start_time = time.perf_counter()
    
    if not root:
        return [], time.perf_counter() - start_time
    
    visited: List[Node] = []
    queue: deque[Node] = deque([root])
    visited_set: Set[str] = set()
    
    while queue:
        node = queue.popleft()  # FIFO - First In, First Out
        
        if node.id not in visited_set:
            visited.append(node)
            visited_set.add(node.id)
            
            # Add children to queue (left first, then right)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    end_time = time.perf_counter()
    return visited, end_time - start_time


def visualize_traversal(root: Node, traversal_type: str = "dfs", 
                       colormap: str = 'plasma') -> Tuple[List[Node], float]:
    """
    Visualize tree traversal with color-coded nodes showing visit order.
    
    Args:
        root: Root node of the binary tree
        traversal_type: Type of traversal ("dfs" or "bfs")
        colormap: Matplotlib colormap for node colors
        
    Returns:
        Tuple of (visited_nodes_list, execution_time)
        
    Raises:
        ValueError: If traversal_type is not "dfs" or "bfs"
        
    Time Complexity: O(n) where n is number of nodes
    Space Complexity: O(n) for graph construction and visualization
    """
    if traversal_type not in ["dfs", "bfs"]:
        raise ValueError("traversal_type must be 'dfs' or 'bfs'")
    
    # Perform traversal
    if traversal_type == "dfs":
        visited_nodes, exec_time = dfs(root)
        title = "Depth-First Search (DFS) Traversal - Stack Implementation"
    else:
        visited_nodes, exec_time = bfs(root)
        title = "Breadth-First Search (BFS) Traversal - Queue Implementation"
    
    if not visited_nodes:
        return visited_nodes, exec_time
    
    # Generate colors for traversal order
    colors = generate_traversal_colors(len(visited_nodes), colormap)
    
    # Update node colors based on traversal order
    for i, node in enumerate(visited_nodes):
        node.color = colors[i]
    
    # Create graph for visualization
    tree = nx.DiGraph()
    pos = {root.id: (0.0, 0.0)}
    tree = add_edges(tree, root, pos)
    
    # Extract colors and labels for matplotlib
    node_colors = [node[1]['color'] for node in tree.nodes(data=True)]
    labels = {node[0]: node[1]['label'] for node in tree.nodes(data=True)}
    
    # Create visualization
    plt.figure(figsize=(8, 6))
    plt.title(f"{title}\nTraversal Order: {[node.val for node in visited_nodes]}", 
              fontsize=14, fontweight='bold')
    
    nx.draw(tree, pos=pos, labels=labels, arrows=False, 
            node_size=2500, node_color=node_colors, font_size=12, 
            font_weight='bold', font_color='white')
    
    sm = plt.cm.ScalarMappable(cmap=colormap, 
                              norm=Normalize(vmin=0, vmax=len(visited_nodes)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Traversal Order (0: First visited, N-1: Last visited)', 
                   rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    
    return visited_nodes, exec_time


def array_to_heap_tree(heap_array: Sequence[Union[int, float]]) -> Optional[Node]:
    """
    Convert array representation of binary heap to tree structure.
    
    Args:
        heap_array: Array representation of the heap
        
    Returns:
        Root node of the constructed tree or None if array is empty
        
    Time Complexity: O(n) where n is length of heap_array
    Space Complexity: O(n) for storing nodes
    """
    if not heap_array:
        return None
    
    # Create nodes for all elements
    nodes = [Node(val) for val in heap_array]
    
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
    
    return nodes[0]


def create_sample_tree() -> Node:
    """
    Create a sample binary tree for testing purposes.
    
    Returns:
        Root node of the sample tree
    Time Complexity: O(1) for creating fixed structure
    """
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)
    
    return root


# Test functions with assertions
def test_dfs_traversal() -> None:
    """Test DFS traversal functionality."""
    print("Testing DFS traversal...")
    
    # Test with sample tree
    root = create_sample_tree()
    visited, exec_time = dfs(root)
    
    # Expected DFS order: [1, 2, 4, 5, 3, 6, 7]
    expected_values = [1, 2, 4, 5, 3, 6, 7]
    actual_values = [node.val for node in visited]
    
    assert actual_values == expected_values, f"DFS order mismatch: expected {expected_values}, got {actual_values}"
    assert len(visited) == 7, f"Expected 7 nodes, got {len(visited)}"
    assert exec_time > 0, "Execution time should be positive"
    
    print(f"✓ DFS test passed (execution time: {exec_time:.6f}s)")
    print(f"  Traversal order: {actual_values}")


def test_bfs_traversal() -> None:
    """Test BFS traversal functionality."""
    print("Testing BFS traversal...")
    
    # Test with sample tree
    root = create_sample_tree()
    visited, exec_time = bfs(root)
    
    # Expected BFS order: [1, 2, 3, 4, 5, 6, 7]
    expected_values = [1, 2, 3, 4, 5, 6, 7]
    actual_values = [node.val for node in visited]
    
    assert actual_values == expected_values, f"BFS order mismatch: expected {expected_values}, got {actual_values}"
    assert len(visited) == 7, f"Expected 7 nodes, got {len(visited)}"
    assert exec_time > 0, "Execution time should be positive"
    
    print(f"✓ BFS test passed (execution time: {exec_time:.6f}s)")
    print(f"  Traversal order: {actual_values}")


def test_color_generation() -> None:
    """Test color generation functionality."""
    print("Testing color generation...")
    
    # Test with different sizes
    colors_5 = generate_traversal_colors(5)
    colors_10 = generate_traversal_colors(10)
    
    assert len(colors_5) == 5, f"Expected 5 colors, got {len(colors_5)}"
    assert len(colors_10) == 10, f"Expected 10 colors, got {len(colors_10)}"
    
    # Check if colors are valid hex codes
    for color in colors_5:
        assert color.startswith('#'), f"Color {color} should start with #"
        assert len(color) == 7, f"Color {color} should be 7 characters long"
    
    # Test empty case
    empty_colors = generate_traversal_colors(0)
    assert len(empty_colors) == 0, "Empty input should return empty list"
    
    print("✓ Color generation test passed")


def test_edge_cases() -> None:
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    # Test with None root
    visited_dfs, _ = dfs(None)
    visited_bfs, _ = bfs(None)
    
    assert len(visited_dfs) == 0, "DFS with None root should return empty list"
    assert len(visited_bfs) == 0, "BFS with None root should return empty list"
    
    # Test with single node
    single_node = Node(42)
    visited_dfs_single, _ = dfs(single_node)
    visited_bfs_single, _ = bfs(single_node)
    
    assert len(visited_dfs_single) == 1, "Single node DFS should return one node"
    assert len(visited_bfs_single) == 1, "Single node BFS should return one node"
    assert visited_dfs_single[0].val == 42, "Single node value should be 42"
    assert visited_bfs_single[0].val == 42, "Single node value should be 42"
    
    # Test invalid traversal type
    try:
        visualize_traversal(single_node, "invalid")
        assert False, "Should raise ValueError for invalid traversal type"
    except ValueError as e:
        assert "must be 'dfs' or 'bfs'" in str(e), "Error message should mention valid types"
    
    print("✓ Edge cases test passed")


def performance_analysis() -> None:
    """Analyze performance of traversal algorithms."""
    print("\nPerformance Analysis:")
    print("-" * 50)
    
    # Test with different tree sizes
    sizes = [7, 15, 31, 63]  # Complete binary trees
    
    for size in sizes:
        # Create array representation and convert to tree
        arr = list(range(1, size + 1))
        root = array_to_heap_tree(arr)
        
        if root:
            # Measure DFS performance
            start = time.perf_counter()
            dfs_visited, _ = dfs(root)
            dfs_time = time.perf_counter() - start
            
            # Measure BFS performance
            start = time.perf_counter()
            bfs_visited, _ = bfs(root)
            bfs_time = time.perf_counter() - start
            
            print(f"Size: {size:2d} | DFS: {dfs_time*1000:6.3f}ms | "
                  f"BFS: {bfs_time*1000:6.3f}ms | "
                  f"Nodes visited: {len(dfs_visited)}")
            
            # Verify both algorithms visit same number of nodes
            assert len(dfs_visited) == len(bfs_visited) == size, \
                f"Both algorithms should visit all {size} nodes"


def demonstrate_traversals() -> None:
    """Demonstrate both DFS and BFS traversals with visualizations."""
    print("\nDemonstrating Tree Traversals:")
    print("=" * 50)
    
    # Create and visualize sample tree
    print("Creating sample binary tree...")
    root = create_sample_tree()
    
    print("\n1. Depth-First Search (DFS) Visualization:")
    dfs_nodes, dfs_time = visualize_traversal(root, "dfs")
    print(f"DFS traversal order: {[node.val for node in dfs_nodes]}")
    print(f"Execution time: {dfs_time:.6f}s")
    
    print("\n2. Breadth-First Search (BFS) Visualization:")
    bfs_nodes, bfs_time = visualize_traversal(root, "bfs")
    print(f"BFS traversal order: {[node.val for node in bfs_nodes]}")
    print(f"Execution time: {bfs_time:.6f}s")
    
    # Demonstrate with heap-based tree
    print("\n3. Heap-based Tree Traversals:")
    heap_array = [1, 3, 6, 5, 9, 8, 10, 15, 12, 20]
    heap_root = array_to_heap_tree(heap_array)
    
    if heap_root:
        print(f"Heap array: {heap_array}")
        
        print("\nDFS on heap tree:")
        heap_dfs_nodes, heap_dfs_time = visualize_traversal(heap_root, "dfs")
        print(f"DFS order: {[node.val for node in heap_dfs_nodes]}")
        
        print("\nBFS on heap tree:")
        heap_bfs_nodes, heap_bfs_time = visualize_traversal(heap_root, "bfs")
        print(f"BFS order: {[node.val for node in heap_bfs_nodes]}")


if __name__ == "__main__":
    print("Binary Tree Traversal Visualization")
    print("=" * 50)
    
    # Run tests
    test_dfs_traversal()
    test_bfs_traversal()
    test_color_generation()
    test_edge_cases()
    
    # Performance analysis
    performance_analysis()
    
    # Demonstrate traversals with visualizations
    demonstrate_traversals()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
