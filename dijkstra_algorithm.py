"""
Dijkstra's Algorithm Implementation with Binary Heap Optimization
Implementation for shortest path finding in weighted graphs
"""

import heapq
import networkx as nx
import matplotlib.pyplot as plt
import time
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class DijkstraGraph:
    """
    Implementation of Dijkstra's algorithm with binary heap optimization.
    Provides methods for graph creation, shortest path computation, and visualization.
    """
    
    def __init__(self):
        """Initialize empty graph."""
        self.graph: nx.DiGraph = nx.DiGraph()
        
    def add_edge(self, source: str, target: str, weight: float) -> None:
        """
        Add weighted edge to the graph.
        
        Args:
            source: Source vertex
            target: Target vertex  
            weight: Edge weight (must be non-negative for Dijkstra)
        """
        if weight < 0:
            raise ValueError("Dijkstra's algorithm requires non-negative weights")
        self.graph.add_edge(source, target, weight=weight)
    
    def create_sample_graph(self) -> None:
        """Create a sample weighted graph for demonstration."""
        edges = [
            ('A', 'B', 4.0), ('A', 'C', 2.0),
            ('B', 'C', 1.0), ('B', 'D', 5.0),
            ('C', 'D', 8.0), ('C', 'E', 10.0),
            ('D', 'E', 2.0), ('D', 'F', 6.0),
            ('E', 'F', 3.0)
        ]
        
        for source, target, weight in edges:
            self.add_edge(source, target, weight)
    
    def dijkstra_heap(self, src: str, dst: str, weight: str = "weight") -> Tuple[List[str], float]:
        """
        Find shortest path using Dijkstra's algorithm with binary heap optimization.
        
        Args:
            src: Source vertex
            dst: Destination vertex
            weight: Edge weight attribute name
            
        Returns:
            Tuple of (path as list of vertices, total path weight)
            
        Raises:
            KeyError: If source or destination not in graph
            nx.NetworkXNoPath: If no path exists between vertices
            
        Time Complexity: O((V + E) log V) where V is vertices, E is edges
        Space Complexity: O(V) for distance and predecessor arrays
        """
        if src not in self.graph or dst not in self.graph:
            raise KeyError(f"Node {src if src not in self.graph else dst} not in graph.")

        # Distance table and predecessor map
        dist: Dict[str, float] = {node: float("inf") for node in self.graph.nodes}
        prev: Dict[str, Optional[str]] = {node: None for node in self.graph.nodes}

        dist[src] = 0.0
        # heapq works as min-heap: (distance, vertex)
        pq: List[Tuple[float, str]] = [(0.0, src)]

        while pq:
            current_dist, node = heapq.heappop(pq)
            
            if node == dst:
                break
                
            # Skip if we've already found a shorter path to this node
            if current_dist > dist[node]:
                continue
                
            # Examine neighbors
            for neighbor, attrs in self.graph[node].items():
                edge_weight = attrs.get(weight, 1.0)
                if edge_weight is None:
                    edge_weight = 1.0
                    
                new_dist = current_dist + float(edge_weight)
                
                # If we found a shorter path, update it
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

        if dist[dst] == float("inf"):
            raise nx.NetworkXNoPath(f"No path from {src!r} to {dst!r}")

        # Reconstruct path
        path: List[str] = []
        current: Optional[str] = dst
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        
        return path, dist[dst]
    
    def dijkstra_all_pairs(self, src: str, weight: str = "weight") -> Dict[str, Tuple[List[str], float]]:
        """
        Find shortest paths from source to all other reachable vertices.

        This is a single-source shortest path algorithm that computes paths
        from a given source vertex to all other vertices in the graph.

        Args:
            src: Source vertex
            weight: Edge weight attribute name

        Returns:
            Dictionary mapping each vertex to (path, distance) from source.
            Unreachable vertices will have an empty path and infinite distance.

        Time Complexity: O((V + E) log V) using a binary heap.
        Space Complexity: O(V) for distance and predecessor tables, plus O(V*L)
                          where L is the average path length, for storing all paths
                          in the result dictionary. In the worst case (dense graph),
                          this can approach O(V^2).
        """
        if src not in self.graph:
            raise KeyError(f"Source node {src} not in graph")

        # Distance table and predecessor map
        dist: Dict[str, float] = {node: float("inf") for node in self.graph.nodes}
        prev: Dict[str, Optional[str]] = {node: None for node in self.graph.nodes}

        dist[src] = 0.0
        pq: List[Tuple[float, str]] = [(0.0, src)]
        visited = set()

        while pq:
            current_dist, node = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            # Examine neighbors
            for neighbor, attrs in self.graph[node].items():
                if neighbor in visited:
                    continue
                    
                edge_weight = attrs.get(weight, 1.0)
                if edge_weight is None:
                    edge_weight = 1.0

                new_dist = current_dist + float(edge_weight)

                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = node
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct all paths
        results = {}
        for target in self.graph.nodes:
            if dist[target] == float("inf"):
                results[target] = ([], float("inf"))
            else:
                path = []
                current = target
                while current is not None:
                    path.append(current)
                    current = prev[current]
                path.reverse()
                results[target] = (path, dist[target])

        return results
    
    def plot_graph(self, highlighted_path: Optional[List[str]] = None, 
                   title: str = "Weighted Graph - Dijkstra's Algorithm", 
                   figsize: Tuple[int, int] = (12, 9)) -> Dict[str, Any]:
        """
        Visualize the graph with optional path highlighting.
        
        Args:
            highlighted_path: Path to highlight in the visualization
            title: Graph title
            figsize: Figure size tuple
            
        Returns:
            Dictionary of node positions for potential reuse
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Generate layout
        pos = nx.spring_layout(self.graph, seed=42, k=0.9)
        
        # Draw all nodes and edges
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_size=3000, 
                              node_color="lightblue", alpha=0.8)
        nx.draw_networkx_labels(self.graph, pos, font_size=10, 
                               font_weight="bold", ax=ax)
        nx.draw_networkx_edges(self.graph, pos, ax=ax, width=1.5, alpha=0.5, 
                              arrows=True, arrowstyle='->', arrowsize=20)
        
        # Highlight path if provided
        if highlighted_path and len(highlighted_path) > 1:
            path_edges = list(zip(highlighted_path[:-1], highlighted_path[1:]))
            nx.draw_networkx_nodes(self.graph, pos, nodelist=highlighted_path, 
                                  node_color='gold', node_size=3500, ax=ax, alpha=1.0)
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, 
                                  edge_color='red', width=3.0, ax=ax, 
                                  arrows=True, arrowstyle='->', arrowsize=20)
        
        # Draw edge labels (weights)
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        formatted_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=formatted_labels, 
                                    font_color='darkgreen', font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        return dict(pos)
    
    def measure_performance(self, src: str, dst: str, iterations: int = 1000) -> Dict[str, float]:
        """
        Measure algorithm performance over multiple iterations.
        
        Args:
            src: Source vertex
            dst: Destination vertex
            iterations: Number of test iterations
            
        Returns:
            Dictionary with performance metrics
        """
        times: List[float] = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                self.dijkstra_heap(src, dst)
            except nx.NetworkXNoPath:
                pass  # Path doesn't exist, but we still measure time
            end_time = time.perf_counter()
            times.append(float(end_time - start_time))
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'total_time': float(np.sum(times))
        }


def create_random_graph(num_vertices: int, edge_probability: float = 0.3, 
                       max_weight: float = 100.0) -> DijkstraGraph:
    """
    Create a random weighted graph for testing.
    
    Args:
        num_vertices: Number of vertices
        edge_probability: Probability of edge between any two vertices
        max_weight: Maximum edge weight
        
    Returns:
        DijkstraGraph instance with random graph
    """
    graph = DijkstraGraph()
    vertices = [f"V{i}" for i in range(num_vertices)]
    
    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j and random.random() < edge_probability:
                weight = random.uniform(1.0, max_weight)
                graph.add_edge(vertices[i], vertices[j], weight)
    
    return graph


# Test functions with assertions
def test_dijkstra_basic():
    """Test basic Dijkstra functionality."""
    graph = DijkstraGraph()
    graph.create_sample_graph()
    
    # Test basic path finding
    path, distance = graph.dijkstra_heap('A', 'F')
    assert isinstance(path, list), "Path should be a list"
    assert isinstance(distance, float), "Distance should be a float"
    assert path[0] == 'A', "Path should start with source"
    assert path[-1] == 'F', "Path should end with destination"
    assert distance >= 0, "Distance should be non-negative"
    
    print("✓ Basic Dijkstra test passed")


def test_dijkstra_no_path():
    """Test behavior when no path exists."""
    graph = DijkstraGraph()
    graph.add_edge('A', 'B', 1.0)
    graph.add_edge('C', 'D', 1.0)
    
    try:
        graph.dijkstra_heap('A', 'C')
        assert False, "Should raise NetworkXNoPath exception"
    except nx.NetworkXNoPath:
        pass  # Expected behavior
    
    print("✓ No path test passed")


def test_dijkstra_single_vertex():
    """Test path from vertex to itself."""
    graph = DijkstraGraph()
    graph.add_edge('A', 'B', 1.0)
    
    path, distance = graph.dijkstra_heap('A', 'A')
    assert path == ['A'], "Path to self should be single vertex"
    assert distance == 0.0, "Distance to self should be zero"
    
    print("✓ Single vertex test passed")


def test_negative_weight():
    """Test rejection of negative weights."""
    graph = DijkstraGraph()
    
    try:
        graph.add_edge('A', 'B', -1.0)
        assert False, "Should reject negative weights"
    except ValueError:
        pass  # Expected behavior
    
    print("✓ Negative weight test passed")


def test_all_pairs():
    """Test all-pairs shortest path computation."""
    graph = DijkstraGraph()
    graph.create_sample_graph()
    
    results = graph.dijkstra_all_pairs('A')
    assert isinstance(results, dict), "Results should be dictionary"
    assert 'A' in results, "Should include source vertex"
    assert results['A'][1] == 0.0, "Distance to source should be zero"
    
    print("✓ All pairs test passed")


def performance_analysis():
    """Analyze algorithm performance on different graph sizes."""
    print("\n=== Performance Analysis ===")
    
    sizes = [10, 50, 100]
    results = {}
    
    for size in sizes:
        print(f"\nTesting graph with {size} vertices...")
        graph = create_random_graph(size)
        vertices = list(graph.graph.nodes)
        
        if len(vertices) >= 2:
            src, dst = random.sample(vertices, 2)
            perf = graph.measure_performance(src, dst, iterations=100)
            results[size] = perf
            
            print(f"  Mean time: {perf['mean_time']:.6f}s")
            print(f"  Std deviation: {perf['std_time']:.6f}s")
            print(f"  Min/Max: {perf['min_time']:.6f}s / {perf['max_time']:.6f}s")
    
    return results


def main():
    """Main demonstration function."""
    print("=== Dijkstra's Algorithm with Binary Heap ===\n")
    
    # Run tests
    print("Running assertions...")
    test_dijkstra_basic()
    test_dijkstra_no_path()
    test_dijkstra_single_vertex()
    test_negative_weight()
    test_all_pairs()
    print("All tests passed! ✓\n")
    
    # Create and visualize sample graph
    print("Creating sample graph...")
    graph = DijkstraGraph()
    graph.create_sample_graph()
    
    print("Graph vertices:", list(graph.graph.nodes))
    print("Graph edges:", list(graph.graph.edges(data=True)))
    
    # Find shortest path
    print("\nFinding shortest path from A to F...")
    path, distance = graph.dijkstra_heap('A', 'F')
    print(f"Shortest path: {' -> '.join(path)}")
    print(f"Total distance: {distance}")
    
    # Find all shortest paths from A
    print("\nFinding all shortest paths from A...")
    all_paths = graph.dijkstra_all_pairs('A')
    for vertex, (path, dist) in all_paths.items():
        if dist != float('inf'):
            print(f"A -> {vertex}: {' -> '.join(path)} (distance: {dist})")
    
    # Performance analysis
    perf_results = performance_analysis()
    
    # Visualize graph
    print("\nVisualizing graph...")
    graph.plot_graph(highlighted_path=path, 
                     title=f"Shortest Path from A to F (distance: {distance})")
    
    return graph, path, distance, perf_results


if __name__ == "__main__":
    results = main()