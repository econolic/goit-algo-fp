"""
Restaurant Supply Chain Optimizer

This module integrates various algorithms to optimize restaurant chain operations:
- Route optimization for deliveries using Dijkstra's algorithm
- Menu planning with knapsack optimization
- Inventory management using min-heap priority queues
- Order tracking with linked lists
- Demand forecasting using Monte Carlo simulation
- Supply chain visualization with NetworkX

Key Features:
- Optimizes delivery routes between restaurant locations
- Maximizes menu variety within budget constraints
- Predicts location-specific demand patterns
- Manages inventory priorities efficiently
- Provides comprehensive performance analysis
- Visualizes network relationships and metrics

Example Usage:
    optimizer = RestaurantOptimizer()
    optimizer.setup_sample_data()
    
    # Optimize delivery routes
    routes = optimizer.optimize_delivery_routes("R1")
    
    # Plan menu and simulate demand
    menu = optimizer.plan_daily_menu("R1")
    demand = optimizer.simulate_daily_demand("R1", days=30)
    
    # Update inventory and visualize
    optimizer.update_inventory("R1", demand)
    optimizer.visualize_supply_chain()

Algorithm Complexities:
- Route Optimization: O((V + E) log V)
- Menu Planning: O(n × budget)
- Demand Simulation: O(n)
- Visualization: O(V + E)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from monte_carlo_dice import simulate_dice_rolls

# Import custom modules
from dijkstra_algorithm import DijkstraGraph
from greed_dp_algorithms import greedy_algorithm, dynamic_programming, FoodItem
from heap_visualization import visualize_heap
from linked_list_operations import LinkedList

@dataclass
class Restaurant:
    """Restaurant location and details."""
    id: str
    name: str
    location: Tuple[float, float]  # (latitude, longitude)
    daily_budget: float
    storage_capacity: int
    inventory: Dict[str, int] = None  # type: ignore  # Store current inventory levels
    
    def __post_init__(self):
        if self.inventory is None:
            self.inventory = {}  # Initialize empty inventory if none provided

@dataclass
class Order:
    """Customer order details."""
    id: str
    restaurant_id: str
    items: List[str]
    total_cost: float
    timestamp: float

class RestaurantOptimizer:
    """Main class for restaurant chain optimization."""
    
    def __init__(self):
        self.restaurants: Dict[str, Restaurant] = {}
        self.delivery_graph = DijkstraGraph()
        self.order_history = LinkedList()
        self.inventory_heap = []
        self.menu_items: Dict[str, FoodItem] = {
            "pizza": FoodItem(cost=50, calories=300),
            "burger": FoodItem(cost=40, calories=250),
            "salad": FoodItem(cost=30, calories=150),
            "pasta": FoodItem(cost=45, calories=280),
            "dessert": FoodItem(cost=25, calories=200)
        }
    
    def setup_sample_data(self):
        """Initialize sample restaurants and delivery routes."""        # Add sample restaurants with initial inventory (100 units each for consistency)
        sample_inventory: Dict[str, int] = {
            "pizza": 100,
            "burger": 100,
            "salad": 100,
            "pasta": 100,
            "dessert": 100
        }
        
        self.restaurants = {
            "R1": Restaurant("R1", "Downtown Diner", (40.7128, -74.0060), 1000.0, 100, sample_inventory.copy()),
            "R2": Restaurant("R2", "Uptown Eats", (40.7589, -73.9851), 1200.0, 150, sample_inventory.copy()),
            "R3": Restaurant("R3", "Harbor View", (40.7023, -73.9875), 800.0, 80, sample_inventory.copy())
        }
        
        # Create delivery route graph
        for r1 in self.restaurants.values():
            for r2 in self.restaurants.values():
                if r1.id != r2.id:
                    # Calculate distance-based weight
                    weight = ((r1.location[0] - r2.location[0])**2 + 
                            (r1.location[1] - r2.location[1])**2)**0.5 * 100
                    self.delivery_graph.add_edge(r1.id, r2.id, weight)
    
    def rebuild_delivery_graph(self):
        """Rebuild the delivery graph from the current set of restaurants."""
        self.delivery_graph = DijkstraGraph()
        for r1 in self.restaurants.values():
            for r2 in self.restaurants.values():
                if r1.id != r2.id:
                    weight = ((r1.location[0] - r2.location[0])**2 + 
                              (r1.location[1] - r2.location[1])**2)**0.5 * 100
                    self.delivery_graph.add_edge(r1.id, r2.id, weight)

    def optimize_delivery_routes(self, start_restaurant_id: str) -> Dict[str, float]:
        """Find optimal delivery routes from a starting restaurant."""
        # Ensure the delivery graph is initialized and contains all restaurants
        if (len(self.delivery_graph.graph.nodes) < len(self.restaurants) or
            not all(rid in self.delivery_graph.graph.nodes for rid in self.restaurants)):
            self.rebuild_delivery_graph()
        if start_restaurant_id not in self.delivery_graph.graph.nodes:
            raise KeyError(f"Source node {start_restaurant_id} not in delivery graph. Available: {list(self.delivery_graph.graph.nodes)}")
        # dijkstra_all_pairs returns Dict[str, Tuple[List[str], float]]
        all_routes = self.delivery_graph.dijkstra_all_pairs(start_restaurant_id, "weight")
        return {dest: dist for dest, (_, dist) in all_routes.items()}
    
    def plan_daily_menu(self, restaurant_id: str) -> List[str]:
        """Optimize menu items for a restaurant within budget constraints."""
        restaurant = self.restaurants[restaurant_id]
        result = dynamic_programming(self.menu_items, int(restaurant.daily_budget))
        return result.items

    def simulate_daily_demand(self, restaurant_id: str, days: int = 30) -> Dict[str, int]:
        """
        Simulate customer demand using Monte Carlo simulation with realistic variations.
        
        Uses monte_carlo_dice.py simulation engine with:
        - Location-specific modifiers
        - Item popularity factors
        - Daily variations

        Args:
            restaurant_id (str): ID of the restaurant to simulate demand for
            days (int): Number of days to simulate (default: 30)
            
        Returns:
            Dict[str, int]: Average daily demand for each menu item
        """
        # Item popularity factors (relative to average)
        popularity_factors = {
            "pizza": 1.4,    # Most popular
            "burger": 1.2,   # Very popular
            "pasta": 1.0,    # Average popularity
            "salad": 0.8,    # Less popular
            "dessert": 0.6   # Least popular
        }

        # Location-based modifiers
        location_mod = {
            "R1": 1.5,  # Downtown - high traffic
            "R2": 1.2,  # Uptown - good traffic
            "R3": 1.0   # Harbor - normal traffic
        }

        # Use Monte Carlo simulation for base demand (3d6 for more normal distribution)
        base_results = simulate_dice_rolls(n_dice=3, n_sides=6, iterations=days * len(self.menu_items))
        base_values = list(base_results.values())
        avg_base_demand = sum(base_values) / len(base_values) * 10  # Scale to realistic numbers

        # Calculate demand with variations for each item
        demand = {}
        for item in self.menu_items:
            # Base demand with popularity and location factors
            avg_demand = (avg_base_demand * 
                        popularity_factors[item] * 
                        location_mod[restaurant_id])
            
            # Simulate daily variations
            daily_demands = []
            for _ in range(days):
                variation = random.normalvariate(1.0, 0.1)  # Normal distribution with 10% std dev
                daily_demands.append(max(1, int(avg_demand * variation)))
            
            # Store average daily demand
            demand[item] = int(sum(daily_demands) / days)

        return demand
    
    def get_inventory(self, restaurant_id: str) -> Dict[str, int]:
        """Get current inventory levels for a restaurant."""
        if restaurant_id not in self.restaurants:
            raise KeyError(f"Restaurant {restaurant_id} not found")
        return self.restaurants[restaurant_id].inventory

    def update_inventory(self, restaurant_id: str, demand: Dict[str, int]) -> None:
        """
        Update inventory based on demand predictions.
        Organizes items in a min-heap structure for priority-based restocking.
        """
        if restaurant_id not in self.restaurants:
            raise KeyError(f"Restaurant {restaurant_id} not found")
        
        restaurant = self.restaurants[restaurant_id]
        
        # Update inventory levels based on demand
        for item, needed in demand.items():
            if item not in restaurant.inventory:
                restaurant.inventory[item] = restaurant.storage_capacity  # Initial stock
            # Calculate new inventory level (bounded by storage capacity)
            new_level = min(
                restaurant.storage_capacity,
                restaurant.inventory[item] + needed  # Restock based on demand
            )
            restaurant.inventory[item] = new_level

        # Convert inventory to list of (count, item) tuples for heap visualization
        inventory_data = [(count, item) for item, count in restaurant.inventory.items()]
        
        # Create a valid min-heap structure
        def heapify(arr: List[Tuple[int, str]], n: int, i: int) -> None:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[left][0] < arr[smallest][0]:
                smallest = left

            if right < n and arr[right][0] < arr[smallest][0]:
                smallest = right

            if smallest != i:
                arr[i], arr[smallest] = arr[smallest], arr[i]
                heapify(arr, n, smallest)

        # Build min-heap
        n = len(inventory_data)
        for i in range(n // 2 - 1, -1, -1):
            heapify(inventory_data, n, i)

        print(f"\nInventory levels for {restaurant.name}:")
        for count, item in sorted(restaurant.inventory.items()):
            print(f"  {item}: {count} units in stock")

        # Visualize the heap structure
        visualize_heap([count for count, _ in inventory_data])
    
    def record_order(self, order: Order):
        """Add order to history using linked list."""
        self.order_history.append(order)

    def process_order(self, restaurant_id: str, items_ordered: Dict[str, int]) -> bool:
        """
        Processes a customer order by validating against the menu and inventory.

        Args:
            restaurant_id (str): The ID of the restaurant receiving the order.
            items_ordered (Dict[str, int]): A dictionary of items and quantities being ordered.

        Returns:
            bool: True if the order is successfully processed, False otherwise.
        """
        if restaurant_id not in self.restaurants:
            print(f"Error: Restaurant {restaurant_id} not found.")
            return False

        restaurant = self.restaurants[restaurant_id]
        menu = self.plan_daily_menu(restaurant_id) # Use optimized menu for validation

        # 1. Validate if all ordered items are on the menu
        for item in items_ordered:
            if item not in menu:
                print(f"Order failed: '{item}' is not on the menu for {restaurant.name}.")
                return False

        # 2. Validate if there is enough inventory
        inventory = self.get_inventory(restaurant_id)
        for item, quantity in items_ordered.items():
            if inventory.get(item, 0) < quantity:
                print(f"Order failed: Not enough inventory for '{item}' at {restaurant.name}.")
                return False

        # 3. Process the order: update inventory and record the order
        total_cost = 0
        for item, quantity in items_ordered.items():
            restaurant.inventory[item] -= quantity
            total_cost += self.menu_items[item]['cost'] * quantity

        # 4. Record the successful order
        new_order = Order(
            id=f"O{len(self.order_history.to_list()) + 1}", # Simple unique ID
            restaurant_id=restaurant_id,
            items=list(items_ordered.keys()),
            total_cost=float(total_cost),
            timestamp=time.time()
        )
        self.record_order(new_order)
        
        print(f"Order {new_order.id} processed successfully for {restaurant.name}.")
        return True

    def visualize_supply_chain(self):
        """Create a visualization of the supply chain network with demand patterns."""
        G = nx.Graph()
        
        # Add nodes (restaurants) with demand data
        node_demands = []
        for r_id, restaurant in self.restaurants.items():
            demand = sum(self.simulate_daily_demand(r_id).values())
            G.add_node(r_id, 
                      demand=float(demand),
                      name=restaurant.name,
                      pos=restaurant.location)
            node_demands.append(demand)
        
        # Convert to numpy array and normalize for visualization
        node_demands = np.array(node_demands)
        normalized_demands = (node_demands - node_demands.min()) / (node_demands.max() - node_demands.min())
        node_sizes = [int(1000 + d * 2000) for d in normalized_demands]  # Convert to integers
        
        # Add edges (delivery routes)
        for r1 in self.restaurants:
            routes = self.optimize_delivery_routes(r1)
            for r2, distance in routes.items():
                if r1 < r2:  # Add each edge only once
                    G.add_edge(r1, r2, weight=float(distance))

        # Get node positions
        pos = nx.spring_layout(G)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # Draw nodes with sizes based on demand
        for i, (node, (x, y)) in enumerate(pos.items()):
            plt.scatter(x, y, s=node_sizes[i], c=[normalized_demands[i]], 
                       cmap='hot', alpha=0.6)
        
        # Add labels
        labels = nx.get_node_attributes(G, 'name')
        for node, (x, y) in pos.items():
            plt.annotate(labels[node], (x, y), xytext=(0, 0), 
                        textcoords='offset points', ha='center', va='center')
        
        # Add title
        plt.title("Restaurant Supply Chain Network\nNode size and color indicate demand level")
        
        # Show the plot
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive optimization report."""
        print("\n=== Restaurant Chain Optimization Report ===\n")
        
        # Analyze delivery routes
        print("Delivery Route Analysis:")
        for r_id in self.restaurants:
            routes = self.optimize_delivery_routes(r_id)
            print(f"Routes from {self.restaurants[r_id].name}:")
            for dest, dist in routes.items():
                print(f"  → {self.restaurants[dest].name}: {dist:.2f} units")
        
        # Analyze menu planning
        print("\nMenu Planning Analysis:")
        for r_id, restaurant in self.restaurants.items():
            menu = self.plan_daily_menu(r_id)
            print(f"{restaurant.name} optimal menu items: {menu}")
        
        # Analyze demand patterns
        print("\nDemand Analysis:")
        for r_id, restaurant in self.restaurants.items():
            demand = self.simulate_daily_demand(r_id)
            print(f"{restaurant.name} predicted demand:")
            for item, count in demand.items():
                print(f"  {item}: {count} units/day")

def main():
    """Main function to demonstrate the restaurant optimization system."""
    # Initialize optimizer
    optimizer = RestaurantOptimizer()
    optimizer.setup_sample_data()
    
    # Run optimization workflow
    print("Starting restaurant chain optimization...\n")
    
    # 1. Route Optimization
    print("1. Optimizing delivery routes...")
    routes = optimizer.optimize_delivery_routes("R1")
    
    # 2. Menu Planning
    print("\n2. Planning daily menus...")
    for r_id in optimizer.restaurants:
        menu = optimizer.plan_daily_menu(r_id)
        print(f"Restaurant {r_id} menu: {menu}")
    
    # 3. Demand Simulation
    print("\n3. Simulating customer demand...")
    demand = optimizer.simulate_daily_demand("R1")
    
    # 4. Inventory Management
    print("\n4. Updating inventory...")
    optimizer.update_inventory("R1", demand)
    
    # 5. Record Sample Order
    print("\n5. Recording sample order...")
    sample_order = Order(
        id="O1",
        restaurant_id="R1",
        items=["pizza", "pasta"],
        total_cost=95.0,
        timestamp=time.time()
    )
    optimizer.record_order(sample_order)
    
    # 6. Visualize Supply Chain
    print("\n6. Visualizing supply chain...")
    optimizer.visualize_supply_chain()
    
    # 7. Generate Report
    print("\n7. Generating optimization report...")
    optimizer.generate_report()

if __name__ == "__main__":
    main()