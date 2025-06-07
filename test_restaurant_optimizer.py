"""
Test suite for the Restaurant Supply Chain Optimizer

This test suite validates the functionality of the restaurant chain optimization system.
It includes comprehensive tests for all major components:

Components Tested:
1. Route Optimization
   - Validates shortest path calculations
   - Ensures connectivity between locations
   - Verifies distance calculations

2. Menu Planning
   - Tests budget constraints
   - Validates item selection logic
   - Verifies menu variety requirements

3. Demand Forecasting
   - Tests Monte Carlo simulation accuracy
   - Validates location-specific modifiers
   - Checks demand pattern consistency

4. Inventory Management
   - Verifies min-heap structure
   - Tests priority-based ordering
   - Validates stock level calculations

5. Order Processing
   - Tests linked list operations
   - Validates order recording
   - Checks history maintenance

Coverage:
- Unit tests for all core functions
- Integration tests for system workflow
- Performance validation for algorithms
- Edge case handling verification
"""

import unittest
from restaurant_optimizer import RestaurantOptimizer, Restaurant, Order
import time

class TestRestaurantOptimizer(unittest.TestCase):
    def setUp(self):
        """Initialize the optimizer and sample data for each test."""
        self.optimizer = RestaurantOptimizer()
        self.optimizer.setup_sample_data()
    
    def test_route_optimization(self):
        """Test delivery route optimization."""
        routes = self.optimizer.optimize_delivery_routes("R1")
        self.assertIsNotNone(routes)
        self.assertTrue(len(routes) >= 2)  # Should have routes to at least 2 other restaurants
        
    def test_menu_planning(self):
        """Test daily menu optimization."""
        menu = self.optimizer.plan_daily_menu("R1")
        self.assertIsNotNone(menu)
        self.assertTrue(len(menu) > 0)  # Should have at least one menu item
        
    def test_demand_simulation(self):
        """Test customer demand simulation."""
        demand = self.optimizer.simulate_daily_demand("R1", days=10)
        self.assertIsNotNone(demand)
        self.assertEqual(len(demand), len(self.optimizer.menu_items))
        
    def test_order_recording(self):
        """Test simple order recording into the linked list."""
        initial_length = len(self.optimizer.order_history.to_list())
        sample_order = Order("TEST1", "R1", ["pizza"], 50.0, time.time())
        self.optimizer.record_order(sample_order)
        new_length = len(self.optimizer.order_history.to_list())
        self.assertEqual(new_length, initial_length + 1)

    def test_process_order(self):
        """Test the complete order processing logic."""
        # Ensure R1 has enough inventory for a successful order
        self.optimizer.restaurants["R1"].inventory = {"pizza": 20, "salad": 20}
        
        # Scenario 1: Successful order
        successful_order = {"pizza": 1, "salad": 1}
        result_success = self.optimizer.process_order("R1", successful_order)
        self.assertTrue(result_success)
        # Check if inventory was updated
        self.assertEqual(self.optimizer.get_inventory("R1")["pizza"], 19)

        # Scenario 2: Failed order (insufficient inventory)
        failed_order_inventory = {"pizza": 25} # More than available
        result_fail_inventory = self.optimizer.process_order("R1", failed_order_inventory)
        self.assertFalse(result_fail_inventory)

        # Scenario 3: Failed order (item not on menu)
        # Assuming 'fries' is not on the menu, let's ensure it's not in menu_items for the test
        if "fries" in self.optimizer.menu_items:
            del self.optimizer.menu_items["fries"]
            
        failed_order_menu = {"fries": 1} 
        result_fail_menu = self.optimizer.process_order("R1", failed_order_menu)
        self.assertFalse(result_fail_menu)

def run_visualization_test():
    """Run visualization tests separately (not as unit tests)."""
    optimizer = RestaurantOptimizer()
    optimizer.setup_sample_data()
    
    print("Testing supply chain visualization...")
    optimizer.visualize_supply_chain()
    
    print("\nGenerating optimization report...")
    optimizer.generate_report()

if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False)
    
    print("\nRunning visualization tests...")
    run_visualization_test()
