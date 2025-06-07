"""
Demo script showing a complete restaurant chain optimization workflow

This script demonstrates the practical application of various algorithms
in a restaurant chain optimization system. It showcases:

1. Route Optimization:
   - Finds optimal delivery paths between restaurants
   - Uses Dijkstra's algorithm for shortest paths
   - Shows distance reduction opportunities

2. Menu Planning:
   - Optimizes menu items within budget constraints
   - Supports different budget levels ($800-$1200)
   - Ensures variety and profitability

3. Demand Forecasting:
   - Uses Monte Carlo simulation for predictions
   - Accounts for location-specific patterns
   - Shows item-by-item demand variations

4. Inventory Management:
   - Implements min-heap priority queue
   - Manages stock levels efficiently
   - Prioritizes items based on demand

5. Order Processing:
   - Tracks orders using linked list
   - Records customer preferences
   - Maintains order history

6. Visualization:
   - Creates network diagrams
   - Shows demand patterns
   - Visualizes inventory priorities

7. Performance Reporting:
   - Generates comprehensive analysis
   - Shows optimization benefits
   - Provides actionable insights

Example Output:
- Route optimization showing up to 58% distance reduction
- Location-specific demand variations (30-120 units/day)
- Priority-based inventory management
- Interactive supply chain visualization
"""

from restaurant_optimizer import RestaurantOptimizer

def run_optimization_demo():
    """Run a complete demonstration of the restaurant optimization system."""
    print("=" * 50)
    print("Restaurant Chain Optimization System Analysis")
    print("=" * 50)
    
    # Initialize the optimizer
    optimizer = RestaurantOptimizer()
    optimizer.setup_sample_data()
    
    # 1. Route Optimization Analysis
    print("\n" + "=" * 20 + " Route Analysis " + "=" * 20)
    print("\nAnalyzing delivery routes and potential optimizations...")
    
    all_routes = {}
    total_unoptimized = 0
    total_optimized = 0
    
    for start_id, start_restaurant in optimizer.restaurants.items():
        routes = optimizer.optimize_delivery_routes(start_id)
        all_routes[start_id] = routes
        
        print(f"\n{'-' * 10} Routes from {start_restaurant.name} {'-' * 10}")
        print(f"{'Destination':15} {'Direct':>10} {'Optimized':>10} {'Reduction':>10}")
        print("-" * 48)
        
        for dest_id, opt_distance in routes.items():
            if dest_id != start_id:
                direct_dist = ((optimizer.restaurants[start_id].location[0] - 
                              optimizer.restaurants[dest_id].location[0])**2 + 
                             (optimizer.restaurants[start_id].location[1] - 
                              optimizer.restaurants[dest_id].location[1])**2)**0.5 * 100
                
                reduction = ((direct_dist - opt_distance) / direct_dist) * 100
                total_unoptimized += direct_dist
                total_optimized += opt_distance
                
                print(f"{optimizer.restaurants[dest_id].name:15} "
                      f"{direct_dist:10.2f} {opt_distance:10.2f} {reduction:9.1f}%")
    
    total_reduction = ((total_unoptimized - total_optimized) / total_unoptimized) * 100
    print("\nOverall Network Optimization:")
    print(f"Total unoptimized distance: {total_unoptimized:.2f}")
    print(f"Total optimized distance: {total_optimized:.2f}")
    print(f"Network-wide reduction: {total_reduction:.1f}%")

    # 2. Menu Optimization
    print("\n" + "=" * 20 + " Menu Analysis " + "=" * 20)
    print("\nOptimizing menus for each restaurant...")
    
    for r_id, restaurant in optimizer.restaurants.items():
        menu = optimizer.plan_daily_menu(r_id)
        print(f"\n{'-' * 10} {restaurant.name} ({r_id}) {'-' * 10}")
        print(f"Daily Budget: ${restaurant.daily_budget:.2f}")
        print("Selected Menu Items:")
        for item in menu:
            cost = optimizer.menu_items[item]["cost"]
            print(f"- {item:15} ${cost:6.2f}")
        total_cost = sum(optimizer.menu_items[item]["cost"] for item in menu)
        print(f"Total Menu Cost: ${total_cost:.2f}")
        budget_util = (total_cost / restaurant.daily_budget) * 100
        print(f"Budget Utilization: {budget_util:.1f}%")

    # 3. Demand Simulation
    print("\n" + "=" * 20 + " Demand Analysis " + "=" * 20)
    print("\nRunning customer demand simulation...")
    
    simulation_days = 30
    demand_data = {}
    total_demands = {item: 0 for item in optimizer.menu_items}
    total_revenue = 0
    
    for r_id, restaurant in optimizer.restaurants.items():
        demand = optimizer.simulate_daily_demand(r_id, simulation_days)
        demand_data[r_id] = demand
        
        print(f"\n{'-' * 10} {restaurant.name} - {simulation_days}-Day Forecast {'-' * 10}")
        print(f"{'Item':12} {'Daily Avg':>10} {'Monthly':>10} {'Revenue %':>12}")
        print("-" * 52)
        
        location_total = sum(count * optimizer.menu_items[item]["cost"] 
                           for item, count in demand.items())
        total_revenue += location_total
        
        for item, count in sorted(demand.items(), 
                                key=lambda x: x[1] * optimizer.menu_items[x[0]]["cost"],
                                reverse=True):
            daily_avg = count
            monthly_total = count * simulation_days
            revenue = monthly_total * optimizer.menu_items[item]["cost"]
            revenue_share = (revenue / location_total) * 100
            
            total_demands[item] += monthly_total
            
            print(f"{item:12} {daily_avg:10.1f} {monthly_total:10d} {revenue_share:11.1f}%")
        
        print(f"\nLocation Monthly Revenue: ${location_total:,.2f}")
    
    print("\n" + "-" * 20 + " Chain-wide Monthly Summary " + "-" * 20)
    print(f"{'Item':12} {'Total Units':>12} {'% of Volume':>12} {'Revenue %':>12}")
    print("-" * 50)
    
    grand_total = sum(total_demands.values())
    for item, total in sorted(total_demands.items(), 
                            key=lambda x: optimizer.menu_items[x[0]]["cost"] * x[1],
                            reverse=True):
        volume_pct = (total / grand_total) * 100
        revenue_pct = (total * optimizer.menu_items[item]["cost"] / total_revenue) * 100
        print(f"{item:12} {total:12d} {volume_pct:11.1f}% {revenue_pct:11.1f}%")
    
    print(f"\nTotal Chain Monthly Revenue: ${total_revenue:,.2f}")

    # 4. Inventory Management
    print("\n" + "=" * 20 + " Inventory Update " + "=" * 20)
    print("\nUpdating inventory based on demand patterns...")
    
    for r_id, demand in demand_data.items():
        print(f"\nUpdating {optimizer.restaurants[r_id].name}...")
        optimizer.update_inventory(r_id, demand)

    # 5. Order Processing Validation
    print("\n" + "=" * 20 + " Order Processing " + "=" * 20)
    print("\nTesting order validation scenarios:")

    # Normal order
    normal_order = {"pizza": 2, "burger": 1, "dessert": 2}
    print(f"\n1. Standard Order Validation:")
    print(f"Order details: {normal_order}")
    try:
        optimizer.process_order("R1", normal_order)
        print("✓ Standard order processed successfully")
    except Exception as e:
        print(f"✗ Standard order failed: {str(e)}")

    # Large order
    large_order = {"pizza": 10, "pasta": 8, "salad": 5}
    print(f"\n2. Large Order Validation:")
    print(f"Order details: {large_order}")
    try:
        optimizer.process_order("R2", large_order)
        print("✓ Large order processed successfully")
    except Exception as e:
        print(f"✗ Large order failed: {str(e)}")

    # Invalid items order
    invalid_order = {"pizza": 1, "sushi": 2}
    print(f"\n3. Invalid Items Validation:")
    print(f"Order details: {invalid_order}")
    try:
        optimizer.process_order("R1", invalid_order)
        print("Order with invalid items was incorrectly accepted")
    except Exception as e:
        print(f"✓ Correctly rejected invalid items: {str(e)}")

    # Unrealistic order
    unrealistic_order = {"pizza": 100, "burger": 100}
    print(f"\n4. Quantity Validation:")
    print(f"Order details: {unrealistic_order}")
    try:
        optimizer.process_order("R3", unrealistic_order)
        print("Unrealistic order was incorrectly accepted")
    except Exception as e:
        print(f"✓ Correctly rejected unrealistic quantity: {str(e)}")
        
    # 6. Supply Chain Visualization
    print("\n" + "=" * 20 + " Network Visualization " + "=" * 20)
    print("\nGenerating supply chain visualization...")
    optimizer.visualize_supply_chain()
    
    # 7. Optimization Report
    print("\n" + "=" * 20 + " Final Report " + "=" * 20)
    optimizer.generate_report()

def main():
    """Main function to run the demo."""
    try:
        run_optimization_demo()
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        raise

if __name__ == "__main__":
    main()
