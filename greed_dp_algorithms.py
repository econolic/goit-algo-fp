"""
Food Selection Optimization: Greedy Algorithm vs Dynamic Programming
Implements two approaches to maximize calorie intake within budget constraints.
"""

import time
from typing import Dict, List, Tuple, TypedDict, Callable, Any
from dataclasses import dataclass

class FoodItem(TypedDict):
    """Type definition for food item properties."""
    cost: int
    calories: int

@dataclass
class OptimizationResult:
    """Result structure for optimization algorithms."""
    items: List[str]
    total_calories: int
    total_cost: int

def validate_inputs(items: Dict[str, FoodItem], budget: int) -> None:
    """Validate input parameters for optimization algorithms."""
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("Budget must be a non-negative integer")
    
    if not items:
        raise ValueError("Items dictionary cannot be empty")
        
    for name, data in items.items():
        if not isinstance(data.get("cost", 0), (int, float)) or data.get("cost", 0) <= 0:
            raise ValueError(f"Invalid cost for item {name}")
        if not isinstance(data.get("calories", 0), (int, float)) or data.get("calories", 0) <= 0:
            raise ValueError(f"Invalid calories for item {name}")


def greedy_algorithm(items: Dict[str, FoodItem], budget: int) -> OptimizationResult:
    """
    Greedy algorithm to select food items maximizing calorie-to-cost ratio.
    
    Time Complexity: O(n log n) due to sorting
    Space Complexity: O(n) for storing sorted items
    
    Args:
        items: Dictionary mapping food names to their FoodItem properties
        budget: Maximum budget constraint
        
    Returns:
        OptimizationResult containing selected items and totals
        
    Example:
        >>> items = {"pizza": {"cost": 50, "calories": 300}}
        >>> result = greedy_algorithm(items, 100)
        >>> result.items
        ['pizza']
    """
    # Validate inputs
    validate_inputs(items, budget)
    # Calculate calorie-to-cost ratio and sort by efficiency (descending)
    efficiency_items = [
        (name, data["calories"] / data["cost"], data["cost"], data["calories"])
        for name, data in items.items()
    ]
    efficiency_items.sort(key=lambda x: x[1], reverse=True)
    
    selected_items: List[str] = []
    total_cost = 0
    total_calories = 0
    
    # Greedily select items with highest efficiency that fit budget
    for name, ratio, cost, calories in efficiency_items:
        if total_cost + cost <= budget:
            selected_items.append(name)
            total_cost += cost
            total_calories += calories
    
    return OptimizationResult(
        items=selected_items,
        total_calories=total_calories,
        total_cost=total_cost
    )


def dynamic_programming(items: Dict[str, FoodItem], budget: int) -> OptimizationResult:
    """
    Dynamic programming approach to find optimal food selection.
    
    Time Complexity: O(n * budget) where n is number of items
    Space Complexity: O(n * budget) for DP table
    
    Args:
        items: Dictionary mapping food names to their cost and calories
        budget: Maximum budget constraint
        
    Returns:
        Dictionary containing optimal items list and maximum calories
    """
    item_list = list(items.items())
    n = len(item_list)
    
    # DP table: dp[i][w] = max calories using first i items with budget w
    dp = [[0 for _ in range(budget + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        name, data = item_list[i - 1]
        cost = data["cost"]
        calories = data["calories"]
        
        for w in range(budget + 1):
            # Don't take current item
            dp[i][w] = dp[i - 1][w]
            
            # Take current item if it fits and improves solution
            if cost <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - cost] + calories)
    
    # Backtrack to find selected items
    selected_items: List[str] = []
    w = budget
    total_cost = 0
    
    for i in range(n, 0, -1):
        # If value differs from above, item i-1 was selected
        if dp[i][w] != dp[i - 1][w]:
            name, data = item_list[i - 1]
            selected_items.append(name)
            total_cost += data["cost"]
            w -= data["cost"]
    
    return OptimizationResult(
        items=selected_items,
        total_calories=dp[n][budget],
        total_cost=total_cost
    )


def measure_performance(
    func: Callable[..., OptimizationResult],
    *args: Any
) -> Tuple[float, OptimizationResult]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function that returns an OptimizationResult
        *args: Arguments to pass to function
        
    Returns:
        Tuple of (execution_time, OptimizationResult)
    """
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    return execution_time, result


def compare_algorithms(items: Dict[str, FoodItem], budget: int) -> None:
    """
    Compare performance and results of both algorithms.
    
    Args:
        items: Dictionary of food items with their costs and calories
        budget: Budget constraint
    """
    print(f"=== Food Selection Optimization (Budget: ${budget}) ===\n")
    
    # Test greedy algorithm
    greedy_time, greedy_result = measure_performance(greedy_algorithm, items, budget)
    print("GREEDY ALGORITHM:")
    print(f"  Selected items: {greedy_result.items}")
    print(f"  Total calories: {greedy_result.total_calories}")
    print(f"  Total cost: ${greedy_result.total_cost}")
    print(f"  Execution time: {greedy_time:.6f} seconds")
    print(f"  Time complexity: O(n log n)")
    print(f"  Space complexity: O(n)\n")
    
    # Test dynamic programming
    dp_time, dp_result = measure_performance(dynamic_programming, items, budget)
    print("DYNAMIC PROGRAMMING:")
    print(f"  Selected items: {dp_result.items}")
    print(f"  Total calories: {dp_result.total_calories}")
    print(f"  Total cost: ${dp_result.total_cost}")
    print(f"  Execution time: {dp_time:.6f} seconds")
    print(f"  Time complexity: O(n × budget)")
    print(f"  Space complexity: O(n × budget)\n")
    
    # Analysis
    print("ANALYSIS:")
    calorie_diff = dp_result.total_calories - greedy_result.total_calories
    if calorie_diff > 0:
        print(f"  Dynamic programming found {calorie_diff} more calories")
        print(f"  Improvement: {(calorie_diff / greedy_result.total_calories * 100):.1f}%")
    elif calorie_diff == 0:
        print("  Both algorithms found the same optimal solution")
    else:
        print("  Greedy algorithm performed better (unexpected)")
    
    print(f"  Performance ratio: DP is {dp_time / greedy_time:.1f}x slower")


# Test data
item_data = {
    "pizza": {"cost": 50, "calories": 300},
    "hamburger": {"cost": 40, "calories": 250},
    "hot-dog": {"cost": 30, "calories": 200},
    "pepsi": {"cost": 10, "calories": 100},
    "cola": {"cost": 15, "calories": 220},
    "potato": {"cost": 25, "calories": 350}
}

# Convert to FoodItem type
items: Dict[str, FoodItem] = {
    name: FoodItem(cost=data["cost"], calories=data["calories"])
    for name, data in item_data.items()
}


def test_algorithms() -> None:
    """Test functions with assertions to verify correctness."""
    
    # Convert test items to FoodItem type
    food_items: Dict[str, FoodItem] = {
        name: FoodItem(cost=data["cost"], calories=data["calories"])
        for name, data in items.items()
    }
    
    # Test case 1: Small budget
    budget = 30
    greedy_result = greedy_algorithm(food_items, budget)
    dp_result = dynamic_programming(food_items, budget)
    
    # Assertions for greedy algorithm
    assert isinstance(greedy_result.items, list), "Greedy should return list of items"
    assert greedy_result.total_cost <= budget, "Greedy should not exceed budget"
    assert greedy_result.total_calories > 0, "Greedy should find some calories"
    
    # Assertions for dynamic programming
    assert isinstance(dp_result.items, list), "DP should return list of items"
    assert dp_result.total_cost <= budget, "DP should not exceed budget"
    assert dp_result.total_calories >= greedy_result.total_calories, "DP should be optimal"
    
    # Test case 2: Large budget
    budget = 100
    greedy_result = greedy_algorithm(food_items, budget)
    dp_result = dynamic_programming(food_items, budget)
    
    assert greedy_result.total_cost <= budget, "Greedy should respect budget constraint"
    assert dp_result.total_cost <= budget, "DP should respect budget constraint"
    assert dp_result.total_calories >= greedy_result.total_calories, "DP should be optimal"
    
    # Test case 3: Zero budget
    budget = 0
    greedy_result = greedy_algorithm(food_items, budget)
    dp_result = dynamic_programming(food_items, budget)
    
    assert len(greedy_result.items) == 0, "No items should be selected with zero budget"
    assert len(dp_result.items) == 0, "No items should be selected with zero budget"
    assert greedy_result.total_calories == 0, "Zero calories with zero budget"
    assert dp_result.total_calories == 0, "Zero calories with zero budget"
    
    print("✓ All assertions passed successfully!")


if __name__ == "__main__":
    # Run tests
    test_algorithms()
    print()
    
    # Compare algorithms with different budgets
    test_budgets = [30, 50, 80, 100]
    
    for budget in test_budgets:
        compare_algorithms(items, budget)
        print("-" * 60)