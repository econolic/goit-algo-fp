"""
Monte Carlo Simulation for Two Dice Rolling
==========================================
This module implements Monte Carlo simulation to estimate probabilities
of sums when rolling two dice, comparing results with analytical calculations.
"""

import random
import time
import itertools
from typing import Dict, List, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def roll_single_die() -> int:
    """
    Roll a single six-sided die.
    
    Returns:
        int: Random integer from 1 to 6 (inclusive)
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return random.randint(1, 6)


def roll_two_dice() -> Tuple[int, int]:
    """
    Roll two six-sided dice simultaneously.
    
    Returns:
        Tuple[int, int]: Results of both dice (die1, die2)
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return roll_single_die(), roll_single_die()


def calculate_dice_sum(die1: int, die2: int) -> int:
    """
    Calculate sum of two dice values.
    
    Args:
        die1 (int): First die value
        die2 (int): Second die value
        
    Returns:
        int: Sum of both dice
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    return die1 + die2


def monte_carlo_simulation(num_simulations: int) -> Dict[int, int]:
    """
    Perform Monte Carlo simulation for dice rolling.
    
    Args:
        num_simulations (int): Number of dice rolls to simulate
        
    Returns:
        Dict[int, int]: Count of each sum occurrence
        
    Time Complexity: O(n) where n = num_simulations
    Space Complexity: O(1) - fixed 11 possible sums (2-12)
    """
    sum_counts = Counter()
    
    start_time = time.perf_counter()
    
    for _ in range(num_simulations):
        die1, die2 = roll_two_dice()
        dice_sum = calculate_dice_sum(die1, die2)
        sum_counts[dice_sum] += 1
    
    end_time = time.perf_counter()
    print(f"Simulation completed in {end_time - start_time:.4f} seconds")
    
    return dict(sum_counts)


def calculate_probabilities(sum_counts: Dict[int, int], 
                          total_simulations: int) -> Dict[int, float]:
    """
    Calculate probabilities from simulation counts.
    
    Args:
        sum_counts (Dict[int, int]): Count of each sum occurrence
        total_simulations (int): Total number of simulations
        
    Returns:
        Dict[int, float]: Probability of each sum
        
    Time Complexity: O(k) where k = number of unique sums
    Space Complexity: O(k)
    """
    probabilities = {}
    
    for dice_sum in range(2, 13):  # All possible sums: 2-12
        count = sum_counts.get(dice_sum, 0)
        probabilities[dice_sum] = count / total_simulations
    
    return probabilities


def dice_sum_pmf(n: int = 2, m: int = 6) -> Dict[int, float]:
    """
    Probability mass function for the sum of *n* fair *m*-sided dice.
    
    Uses probability theory to generate all possible outcomes and calculate
    their frequencies, providing a general solution for any number and type of dice.
    
    Args:
        n (int): Number of dice (default: 2)
        m (int): Number of sides per die (default: 6)
        
    Returns:
        Dict[int, float]: Keys – possible sums, values – probabilities
        
    Raises:
        ValueError: If n or m are not positive integers
        
    Time Complexity: O(m^n) - generates all possible outcomes
    Space Complexity: O(m^n) - stores all outcome combinations
    """
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers")
    
    # Generate all possible tuples of outcomes using Cartesian product
    outcomes = itertools.product(range(1, m + 1), repeat=n)
    
    # Count frequency of each sum
    freq: Dict[int, int] = {}
    for outcome_tuple in outcomes:
        dice_sum = sum(outcome_tuple)
        freq.setdefault(dice_sum, 0)
        freq[dice_sum] += 1
    
    # Calculate probabilities: P(sum) = frequency / total_outcomes
    total_outcomes = m ** n
    return {sum_value: frequency / total_outcomes 
            for sum_value, frequency in freq.items()}


def get_analytical_probabilities() -> Dict[int, float]:
    """
    Get theoretical probabilities for two standard dice sums.
    
    Uses the general dice_sum_pmf function for two 6-sided dice.
    
    Returns:
        Dict[int, float]: Analytical probabilities for each sum (2-12)
        
    Time Complexity: O(6^2) = O(36) = O(1) for fixed dice configuration
    Space Complexity: O(6^2) = O(36) = O(1) for fixed dice configuration
    """
    return dice_sum_pmf(n=2, m=6)


def create_comparison_table(monte_carlo_probs: Dict[int, float],
                          analytical_probs: Dict[int, float]) -> pd.DataFrame:
    """
    Create comparison table of Monte Carlo vs Analytical probabilities.
    
    Args:
        monte_carlo_probs (Dict[int, float]): Monte Carlo probabilities
        analytical_probs (Dict[int, float]): Analytical probabilities
        
    Returns:
        pd.DataFrame: Comparison table
        
    Time Complexity: O(k) where k = number of sums
    Space Complexity: O(k)
    """
    data = []
    
    for dice_sum in range(2, 13):
        mc_prob = monte_carlo_probs[dice_sum]
        analytical_prob = analytical_probs[dice_sum]
        error = abs(mc_prob - analytical_prob)
        error_percent = (error / analytical_prob) * 100
        
        data.append({
            'Sum': dice_sum,
            'Monte Carlo (%)': f"{mc_prob * 100:.2f}",
            'Analytical (%)': f"{analytical_prob * 100:.2f}",
            'Absolute Error': f"{error:.6f}",
            'Relative Error (%)': f"{error_percent:.2f}"
        })
    
    return pd.DataFrame(data)


def plot_probability_comparison(monte_carlo_probs: Dict[int, float],
                              analytical_probs: Dict[int, float]) -> None:
    """
    Create visualization comparing Monte Carlo and analytical probabilities.
    
    Args:
        monte_carlo_probs (Dict[int, float]): Monte Carlo probabilities
        analytical_probs (Dict[int, float]): Analytical probabilities
        
    Time Complexity: O(k) where k = number of sums
    Space Complexity: O(k)
    """
    sums = list(range(2, 13))
    mc_values = [monte_carlo_probs[s] for s in sums]
    analytical_values = [analytical_probs[s] for s in sums]
    
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(sums))
    width = 0.35
    
    plt.bar(x - width/2, mc_values, width, label='Monte Carlo', 
            alpha=0.8, color='skyblue')
    plt.bar(x + width/2, analytical_values, width, label='Analytical', 
            alpha=0.8, color='lightcoral')
    
    plt.xlabel('Dice Sum')
    plt.ylabel('Probability')
    plt.title('Comparison of Monte Carlo vs Analytical Probabilities\n'
              'for Two Dice Rolling')
    plt.xticks(x, [str(s) for s in sums])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mc_val, anal_val) in enumerate(zip(mc_values, analytical_values)):
        plt.text(i - width/2, mc_val + 0.005, f'{mc_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, anal_val + 0.005, f'{anal_val:.3f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def analyze_convergence(num_simulations_list: List[int]) -> List[Tuple[int, float]]:
    """
    Analyze how Monte Carlo estimates converge to analytical values.
    
    Args:
        num_simulations_list (List[int]): List of simulation counts to test
        
    Time Complexity: O(n*m) where n = max simulations, m = len(num_simulations_list)
    Space Complexity: O(m)
    """
    analytical_probs = get_analytical_probabilities()
    convergence_data = []
    
    plt.figure(figsize=(12, 8))
    
    for num_sims in num_simulations_list:
        sum_counts = monte_carlo_simulation(num_sims)
        mc_probs = calculate_probabilities(sum_counts, num_sims)
        
        # Calculate mean absolute error
        total_error = sum(abs(mc_probs[s] - analytical_probs[s]) 
                         for s in range(2, 13))
        mean_error = total_error / 11
        convergence_data.append((num_sims, mean_error))
    
    # Plot convergence
    sim_counts, errors = zip(*convergence_data)
    plt.loglog(sim_counts, errors, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Number of Simulations')
    plt.ylabel('Mean Absolute Error')
    plt.title('Monte Carlo Convergence Analysis')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return convergence_data


def simulate_dice_rolls(n_dice: int = 2, n_sides: int = 6, iterations: int = 1000000) -> Dict[int, int]:
    """
    Run a Monte Carlo simulation for rolling multiple dice.
    
    Args:
        n_dice (int): Number of dice to roll (default: 2)
        n_sides (int): Number of sides on each die (default: 6)
        iterations (int): Number of iterations to run (default: 1,000,000)
    
    Returns:
        Dict[int, int]: Dictionary mapping sums to their raw counts
        
    Time Complexity: O(iterations)
    Space Complexity: O(n_dice * n_sides)
    """
    results = Counter()
    
    for _ in range(iterations):
        # Roll n_dice and sum their values
        roll_sum = sum(random.randint(1, n_sides) for _ in range(n_dice))
        results[roll_sum] += 1
    
    return dict(results)


def main() -> None:
    """
    Main function to run the complete Monte Carlo dice simulation analysis.
    
    Time Complexity: O(n) where n = number of simulations
    Space Complexity: O(1)
    """
    print("Monte Carlo Simulation for Two Dice Rolling")
    print("=" * 50)
    
    # Configuration
    num_simulations = 1_000_000
    random.seed(42)  # For reproducible results
    
    print(f"\nRunning {num_simulations:,} simulations...")
    
    # Run simulation
    start_time = time.perf_counter()
    sum_counts = monte_carlo_simulation(num_simulations)
    monte_carlo_probs = calculate_probabilities(sum_counts, num_simulations)
    end_time = time.perf_counter()
    
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    
    # Get analytical probabilities
    analytical_probs = get_analytical_probabilities()
    
    # Create comparison table
    comparison_df = create_comparison_table(monte_carlo_probs, analytical_probs)
    print("\nProbability Comparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Calculate overall accuracy
    total_error = sum(abs(monte_carlo_probs[s] - analytical_probs[s]) 
                     for s in range(2, 13))
    mean_error = total_error / 11
    print(f"\nMean Absolute Error: {mean_error:.6f}")
    print(f"Mean Relative Error: {mean_error / (1/11) * 100:.2f}%")
    
    # Create visualizations
    plot_probability_comparison(monte_carlo_probs, analytical_probs)
    
    # Analyze convergence
    print("\nAnalyzing convergence...")
    convergence_results = analyze_convergence([1000, 5000, 10000, 50000, 
                                             100000, 500000, 1000000])
    
    print("\nConvergence Analysis:")
    for sims, error in convergence_results:
        print(f"{sims:8,} simulations: Mean error = {error:.6f}")


# Test functions with assertions
def test_roll_single_die() -> None:
    """Test single die roll function."""
    for _ in range(100):
        result = roll_single_die()
        assert 1 <= result <= 6, f"Die result {result} out of range"
    print("✓ test_roll_single_die passed")


def test_roll_two_dice() -> None:
    """Test two dice roll function."""
    for _ in range(100):
        die1, die2 = roll_two_dice()
        assert 1 <= die1 <= 6, f"Die1 result {die1} out of range"
        assert 1 <= die2 <= 6, f"Die2 result {die2} out of range"
    print("✓ test_roll_two_dice passed")


def test_calculate_dice_sum() -> None:
    """Test dice sum calculation."""
    assert calculate_dice_sum(1, 1) == 2
    assert calculate_dice_sum(6, 6) == 12
    assert calculate_dice_sum(3, 4) == 7
    print("✓ test_calculate_dice_sum passed")


def test_monte_carlo_simulation() -> None:
    """Test Monte Carlo simulation function."""
    result = monte_carlo_simulation(1000)
    assert isinstance(result, dict)
    assert all(2 <= k <= 12 for k in result.keys())
    assert sum(result.values()) == 1000
    print("✓ test_monte_carlo_simulation passed")


def test_calculate_probabilities() -> None:
    """Test probability calculation."""
    test_counts = {2: 28, 7: 167, 12: 28}  # Sample data
    test_total = 1000
    probs = calculate_probabilities(test_counts, test_total)
    
    assert abs(probs[2] - 0.028) < 0.001
    assert abs(probs[7] - 0.167) < 0.001
    assert abs(probs[12] - 0.028) < 0.001
    print("✓ test_calculate_probabilities passed")


def test_analytical_probabilities() -> None:
    """Test analytical probability calculation."""
    probs = get_analytical_probabilities()
    
    # Test specific known values
    assert abs(probs[2] - 1/36) < 0.001
    assert abs(probs[7] - 6/36) < 0.001
    assert abs(probs[12] - 1/36) < 0.001
    
    # Test that probabilities sum to 1
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.001
    print("✓ test_analytical_probabilities passed")


def test_dice_sum_pmf() -> None:
    """Test the general dice PMF function."""
    # Test two standard dice
    pmf = dice_sum_pmf(2, 6)
    assert abs(pmf[7] - 6/36) < 1e-12
    assert abs(sum(pmf.values()) - 1) < 1e-12
    
    # Test single die
    single_die = dice_sum_pmf(1, 6)
    assert len(single_die) == 6
    assert all(abs(prob - 1/6) < 1e-12 for prob in single_die.values())
    
    # Test three dice
    three_dice = dice_sum_pmf(3, 6)
    assert min(three_dice.keys()) == 3  # minimum sum: 1+1+1
    assert max(three_dice.keys()) == 18  # maximum sum: 6+6+6
    assert abs(sum(three_dice.values()) - 1) < 1e-12
    
    # Test edge cases
    try:
        dice_sum_pmf(0, 6)
        assert False, "Should raise ValueError for n=0"
    except ValueError:
        pass
    
    try:
        dice_sum_pmf(2, 0)
        assert False, "Should raise ValueError for m=0"
    except ValueError:
        pass
    
    print("✓ test_dice_sum_pmf passed")


def run_all_tests() -> None:
    """Run all test functions."""
    print("Running tests...")
    test_roll_single_die()
    test_roll_two_dice()
    test_calculate_dice_sum()
    test_monte_carlo_simulation()
    test_calculate_probabilities()
    test_dice_sum_pmf()
    test_analytical_probabilities()
    print("All tests passed!\n")


if __name__ == "__main__":
    # Run tests first
    run_all_tests()
    
    # Run main simulation
    main()
