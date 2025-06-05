"""
Singly Linked List Implementation with Advanced Operations
=======================================================

This module implements a singly linked list with operations for reversal, 
sorting (merge sort), and merging two sorted lists.
It includes comprehensive tests with assertions and performance measurements.

"""

from typing import Optional, List, Tuple
import time
import random
from dataclasses import dataclass


@dataclass
class ListNode:
    """Node class for singly linked list."""
    val: int
    next: Optional['ListNode'] = None


class LinkedList:
    """Singly linked list implementation with advanced operations."""
    
    def __init__(self, values: Optional[List[int]] = None):
        """Initialize linked list from list of values."""
        self.head: Optional[ListNode] = None
        if values:
            self._build_from_list(values)
    
    def _build_from_list(self, values: List[int]) -> None:
        """Build linked list from list of values."""
        if not values:
            return
        
        self.head = ListNode(values[0])
        current = self.head
        for val in values[1:]:
            current.next = ListNode(val)
            current = current.next
    
    def to_list(self) -> List[int]:
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.val)
            current = current.next
        return result
    
    def __str__(self) -> str:
        """String representation of linked list."""
        return " -> ".join(map(str, self.to_list())) + " -> None"


def reverse_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a singly linked list by changing node pointers.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Args:
        head: Head of the linked list to reverse
        
    Returns:
        New head of the reversed linked list
    """
    prev = None
    current = head
    
    while current:
        next_temp = current.next  # Store next node
        current.next = prev       # Reverse the link
        prev = current            # Move prev forward
        current = next_temp       # Move current forward
    
    return prev  # prev is now the new head


def merge_sort_linked_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Sort a singly linked list using merge sort algorithm.
    
    Time Complexity: O(n log n)
    Space Complexity: O(log n) due to recursion stack
    
    Args:
        head: Head of the linked list to sort
        
    Returns:
        Head of the sorted linked list
    """
    if not head or not head.next:
        return head
    
    # Split the list into two halves
    mid = _get_middle(head)
    right_head = mid.next
    mid.next = None  # Break the connection
    
    # Recursively sort both halves
    left = merge_sort_linked_list(head)
    right = merge_sort_linked_list(right_head)
    
    # Merge the sorted halves
    return _merge_sorted_lists(left, right)


def _get_middle(head: ListNode) -> ListNode:
    """
    Find the middle node of a linked list using two pointers.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    slow = head
    fast = head
    prev = None
    
    while fast and fast.next:
        prev = slow
        # Add a check to ensure slow is not None before accessing slow.next
        # In a correctly formed list and with the loop condition, slow should not be None here,
        # but this check prevents AttributeError if an unexpected state is reached.
        if slow is None:
            break # Prevent accessing .next on None
        slow = slow.next
        fast = fast.next.next
    
    return prev if prev else head


def _merge_sorted_lists(list1: Optional[ListNode], 
                       list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Merge two sorted linked lists into one sorted list.
    
    Time Complexity: O(n + m) where n, m are lengths of lists
    Space Complexity: O(1)
    
    Args:
        list1: Head of first sorted linked list
        list2: Head of second sorted linked list
        
    Returns:
        Head of merged sorted linked list
    """
    dummy = ListNode(0)  # Dummy node to simplify logic
    current = dummy
    
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next
    
    # Append remaining nodes
    current.next = list1 or list2
    
    return dummy.next


def merge_two_sorted_lists(list1: Optional[ListNode], 
                          list2: Optional[ListNode]) -> Optional[ListNode]:
    """
    Public interface for merging two sorted linked lists.
    
    Time Complexity: O(n + m)
    Space Complexity: O(1)
    
    Args:
        list1: Head of first sorted linked list
        list2: Head of second sorted linked list
        
    Returns:
        Head of merged sorted linked list
    """
    return _merge_sorted_lists(list1, list2)


def measure_performance(func, *args) -> Tuple[Optional[ListNode], float]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Arguments for the function
        
    Returns:
        Tuple of (result, execution_time_in_seconds)
    """
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time


# Test Suite with Assertions and Performance Measurements
def run_tests():
    """Run tests with assertions and performance measurements."""
    
    print("=" * 60)
    print("SINGLY LINKED LIST OPERATIONS - TESTING")
    print("=" * 60)
    
    # Test 1: Reverse Linked List
    print("\n1. REVERSE LINKED LIST TESTING")
    print("-" * 40)
    
    # Test case 1: Normal list
    test_list1 = LinkedList([1, 2, 3, 4, 5])
    print(f"Original: {test_list1}")
    
    reversed_head, reverse_time = measure_performance(reverse_linked_list, test_list1.head)
    reversed_list = LinkedList()
    reversed_list.head = reversed_head
    print(f"Reversed: {reversed_list}")
    print(f"Execution time: {reverse_time:.6f} seconds")
    
    # Assertions for reverse
    assert reversed_list.to_list() == [5, 4, 3, 2, 1], "Reverse test 1 failed"
    
    # Test case 2: Single element
    single_node = LinkedList([42])
    reversed_single, _ = measure_performance(reverse_linked_list, single_node.head)
    reversed_single_list = LinkedList()
    reversed_single_list.head = reversed_single
    assert reversed_single_list.to_list() == [42], "Reverse test 2 failed"
    
    # Test case 3: Empty list
    empty_reversed, _ = measure_performance(reverse_linked_list, None)
    assert empty_reversed is None, "Reverse test 3 failed"
    
    print("✓ All reverse tests passed")
    
    # Test 2: Merge Sort
    print("\n2. MERGE SORT TESTING")
    print("-" * 40)
    
    # Test case 1: Unsorted list
    unsorted_values = [64, 34, 25, 12, 22, 11, 90, 5, 77, 30]
    unsorted_list = LinkedList(unsorted_values)
    print(f"Original: {unsorted_list}")
    
    sorted_head, sort_time = measure_performance(merge_sort_linked_list, unsorted_list.head)
    sorted_list = LinkedList()
    sorted_list.head = sorted_head
    print(f"Sorted: {sorted_list}")
    print(f"Execution time: {sort_time:.6f} seconds")
    
    expected_sorted = sorted(unsorted_values)
    assert sorted_list.to_list() == expected_sorted, "Sort test 1 failed"
    
    # Test case 2: Already sorted
    already_sorted = LinkedList([1, 2, 3, 4, 5])
    sorted_head2, sort_time2 = measure_performance(merge_sort_linked_list, already_sorted.head)
    sorted_list2 = LinkedList()
    sorted_list2.head = sorted_head2
    assert sorted_list2.to_list() == [1, 2, 3, 4, 5], "Sort test 2 failed"
    
    # Test case 3: Reverse sorted
    reverse_sorted = LinkedList([5, 4, 3, 2, 1])
    sorted_head3, sort_time3 = measure_performance(merge_sort_linked_list, reverse_sorted.head)
    sorted_list3 = LinkedList()
    sorted_list3.head = sorted_head3
    assert sorted_list3.to_list() == [1, 2, 3, 4, 5], "Sort test 3 failed"
    
    print("✓ All sort tests passed")
    
    # Test 3: Merge Two Sorted Lists
    print("\n3. MERGE TWO SORTED LISTS TESTING")
    print("-" * 40)
    
    # Test case 1: Normal merge
    list_a = LinkedList([1, 3, 5, 7])
    list_b = LinkedList([2, 4, 6, 8])
    print(f"List A: {list_a}")
    print(f"List B: {list_b}")
    
    merged_head, merge_time = measure_performance(
        merge_two_sorted_lists, list_a.head, list_b.head
    )
    merged_list = LinkedList()
    merged_list.head = merged_head
    print(f"Merged: {merged_list}")
    print(f"Execution time: {merge_time:.6f} seconds")
    
    assert merged_list.to_list() == [1, 2, 3, 4, 5, 6, 7, 8], "Merge test 1 failed"
    
    # Test case 2: One empty list
    list_c = LinkedList([1, 2, 3])
    merged_head2, _ = measure_performance(merge_two_sorted_lists, list_c.head, None)
    merged_list2 = LinkedList()
    merged_list2.head = merged_head2
    assert merged_list2.to_list() == [1, 2, 3], "Merge test 2 failed"
    
    # Test case 3: Different lengths
    list_d = LinkedList([1, 5, 9])
    list_e = LinkedList([2, 3, 4, 6, 7, 8, 10])
    merged_head3, _ = measure_performance(merge_two_sorted_lists, list_d.head, list_e.head)
    merged_list3 = LinkedList()
    merged_list3.head = merged_head3
    assert merged_list3.to_list() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Merge test 3 failed"
    
    print("✓ All merge tests passed")
    
    # Performance Analysis with Larger Datasets
    print("\n4. PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    sizes = [100, 500, 1000, 2000]
    
    for size in sizes:
        print(f"\nDataset size: {size} elements")
        
        # Generate random data
        random_data = [random.randint(1, 1000) for _ in range(size)]
        test_list = LinkedList(random_data)
        
        # Test reverse performance
        _, reverse_time = measure_performance(reverse_linked_list, test_list.head)
        print(f"  Reverse time: {reverse_time:.6f}s")
        
        # Test sort performance
        test_list_sort = LinkedList(random_data.copy())
        _, sort_time = measure_performance(merge_sort_linked_list, test_list_sort.head)
        print(f"  Sort time: {sort_time:.6f}s")
        
        # Test merge performance (two halves)
        mid = size // 2
        list1 = LinkedList(sorted(random_data[:mid]))
        list2 = LinkedList(sorted(random_data[mid:]))
        _, merge_time = measure_performance(merge_two_sorted_lists, list1.head, list2.head)
        print(f"  Merge time: {merge_time:.6f}s")


if __name__ == "__main__":
    run_tests()
    
    print("\n" + "=" * 60)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 60)
    print("1. Reverse Linked List:")
    print("   - Time Complexity: O(n)")
    print("   - Space Complexity: O(1)")
    print("   - Performance: Linear growth with input size")
    
    print("\n2. Merge Sort:")
    print("   - Time Complexity: O(n log n)")
    print("   - Space Complexity: O(log n) - recursion stack")
    print("   - Performance: Efficient for large datasets")
    
    print("\n3. Merge Two Sorted Lists:")
    print("   - Time Complexity: O(n + m)")
    print("   - Space Complexity: O(1)")
    print("   - Performance: Linear in total input size")
    
    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("• Reverse operation is highly efficient with constant space usage")
    print("• Merge sort provides optimal O(n log n) sorting for linked lists")
    print("• Merging sorted lists is optimal with linear time complexity")
    print("• All implementations are memory-efficient and production-ready")
    print("• Performance scales predictably according to theoretical complexity")
