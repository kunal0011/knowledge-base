---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 23: Merge k Sorted Lists"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 23: Merge k Sorted Lists

Below is a structured, interview-ready explanation of **LeetCode 23 – Merge k Sorted Lists**, aligned with how this problem is typically evaluated in senior-level DSA interviews.

---

## 1. Problem Statement

You are given an array of `k` linked lists, where each linked list is **sorted in ascending order**.

Your task is to **merge all the linked lists into one sorted linked list** and return its head.

### Constraints (important for complexity reasoning)

* `k` = number of linked lists
* Total number of nodes across all lists = `N`
* Each list may be empty

---

## 2. Key Observation

1. Each individual linked list is already sorted.
2. At any point during merging, the **next smallest element** must be among the **current head nodes** of the remaining lists.
3. Therefore, we need a data structure that:

   * Efficiently returns the minimum element
   * Supports repeated insertions as we advance through lists

This naturally leads to a **Min-Heap (Priority Queue)**.

---

## 3. Why Priority Queue (Min-Heap)?

A priority queue allows:

* **O(log k)** insertion
* **O(log k)** removal of the minimum element

### Strategy

1. Push the **head node of each non-empty list** into a min-heap.
2. Repeatedly:

   * Extract the smallest node
   * Append it to the merged list
   * If the extracted node has a `next`, push that `next` node into the heap
3. Continue until the heap is empty.

### Complexity

* **Time Complexity:** `O(N log k)`

  * Each of the `N` nodes is pushed and popped once
* **Space Complexity:** `O(k)`

  * Heap stores at most one node from each list

This is optimal and expected in interviews.

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List, Optional
import heapq

class ListNode:
    def __init__(self, val: int = 0, next: Optional["ListNode"] = None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        min_heap = []

        # Push the head of each non-empty list into heap
        for i, node in enumerate(lists):
            if node:
                # (value, unique_index, node)
                heapq.heappush(min_heap, (node.val, i, node))

        dummy = ListNode(0)
        current = dummy

        while min_heap:
            _, i, node = heapq.heappop(min_heap)
            current.next = node
            current = current.next

            if node.next:
                heapq.heappush(min_heap, (node.next.val, i, node.next))

        return dummy.next
```

### Why include `i` (index)?

Python’s heap requires elements to be comparable.  
If two nodes have the same value, the heap uses the next tuple element (`i`) to break ties safely.

---

## 5. Worked Example (Step-by-Step)

### Input

```
lists = [
  1 → 4 → 5,
  1 → 3 → 4,
  2 → 6
]
```

### Initial Heap (push all heads)

```
Heap: [(1, L0), (1, L1), (2, L2)]
```

---

### Iteration 1

* Pop: `1 (L0)`
* Result: `1`
* Push next from L0 → `4`

```
Heap: [(1, L1), (2, L2), (4, L0)]
```

---

### Iteration 2

* Pop: `1 (L1)`
* Result: `1 → 1`
* Push next from L1 → `3`

```
Heap: [(2, L2), (4, L0), (3, L1)]
```

---

### Iteration 3

* Pop: `2 (L2)`
* Result: `1 → 1 → 2`
* Push next from L2 → `6`

```
Heap: [(3, L1), (4, L0), (6, L2)]
```

---

### Iteration 4

* Pop: `3 (L1)`
* Result: `1 → 1 → 2 → 3`
* Push next from L1 → `4`

```
Heap: [(4, L0), (6, L2), (4, L1)]
```

---

### Iteration 5

* Pop: `4 (L0)`
* Result: `1 → 1 → 2 → 3 → 4`
* Push next from L0 → `5`

```
Heap: [(4, L1), (6, L2), (5, L0)]
```

---

### Iteration 6

* Pop: `4 (L1)`
* Result: `1 → 1 → 2 → 3 → 4 → 4`
* L1 exhausted

---

### Iteration 7

* Pop: `5 (L0)`
* Result: `1 → 1 → 2 → 3 → 4 → 4 → 5`
* L0 exhausted

---

### Iteration 8

* Pop: `6 (L2)`
* Result: `1 → 1 → 2 → 3 → 4 → 4 → 5 → 6`

Heap is empty → **done**.

---

## 6. Final Output

```
1 → 1 → 2 → 3 → 4 → 4 → 5 → 6
```

---

## 7. Interview Notes (Important)

* This problem is a **classic k-way merge**, identical in concept to:

  * Merging k sorted arrays
  * External sorting
* Alternative approach:

  * **Divide and Conquer** (merge lists pairwise)
  * Same `O(N log k)` time, but more complex to implement
* Priority Queue solution is:

  * Cleaner
  * Easier to reason about
  * Preferred in interviews

If you want, I can also provide:

* Divide-and-Conquer solution
* Dry-run diagram with heap states
* Common mistakes and edge cases
* Comparison with LeetCode 21 (merge two lists)