---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 912: Sort an Array"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 912: Sort an Array

Below is a **structured, interview-ready explanation** of **LeetCode 912 – Sort an Array**, focused specifically on the **Priority Queue (Heap) technique**, as requested.

---

## 1. Problem Statement

**LeetCode 912 – Sort an Array**

Given an array of integers `nums`, sort the array in **ascending order** and return it.

**Constraints**

* `1 ≤ nums.length ≤ 50,000`
* `-50,000 ≤ nums[i] ≤ 50,000`

You must sort the array **without using built-in sort functions**.

---

## 2. Key Observation

* This is a **fundamental sorting problem**, but the constraint *“do not use built-in sort”* forces us to implement a sorting algorithm.
* Multiple algorithms are acceptable:

  * Merge Sort
  * Quick Sort
  * Heap Sort
  * Counting Sort (because range is bounded)
* The **Priority Queue (Heap)** approach is:

  * Simple to reason about
  * Guaranteed **O(n log n)** time
  * Stable in behavior (no worst-case degradation like Quick Sort)

---

## 3. Priority Queue (Heap) Technique

### Core Idea

1. Insert all elements into a **min-heap**
2. Repeatedly extract the smallest element
3. Append extracted elements to the result array

This process naturally produces a sorted array.

---

### Why This Works

* A **min-heap** always keeps the smallest element at the top
* Each insertion and removal costs `O(log n)`
* Total operations:

  * `n` inserts
  * `n` removals  
    → **O(n log n)**

---

### Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | `O(n log n)` |
| Space Complexity | `O(n)` (heap + result array) |
| In-place | ❌ No |

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        min_heap: List[int] = []
        
        # Step 1: Build the heap
        for num in nums:
            heapq.heappush(min_heap, num)
        
        # Step 2: Extract elements in sorted order
        sorted_nums: List[int] = []
        while min_heap:
            sorted_nums.append(heapq.heappop(min_heap))
        
        return sorted_nums
```

---

## 5. Worked Example (Step-by-Step)

### Example 1

**Input**

```text
nums = [5, 2, 3, 1]
```

---

### Step 1: Insert into Min-Heap

| Operation | Heap State |
| --- | --- |
| push 5 | [5] |
| push 2 | [2, 5] |
| push 3 | [2, 5, 3] |
| push 1 | [1, 2, 3, 5] |

*(Heap structure ensures smallest element is always on top)*

---

### Step 2: Extract Elements

| pop() | Result Array | Heap After Pop |
| --- | --- | --- |
| 1 | [1] | [2, 5, 3] |
| 2 | [1, 2] | [3, 5] |
| 3 | [1, 2, 3] | [5] |
| 5 | [1, 2, 3, 5] | [] |

---

### Final Output

```
[1, 2, 3, 5]
```

---

## 6. Another Example

### Example 2

**Input**

```text
nums = [5, 1, 1, 2, 0, 0]
```

**Sorted Output**

```
[0, 0, 1, 1, 2, 5]
```

Heap handles:

* Duplicate values
* Negative numbers
* Arbitrary ordering

without any special logic.

---

## 7. When to Prefer Priority Queue for This Problem

Use Heap Sort / Priority Queue when:

* You want **predictable O(n log n)** performance
* You want simpler code than in-place Quick Sort
* You do not need in-place sorting
* Stability and worst-case guarantees matter

Avoid it if:

* Strict **O(1)** extra space is required
* You want faster constants (Counting Sort works better here due to bounded range)

---

## 8. Summary

* **LeetCode 912** tests your understanding of core sorting algorithms
* **Priority Queue** offers:

  * Clean implementation
  * Guaranteed performance
  * Strong interview clarity
* Heap approach is essentially **Heap Sort implemented via a Min-Heap**

---

If you want next:

* **Merge Sort version with recursion tree**
* **Counting Sort exploiting value range**
* **In-place Heap Sort (without extra array)**

tell me which one you want.