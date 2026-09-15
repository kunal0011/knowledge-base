---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 215: Kth Largest Element in an Array"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 215: Kth Largest Element in an Array

Below is a structured, interview-ready explanation of **LeetCode 215 – Kth Largest Element in an Array**, aligned with standard algorithmic reasoning and production-quality Python.

---

## LeetCode 215: Kth Largest Element in an Array

### Problem Statement

Given an integer array `nums` and an integer `k`, return the **kth largest element** in the array.

**Important clarifications**

* The kth largest element is the element that would appear at index `n - k` if the array were sorted in ascending order.
* You **must not** fully sort the array unless explicitly choosing that approach.
* Duplicate values are counted separately.

**Example**

```text
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

---

## Key Observation

To find the **kth largest**, you do **not** need the entire array sorted.

Key insight:

* Maintain only the **top k largest elements** seen so far.
* Among these k elements, the smallest one is the **kth largest overall**.

This naturally maps to a **Min-Heap of size k**.

---

## Priority Queue (Heap) Technique

### Why a Min-Heap?

* Python’s `heapq` implements a **min-heap**.
* If we keep a heap of size `k`:

  * The smallest element in the heap (`heap[0]`) is the kth largest element seen so far.
  * Any number smaller than `heap[0]` can be ignored.
  * Any number larger than `heap[0]` deserves a place in the top k.

### Algorithm

1. Initialize an empty min-heap.
2. Iterate through each element `x` in `nums`:

   * Push `x` into the heap.
   * If heap size exceeds `k`, pop the smallest element.
3. After processing all elements:

   * The root of the heap (`heap[0]`) is the kth largest element.

---

## Complexity Analysis

* **Time Complexity:** `O(n log k)`

  * Each insertion/removal costs `log k`, done `n` times.
* **Space Complexity:** `O(k)`

  * Heap stores at most `k` elements.

This is optimal for large `n` with small `k`.

---

## Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        min_heap: List[int] = []

        for num in nums:
            heapq.heappush(min_heap, num)

            # Keep heap size at most k
            if len(min_heap) > k:
                heapq.heappop(min_heap)

        # Root of the min-heap is the kth largest element
        return min_heap[0]
```

---

## Worked Example 1

```text
nums = [3, 2, 1, 5, 6, 4]
k = 2
```

### Step-by-step Heap Evolution

| Step | Element | Heap (min-heap) | Action |
| --- | --- | --- | --- |
| 1 | 3 | [3] | push |
| 2 | 2 | [2, 3] | push |
| 3 | 1 | [1, 3, 2] → [2, 3] | push → pop |
| 4 | 5 | [2, 3, 5] → [3, 5] | push → pop |
| 5 | 6 | [3, 5, 6] → [5, 6] | push → pop |
| 6 | 4 | [4, 6, 5] → [5, 6] | push → pop |

Final heap: `[5, 6]`  
**Answer:** `5`

---

## Worked Example 2 (With Duplicates)

```text
nums = [3, 2, 3, 1, 2, 4, 5, 5, 6]
k = 4
```

Sorted view (for understanding only):

```
[1, 2, 2, 3, 3, 4, 5, 5, 6]
```

4th largest = `4`

### Heap Evolution Summary

* Heap always keeps the largest 4 elements seen so far.
* Final heap will be something like: `[4, 5, 5, 6]`
* Root = `4`

**Answer:** `4`

---

## Why This Approach Is Preferred in Interviews

* Avoids full sorting (`O(n log n)`).
* Demonstrates understanding of heap invariants.
* Scales efficiently for large datasets.
* Clean and production-safe.

---

## Related Variations

* Use **Quickselect** for average `O(n)` time.
* Convert to **kth smallest** by index transformation.
* Streaming data version using the same min-heap idea.

If you want, I can also:

* Explain the **Quickselect** approach with partition logic.
* Compare **heap vs quickselect** tradeoffs.
* Show a **max-heap simulation** in Python.