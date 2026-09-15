---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 373: Find K Pairs with Smallest Sums"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 373: Find K Pairs with Smallest Sums

Below is a **complete, structured explanation of LeetCode 373 – Find K Pairs with Smallest Sums**, aligned with how priority queues are used in competitive programming and interviews.

---

## 1. Problem Statement (LeetCode 373)

You are given **two sorted integer arrays** `nums1` and `nums2`, both sorted in **non-decreasing order**, and an integer `k`.

Define a pair `(u, v)` where:

* `u` is from `nums1`
* `v` is from `nums2`

The **sum of the pair** is `u + v`.

### Task

Return the **k pairs** `(u, v)` with the **smallest sums**.

### Constraints (important for design)

* `1 ≤ len(nums1), len(nums2) ≤ 10^5`
* Arrays are **already sorted**
* `k` can be large, but `k ≤ len(nums1) * len(nums2)`

---

## 2. Why Brute Force Will Not Work

A brute-force approach would:

* Generate all possible pairs → `O(n * m)`
* Sort them → `O((n*m) log (n*m))`

This is **completely infeasible** when arrays are large.

We need to exploit:

* **Sorted property**
* **Top-k selection**
* **Incremental generation of smallest sums**

---

## 3. Key Observation (Most Important Insight)

Since both arrays are sorted:

* The **smallest possible sum** is:

  ```
  nums1[0] + nums2[0]
  ```
* For a fixed `nums1[i]`, the sums:

  ```
  nums1[i] + nums2[0],
  nums1[i] + nums2[1],
  nums1[i] + nums2[2], ...
  ```

  are **in increasing order**.

So for each `nums1[i]`, pairing it with `nums2` behaves like a **sorted list of sums**.

### This becomes:

> “Merge k sorted lists and extract k smallest elements”

This is a **classic Min-Heap / Priority Queue** use case.

---

## 4. Priority Queue (Min-Heap) Technique

### Heap State Design

Each heap entry represents:

```
(sum, i, j)
```

Where:

* `sum = nums1[i] + nums2[j]`
* `(i, j)` points to a pair

### Initialization Strategy

Only initialize:

```
(nums1[i] + nums2[0], i, 0)
```

for:

```
i = 0 to min(k-1, len(nums1)-1)
```

Why?

* Because the smallest pair for `nums1[i]` always starts with `nums2[0]`
* We never need more than `k` rows initially

---

## 5. Algorithm (Step-by-Step)

1. If either array is empty or `k == 0`, return `[]`
2. Initialize a **min-heap**
3. Push `(nums1[i] + nums2[0], i, 0)` for `i in [0 … min(k-1, len(nums1)-1)]`
4. Repeat `k` times:

   * Pop the smallest sum `(sum, i, j)`
   * Add pair `[nums1[i], nums2[j]]` to result
   * If `j + 1 < len(nums2)`, push:

     ```
     (nums1[i] + nums2[j+1], i, j+1)
     ```

---

## 6. Python 3 Solution (With Typing)

```python
from typing import List
import heapq

class Solution:
    def kSmallestPairs(
        self,
        nums1: List[int],
        nums2: List[int],
        k: int
    ) -> List[List[int]]:

        if not nums1 or not nums2 or k == 0:
            return []

        min_heap = []
        result: List[List[int]] = []

        # Initialize heap with first column (j = 0)
        for i in range(min(k, len(nums1))):
            heapq.heappush(
                min_heap,
                (nums1[i] + nums2[0], i, 0)
            )

        # Extract k smallest pairs
        while min_heap and len(result) < k:
            _, i, j = heapq.heappop(min_heap)
            result.append([nums1[i], nums2[j]])

            # Move to next column in nums2
            if j + 1 < len(nums2):
                heapq.heappush(
                    min_heap,
                    (nums1[i] + nums2[j + 1], i, j + 1)
                )

        return result
```

---

## 7. Fully Worked Example

### Input

```
nums1 = [1, 7, 11]
nums2 = [2, 4, 6]
k = 3
```

---

### Step 1: Initial Heap

Push `(nums1[i] + nums2[0])`

| Pair | Sum | Heap Entry |
| --- | --- | --- |
| (1,2) | 3 | (3, 0, 0) |
| (7,2) | 9 | (9, 1, 0) |
| (11,2) | 13 | (13, 2, 0) |

Heap:

```
[(3,0,0), (9,1,0), (13,2,0)]
```

---

### Step 2: Pop #1

Pop `(3,0,0)` → pair `(1,2)`

Push next from same row → `(1,4)`

```
(1 + 4, 0, 1) = (5,0,1)
```

Heap:

```
[(5,0,1), (13,2,0), (9,1,0)]
```

Result:

```
[[1,2]]
```

---

### Step 3: Pop #2

Pop `(5,0,1)` → pair `(1,4)`

Push `(1,6)`

```
(7,0,2)
```

Heap:

```
[(7,0,2), (13,2,0), (9,1,0)]
```

Result:

```
[[1,2], [1,4]]
```

---

### Step 4: Pop #3

Pop `(7,0,2)` → pair `(1,6)`

No further push (nums2 exhausted for row 0)

Result:

```
[[1,2], [1,4], [1,6]]
```

Stop because `k = 3`.

---

## 8. Time & Space Complexity

### Time Complexity

```
O(k log k)
```

* Each heap operation is `log k`
* At most `k` pops and pushes

### Space Complexity

```
O(k)
```

* Heap holds at most `k` elements

---

## 9. Interview Takeaways

* This problem is **not** about pair generation
* It is about:

  * **Sorted arrays**
  * **K-way merge**
  * **Min-heap frontier expansion**
* Very similar pattern appears in:

  * K smallest elements in sorted matrix
  * Merge k sorted lists
  * Top-k sum / product problems

If you want, I can also:

* Draw the **implicit matrix + BFS frontier**
* Compare this with a **binary search on answer** approach
* Show why a **visited set is NOT required** here