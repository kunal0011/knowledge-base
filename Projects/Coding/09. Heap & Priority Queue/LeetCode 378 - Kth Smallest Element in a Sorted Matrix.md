---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 378: Kth Smallest Element in a Sorted Matrix"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 378: Kth Smallest Element in a Sorted Matrix

Below is a complete, interview-grade explanation of **LeetCode 378 – Kth Smallest Element in a Sorted Matrix**, structured exactly as requested.

---

## 1. Problem Statement

You are given an `n x n` matrix where:

* Each **row** is sorted in ascending order.
* Each **column** is sorted in ascending order.

Return the **k-th smallest element** in the matrix.

### Constraints

* `1 <= n <= 300`
* `1 <= k <= n^2`
* Matrix elements can be negative.
* You must return the element in the **sorted order of all n² elements**, not the k-th distinct element.

---

## 2. Key Observations

1. **Matrix is globally sorted in two dimensions**

   * Smallest element is at `(0,0)`
   * Largest element is at `(n-1, n-1)`
2. **Each row behaves like a sorted list**

   * We effectively have `n` sorted lists.
3. **This resembles a “merge k sorted lists” problem**

   * We want the k-th smallest among all elements across sorted rows.
4. **Brute force approaches are inefficient**

   * Flatten + sort → `O(n² log n²)` → too slow for large `n`
5. **Priority Queue (Min-Heap) fits naturally**

   * Always extract the next smallest available candidate.

---

## 3. Priority Queue (Min-Heap) Technique

### Core Idea

Treat each row as a sorted list.  
Push the **first element of each row** into a min-heap.

Each heap entry tracks:

```
(value, row_index, column_index)
```

### Algorithm

1. Initialize a min-heap.
2. Push the first element of each row:

   ```
   (matrix[row][0], row, 0)
   ```
3. Repeat `k` times:

   * Pop the smallest element from the heap.
   * If the popped element is from column `c`,  
     push the **next element in the same row** `(c + 1)` if it exists.
4. The k-th pop gives the answer.

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List
import heapq

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        min_heap: List[tuple[int, int, int]] = []

        # Step 1: push first element of each row
        for row in range(n):
            heapq.heappush(min_heap, (matrix[row][0], row, 0))

        # Step 2: extract min k times
        for _ in range(k):
            value, row, col = heapq.heappop(min_heap)

            # Push next element in the same row
            if col + 1 < n:
                heapq.heappush(
                    min_heap,
                    (matrix[row][col + 1], row, col + 1)
                )

        return value
```

---

## 5. Worked Example (Step-by-Step)

### Input

```
matrix =
[
  [1,  5,  9],
  [10, 11, 13],
  [12, 13, 15]
]
k = 8
```

### Initial Heap (first column)

```
(1, row=0, col=0)
(10, row=1, col=0)
(12, row=2, col=0)
```

### Extraction Process

| Pop # | Popped Value | Push Next | Heap After Push |
| --- | --- | --- | --- |
| 1 | 1 | 5 | 5, 10, 12 |
| 2 | 5 | 9 | 9, 10, 12 |
| 3 | 9 | — | 10, 12 |
| 4 | 10 | 11 | 11, 12 |
| 5 | 11 | 13 | 12, 13 |
| 6 | 12 | 13 | 13, 13 |
| 7 | 13 | — | 13 |
| 8 | **13** | — | — |

### Answer

```
13
```

---

## 6. Complexity Analysis

### Time Complexity

* Heap initialization: `O(n)`
* Each pop/push: `O(log n)`
* Performed `k` times:

```
O(k log n)
```

### Space Complexity

* Heap stores at most `n` elements:

```
O(n)
```

---

## 7. Why This Approach Is Strong in Interviews

* Maps directly to **merge k sorted lists**
* Uses **heap state tracking (row, col)** cleanly
* Avoids scanning entire matrix
* Scales efficiently for large `n`

---

If you want, I can also explain:

* **Binary Search on value space** solution (most optimal)
* **Comparison between Heap vs Binary Search**
* **Visualization of heap evolution**
* **Follow-up: kth largest variation**

Just tell me.