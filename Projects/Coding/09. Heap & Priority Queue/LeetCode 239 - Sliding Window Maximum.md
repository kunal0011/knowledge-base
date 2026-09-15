---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 239: Sliding Window Maximum"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 239: Sliding Window Maximum

Below is a complete, structured explanation of **LeetCode 239 – Sliding Window Maximum**, aligned with your usual learning format.

---

## 1. Problem Statement

You are given an integer array `nums` and an integer `k`.

There is a sliding window of size `k` that moves from left to right across the array.  
At each position of the window, return the **maximum element** inside that window.

### Input

* `nums`: List[int]
* `k`: int (window size)

### Output

* List[int]: maximum value for each window position

### Example

```text
nums = [1,3,-1,-3,5,3,6,7], k = 3
Output = [3,3,5,5,6,7]
```

---

## 2. Key Observation

A brute-force solution checks each window in `O(k)` time, leading to **O(n·k)** overall — too slow.

### What we need:

* Quickly **remove elements that exit the window**
* Quickly **know the maximum in the current window**

This leads to data structures that support:

* Efficient insertion
* Efficient maximum retrieval

---

## 3. Priority Queue (Heap) Technique

Python’s `heapq` is a **min-heap**, so to simulate a **max-heap**:

* Store values as `(-value, index)`
* The index helps us detect **stale elements** (elements no longer in the window)

### Core Idea

1. Push every element into the heap with its index.
2. The heap top always gives the maximum (via negative value).
3. Before using the top:

   * Remove elements whose index is **outside the current window**.

---

## 4. Algorithm (Priority Queue Approach)

### Steps

1. Initialize an empty heap.
2. Iterate through `nums` with index `i`:

   * Push `(-nums[i], i)` into heap.
3. Once `i >= k - 1`:

   * Pop elements while their index `< i - k + 1` (out of window).
   * The top of heap is the window maximum.
   * Append it to result.

---

## 5. Python 3 Solution (with typing)

```python
from typing import List
import heapq

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        max_heap: List[tuple[int, int]] = []
        result: List[int] = []

        for i, num in enumerate(nums):
            # Push current element with negative value for max-heap behavior
            heapq.heappush(max_heap, (-num, i))

            # Start recording results when the first window is complete
            if i >= k - 1:
                # Remove elements that are out of the current window
                while max_heap[0][1] < i - k + 1:
                    heapq.heappop(max_heap)

                # Current maximum is at the top
                result.append(-max_heap[0][0])

        return result
```

---

## 6. Worked Example (Step-by-Step)

### Input

```text
nums = [1,3,-1,-3,5,3,6,7]
k = 3
```

---

### Window 1: indices [0..2] → [1, 3, -1]

Heap contains:

```
(-3,1), (-1,0), (1,2)
```

* Valid window start = `0`
* Max = `3`

Result → `[3]`

---

### Window 2: indices [1..3] → [3, -1, -3]

Heap top = `(-3,1)`  
Index `1 >= 1` → valid

Result → `[3, 3]`

---

### Window 3: indices [2..4] → [-1, -3, 5]

Heap top = `(-5,4)`  
Max = `5`

Result → `[3, 3, 5]`

---

### Window 4: indices [3..5] → [-3, 5, 3]

Heap top = `(-5,4)`  
Max = `5`

Result → `[3, 3, 5, 5]`

---

### Window 5: indices [4..6] → [5, 3, 6]

Heap top = `(-6,6)`  
Max = `6`

Result → `[3, 3, 5, 5, 6]`

---

### Window 6: indices [5..7] → [3, 6, 7]

Heap top = `(-7,7)`  
Max = `7`

Final Result:

```
[3, 3, 5, 5, 6, 7]
```

---

## 7. Complexity Analysis

### Time Complexity

* Each element is pushed once and popped at most once
* **O(n log n)** in worst case

### Space Complexity

* Heap stores up to `n` elements
* **O(n)**

---

## 8. Important Notes

* This solution is **correct and intuitive**, but **not optimal**
* The **optimal solution** uses a **monotonic deque** with **O(n)** time
* Priority Queue is often preferred when:

  * You want simpler logic
  * You already understand heap mechanics

If you want, I can next:

* Convert this to the **deque-based O(n) solution**
* Compare **heap vs deque trade-offs**
* Draw the **heap evolution visually per window**