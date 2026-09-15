---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff ≤ Limit"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff ≤ Limit

Below is a **complete, interview-grade explanation** of **LeetCode 1438 – Longest Continuous Subarray With Absolute Diff ≤ Limit**, structured exactly as requested.

---

## 1. Problem Statement

You are given an integer array `nums` and an integer `limit`.

A **continuous subarray** is valid if:

```
max(subarray) − min(subarray) ≤ limit
```

Return the **length of the longest** such subarray.

---

### Example

```text
Input:  nums = [8,2,4,7], limit = 4
Output: 2
```

---

## 2. Key Observation

The constraint depends **only on the maximum and minimum values** in the current subarray.

So, for any window `[l … r]`, we need to efficiently know:

* Maximum element
* Minimum element

This naturally leads to:

* **Sliding Window** (two pointers)
* **Data structure to maintain min and max dynamically**

---

## 3. Why Priority Queue (Heap) Works Here

We use **two heaps**:

1. **Max Heap** → to get current maximum
2. **Min Heap** → to get current minimum

Each heap stores:

```
(value, index)
```

Why index?

* Because when the left pointer `l` moves forward, old elements become **invalid**
* We lazily remove elements whose index `< l`

---

## 4. Algorithm (Priority Queue Technique)

### Steps

1. Initialize:

   * `min_heap = []`
   * `max_heap = []`
   * `left = 0`
   * `ans = 0`
2. Expand window by moving `right`
3. Push `(nums[right], right)` into:

   * `min_heap`
   * `max_heap` (store as `(-value, index)`)
4. While window is **invalid**:

   ```
   -max_heap[0][0] - min_heap[0][0] > limit
   ```

   * Move `left` forward
   * Remove outdated elements from both heaps
5. Update answer:

   ```
   ans = max(ans, right - left + 1)
   ```

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List
import heapq

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_heap = []  # (value, index)
        max_heap = []  # (-value, index)

        left = 0
        result = 0

        for right, value in enumerate(nums):
            heapq.heappush(min_heap, (value, right))
            heapq.heappush(max_heap, (-value, right))

            # Shrink window until condition satisfied
            while -max_heap[0][0] - min_heap[0][0] > limit:
                left += 1

                # Remove outdated elements
                while min_heap[0][1] < left:
                    heapq.heappop(min_heap)

                while max_heap[0][1] < left:
                    heapq.heappop(max_heap)

            result = max(result, right - left + 1)

        return result
```

---

## 6. Worked Example (Step-by-Step)

### Input

```text
nums = [8, 2, 4, 7]
limit = 4
```

---

### Step 1: right = 0 → [8]

```
min = 8, max = 8
diff = 0 ≤ 4
window = [8]
length = 1
```

---

### Step 2: right = 1 → [8, 2]

```
min = 2, max = 8
diff = 6 > 4 ❌
```

Shrink window:

```
left moves from 0 → 1
window = [2]
```

Valid again:

```
length = 1
```

---

### Step 3: right = 2 → [2, 4]

```
min = 2, max = 4
diff = 2 ≤ 4
length = 2 ✔
```

---

### Step 4: right = 3 → [2, 4, 7]

```
min = 2, max = 7
diff = 5 > 4 ❌
```

Shrink window:

```
left = 2
window = [4, 7]
min = 4, max = 7
diff = 3 ≤ 4
length = 2
```

---

### Final Answer

```
2
```

---

## 7. Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n log n)** (heap operations per element) |
| Space | **O(n)** (heaps store indices) |

---

## 8. Interview Insight

* This problem is a **classic sliding window + range constraint**
* Priority Queue is intuitive but **not optimal**
* Optimal solution uses **Monotonic Deques (O(n))**
* PQ approach is still **fully acceptable** and easier to reason about

If you want, I can:

* Show **deque-based O(n) solution**
* Compare **PQ vs Deque**
* Convert this into a **template for similar problems**
* Explain **why lazy deletion is mandatory**

Just tell me how deep you want to go.