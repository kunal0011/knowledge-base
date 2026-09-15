---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

Below is a structured, interview-ready explanation for **LeetCode 581 – Shortest Unsorted Continuous Subarray**, aligned with the format you have been using.

---

## LeetCode 581 — Shortest Unsorted Continuous Subarray

---

### Problem Statement

Given an integer array `nums`, return the length of the shortest continuous subarray such that **if this subarray is sorted in ascending order**, the **entire array becomes sorted**.

If the array is already sorted, return `0`.

---

### Key Observations

1. A fully sorted array satisfies:

   ```
   nums[i] <= nums[i + 1] for all i
   ```
2. The disorder occurs where this condition is violated.

   * The **left boundary** is where elements start being greater than something to their right.
   * The **right boundary** is where elements are smaller than something to their left.
3. Sorting only the minimal subarray that covers all such violations is sufficient.
4. We need to identify:

   * The **leftmost index** that must be included
   * The **rightmost index** that must be included

---

### Stack Key Insight (Monotonic Stack)

We can detect disorder efficiently using **monotonic stacks**.

#### 1. Left Boundary (Using Increasing Stack)

* Traverse from **left → right**
* Maintain a **monotonic increasing stack**
* If `nums[i] < nums[stack.top]`, then:

  * Pop until the order is restored
  * Track the **minimum index popped** → this is the left boundary

#### 2. Right Boundary (Using Decreasing Stack)

* Traverse from **right → left**
* Maintain a **monotonic decreasing stack**
* If `nums[i] > nums[stack.top]`, then:

  * Pop until order is restored
  * Track the **maximum index popped** → this is the right boundary

---

### Why Stack Works

* A monotonic stack preserves sorted order of indices
* Any violation immediately tells us:

  * Which indices are **out of place**
  * How far the disorder spreads

This avoids sorting and runs in **O(n)** time.

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        stack = []
        left = n
        right = 0

        # Find left boundary
        for i in range(n):
            while stack and nums[i] < nums[stack[-1]]:
                left = min(left, stack.pop())
            stack.append(i)

        stack.clear()

        # Find right boundary
        for i in range(n - 1, -1, -1):
            while stack and nums[i] > nums[stack[-1]]:
                right = max(right, stack.pop())
            stack.append(i)

        return max(0, right - left + 1)
```

---

### Worked Out Example

#### Input

```text
nums = [2, 6, 4, 8, 10, 9, 15]
```

---

### Step 1: Left Boundary Detection

| Index | Value | Stack (indices) | Action | Left |
| --- | --- | --- | --- | --- |
| 0 | 2 | [] | push | 7 |
| 1 | 6 | [0] | push | 7 |
| 2 | 4 | [0,1] | pop 1 | 1 |
|  |  | [0] | push | 1 |
| 3 | 8 | [0,2] | push | 1 |
| 4 | 10 | [0,2,3] | push | 1 |
| 5 | 9 | [0,2,3,4] | pop 4 | 1 |
|  |  | [0,2,3] | push | 1 |
| 6 | 15 | [...] | push | 1 |

✔ Left boundary = **1**

---

### Step 2: Right Boundary Detection

| Index | Value | Stack | Action | Right |
| --- | --- | --- | --- | --- |
| 6 | 15 | [] | push | 0 |
| 5 | 9 | [6] | push | 0 |
| 4 | 10 | [6,5] | pop 5 | 5 |
|  |  | [6] | push | 5 |
| 3 | 8 | [...] | push | 5 |
| 2 | 4 | [...] | push | 5 |
| 1 | 6 | [...] | pop 2 | 5 |
|  |  | [...] | pop 3 | 5 |

✔ Right boundary = **5**

---

### Final Calculation

```
Length = right - left + 1
        = 5 - 1 + 1
        = 5
```

---

### Output

```
5
```

---

### Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(n)** (stack) |

---

### Summary

* The problem reduces to identifying the **smallest window of disorder**
* Monotonic stacks detect boundary violations in linear time
* This approach is optimal and interview-preferred

If you want, I can also:

* Compare this with the **sorting-based O(n log n)** approach
* Provide a **one-pass greedy alternative**
* Draw a **stack evolution diagram** similar to your backtracking trees