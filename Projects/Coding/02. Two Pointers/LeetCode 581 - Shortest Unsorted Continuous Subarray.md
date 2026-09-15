---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

Below is a structured, interview-grade explanation of **LeetCode 581 – Shortest Unsorted Continuous Subarray**, aligned with your requested format.

---

## LeetCode 581 – Shortest Unsorted Continuous Subarray

### Problem Statement

Given an integer array `nums`, return the length of the shortest **continuous subarray** such that if you sort only this subarray in **ascending order**, the **entire array** becomes sorted.

If the array is already sorted, return `0`.

**Example**

```text
Input:  nums = [2, 6, 4, 8, 10, 9, 15]
Output: 5
```

---

## Key Observation

1. If the array were already sorted, every prefix minimum and suffix maximum would respect ordering.
2. The unsorted subarray is bounded by:

   * The **first index from the left** where the ordering breaks.
   * The **first index from the right** where the ordering breaks.
3. Once these boundaries are identified, the subarray must expand to include:

   * Any element on the left **greater than the minimum** of the unsorted region.
   * Any element on the right **less than the maximum** of the unsorted region.

This naturally leads to a **two-pointer / bidirectional scan** strategy.

---

## Two Pointer Technique (Linear Time, Constant Space)

### Step 1: Scan from Left → Right

Track the **maximum seen so far**.

* If `nums[i] < max_so_far`, the array is unsorted at index `i`.
* Update `right` boundary.

### Step 2: Scan from Right → Left

Track the **minimum seen so far**.

* If `nums[i] > min_so_far`, the array is unsorted at index `i`.
* Update `left` boundary.

### Step 3: Compute Result

* If `right` was never updated → array is already sorted → return `0`
* Else → return `right - left + 1`

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n: int = len(nums)
        
        left: int = -1
        right: int = -1
        
        max_seen: int = nums[0]
        for i in range(1, n):
            if nums[i] < max_seen:
                right = i
            else:
                max_seen = nums[i]
        
        min_seen: int = nums[-1]
        for i in range(n - 2, -1, -1):
            if nums[i] > min_seen:
                left = i
            else:
                min_seen = nums[i]
        
        if right == -1:
            return 0
        
        return right - left + 1
```

---

## Worked-Out Example

### Input

```text
nums = [2, 6, 4, 8, 10, 9, 15]
```

---

### Left → Right Scan (Find `right`)

| Index | Value | max\_seen | Condition | right |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 | — | — |
| 1 | 6 | 6 | sorted | — |
| 2 | 4 | 6 | 4 < 6 ❌ | 2 |
| 3 | 8 | 8 | sorted | 2 |
| 4 | 10 | 10 | sorted | 2 |
| 5 | 9 | 10 | 9 < 10 ❌ | 5 |
| 6 | 15 | 15 | sorted | 5 |

**right = 5**

---

### Right → Left Scan (Find `left`)

| Index | Value | min\_seen | Condition | left |
| --- | --- | --- | --- | --- |
| 6 | 15 | 15 | — | — |
| 5 | 9 | 9 | sorted | — |
| 4 | 10 | 9 | 10 > 9 ❌ | 4 |
| 3 | 8 | 8 | sorted | 4 |
| 2 | 4 | 4 | sorted | 4 |
| 1 | 6 | 4 | 6 > 4 ❌ | 1 |
| 0 | 2 | 2 | sorted | 1 |

**left = 1**

---

### Final Answer

```
Length = right - left + 1
        = 5 - 1 + 1
        = 5
```

---

## Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(1)`
* **Technique Used:** Two pointers / Bidirectional scan

---

If you want, I can also:

* Compare this with the **sorting-based O(n log n)** approach
* Show a **monotonic stack** solution
* Explain why this works using **inversion logic**

Just tell me how deep you want to go.