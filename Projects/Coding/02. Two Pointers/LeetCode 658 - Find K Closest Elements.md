---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 658: Find K Closest Elements"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 658: Find K Closest Elements

Below is a structured, interview-ready explanation for **LeetCode 658 – Find K Closest Elements**, aligned with your requested format.

---

## Problem Statement

Given a **sorted integer array** `arr`, two integers `k` and `x`, return the **k closest integers to `x`** in the array.

* An integer `a` is closer to `x` than `b` if:

  1. `|a − x| < |b − x|`, or
  2. `|a − x| == |b − x|` and `a < b`
* The result must be returned **in ascending order**.

This problem is from LeetCode.

---

## Key Observation

1. The array is **already sorted**.
2. We do **not** need to individually compare all distances.
3. Instead of selecting `k` elements, it is easier to **remove `n − k` elements** that are *farthest* from `x`.
4. Because the array is sorted, the farthest elements must lie at the **extremes** (leftmost or rightmost).

This leads directly to a **two-pointer shrinking window** strategy.

---

## Two Pointer Technique (Core Idea)

1. Initialize two pointers:

   * `left = 0`
   * `right = len(arr) - 1`
2. While the window size is larger than `k`:

   * Compare distances:

     * `|arr[left] - x|`
     * `|arr[right] - x|`
   * Remove the element with the **larger distance**:

     * If left is farther → `left += 1`
     * Else → `right -= 1`
3. When the window size becomes exactly `k`, the remaining subarray is the answer.

This works because:

* The closest `k` elements must form a **contiguous subarray** in a sorted array.
* We always discard the worst candidate.

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - 1

        # Shrink the window until its size is k
        while right - left + 1 > k:
            if abs(arr[left] - x) > abs(arr[right] - x):
                left += 1
            else:
                right -= 1

        return arr[left:right + 1]
```

---

## Worked Out Example

### Input

```
arr = [1, 2, 3, 4, 5]
k = 4
x = 3
```

### Step-by-Step Execution

| left | right | Window | Compare Distances | Action |
| --- | --- | --- | --- | --- |
| 0 | 4 | [1,2,3,4,5] |  | 1−3 |
| 0 | 3 | [1,2,3,4] | window size == k | stop |

### Output

```
[1, 2, 3, 4]
```

---

## Time & Space Complexity

* **Time Complexity:** `O(n − k)`  
  (we remove exactly `n − k` elements)
* **Space Complexity:** `O(1)`  
  (ignoring output storage)

---

## Key Takeaway

This problem is not about “finding closest elements” directly—it is about **eliminating the farthest elements efficiently using sorted order**. Recognizing this transformation is the main interview insight.

If you want, I can also explain:

* Binary search + sliding window variant
* Why a heap solution is suboptimal here
* Edge cases interviewers often probe