---
date: "2025-12-23"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 1574: Shortest Subarray to be Removed to Make Array Sorted"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 1574: Shortest Subarray to be Removed to Make Array Sorted

Below is a structured, interview-grade explanation of **LeetCode 1574 – Shortest Subarray to be Removed to Make Array Sorted**, aligned with your requested format.

---

## Problem Statement

Given an integer array `arr`, remove **one contiguous subarray (possibly empty)** such that the remaining elements are **non-decreasing**.  
Return the **minimum length** of the subarray that must be removed.

**Non-decreasing** means:  
`arr[i] <= arr[i + 1]`

---

## Key Observation

1. If the array is already non-decreasing, the answer is `0`.
2. Any valid solution keeps:

   * a **non-decreasing prefix**, and
   * a **non-decreasing suffix**
3. The subarray removed lies **between** some prefix and suffix.
4. Therefore, the problem reduces to:

   > Find the **maximum length** of a prefix + suffix that can be merged while maintaining sorted order.

---

## Two Pointer Technique (Core Insight)

### Step 1: Identify the longest sorted prefix

Traverse from the left until the order breaks.

```
arr[0] <= arr[1] <= ... <= arr[left]
```

### Step 2: Identify the longest sorted suffix

Traverse from the right until the order breaks.

```
arr[right] <= arr[right+1] <= ... <= arr[n-1]
```

### Step 3: Merge prefix and suffix

Use **two pointers**:

* `i` on the prefix
* `j` on the suffix

Try to connect `arr[i]` with `arr[j]` such that:

```
arr[i] <= arr[j]
```

If valid, we can remove everything **between `i` and `j`**.

---

## Algorithm (High Level)

1. Find `left` = end of sorted prefix
2. Find `right` = start of sorted suffix
3. Initialize answer as:

   ```
   min(n - left - 1, right)
   ```
4. Use two pointers `i` and `j` to minimize `(j - i - 1)`

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        n = len(arr)

        # Step 1: Find longest non-decreasing prefix
        left = 0
        while left + 1 < n and arr[left] <= arr[left + 1]:
            left += 1

        # Already sorted
        if left == n - 1:
            return 0

        # Step 2: Find longest non-decreasing suffix
        right = n - 1
        while right > 0 and arr[right - 1] <= arr[right]:
            right -= 1

        # Step 3: Initial answer
        ans = min(n - left - 1, right)

        # Step 4: Two-pointer merge
        i, j = 0, right
        while i <= left and j < n:
            if arr[i] <= arr[j]:
                ans = min(ans, j - i - 1)
                i += 1
            else:
                j += 1

        return ans
```

---

## Worked Example

### Input

```
arr = [1, 2, 3, 10, 4, 2, 3, 5]
```

### Step 1: Sorted Prefix

```
[1, 2, 3, 10] ❌ break at 10 > 4
left = 3
```

### Step 2: Sorted Suffix

```
[2, 3, 5] is sorted
right = 5
```

### Step 3: Initial Answer

```
min(n - left - 1, right)
= min(8 - 3 - 1, 5)
= min(4, 5)
= 4
```

### Step 4: Two-Pointer Merge

| i | j | arr[i] | arr[j] | Action | Removed Length |
| --- | --- | --- | --- | --- | --- |
| 0 | 5 | 1 | 2 | valid | 4 |
| 1 | 5 | 2 | 2 | valid | 3 |
| 2 | 5 | 3 | 2 | ❌ | move j |
| 2 | 6 | 3 | 3 | valid | **3** |
| 3 | 7 | 10 | 5 | ❌ | stop |

### Final Answer

```
3
```

**Remove subarray:** `[10, 4, 2]`

---

## Time & Space Complexity

* **Time:** `O(n)`
* **Space:** `O(1)` (in-place pointers)

---

## Takeaway Pattern

This problem is a classic example of:

* **Prefix–Suffix decomposition**
* **Two-pointer merge on sorted segments**

This pattern appears frequently in array optimization problems where **one contiguous block may be removed**.

If you want, I can also provide:

* Binary-search based alternative
* Visualization with pointer movement
* Comparison with similar problems (e.g., LeetCode 581, 978)

Just let me know.