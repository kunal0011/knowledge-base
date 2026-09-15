---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

**LeetCode 581 – Shortest Unsorted Continuous Subarray**, structured exactly as requested.

---

## 1. Problem Statement

Given an integer array `nums`, return the length of the **shortest continuous subarray** such that if this subarray is sorted in ascending order, the **entire array becomes sorted**.

If the array is already sorted, return `0`.

### Constraints

* `1 ≤ nums.length ≤ 10^4`
* `-10^5 ≤ nums[i] ≤ 10^5`

---

## 2. Key Observation

If the array were fully sorted:

* Every element would be **≥ max of all elements to its left**
* Every element would be **≤ min of all elements to its right**

Hence:

* The disorder occurs when an element violates this monotonic property.
* We do **not** need to sort the array or find the exact subarray explicitly.

Instead, we identify:

* **Right boundary**: where an element is **smaller than the maximum seen so far**
* **Left boundary**: where an element is **larger than the minimum seen so far from the right**

This enables a **linear-time greedy solution**.

---

## 3. Greedy Solution – Core Trick

### Two-pass greedy scan

#### Pass 1: Left → Right

* Maintain `max_so_far`
* If `nums[i] < max_so_far`, the array is unsorted at `i`
* Update `right = i`

#### Pass 2: Right → Left

* Maintain `min_so_far`
* If `nums[i] > min_so_far`, the array is unsorted at `i`
* Update `left = i`

### Final Answer

```
length = right - left + 1
```

If `right` was never updated → array already sorted → return `0`.

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        
        max_so_far = float('-inf')
        right = -1
        
        # Left to right scan
        for i in range(n):
            if nums[i] < max_so_far:
                right = i
            else:
                max_so_far = nums[i]
        
        min_so_far = float('inf')
        left = -1
        
        # Right to left scan
        for i in range(n - 1, -1, -1):
            if nums[i] > min_so_far:
                left = i
            else:
                min_so_far = nums[i]
        
        # Already sorted
        if right == -1:
            return 0
        
        return right - left + 1
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```text
nums = [2, 6, 4, 8, 10, 9, 15]
```

---

### Step 1: Left → Right Scan (Finding `right`)

| i | nums[i] | max\_so\_far | Condition | right |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 | OK | -1 |
| 1 | 6 | 6 | OK | -1 |
| 2 | 4 | 6 | 4 < 6 → disorder | 2 |
| 3 | 8 | 8 | OK | 2 |
| 4 | 10 | 10 | OK | 2 |
| 5 | 9 | 10 | 9 < 10 → disorder | 5 |
| 6 | 15 | 15 | OK | 5 |

✅ **Right boundary = 5**

---

### Step 2: Right → Left Scan (Finding `left`)

| i | nums[i] | min\_so\_far | Condition | left |
| --- | --- | --- | --- | --- |
| 6 | 15 | 15 | OK | -1 |
| 5 | 9 | 9 | OK | -1 |
| 4 | 10 | 9 | 10 > 9 → disorder | 4 |
| 3 | 8 | 8 | OK | 4 |
| 2 | 4 | 4 | OK | 4 |
| 1 | 6 | 4 | 6 > 4 → disorder | 1 |
| 0 | 2 | 2 | OK | 1 |

✅ **Left boundary = 1**

---

### Step 3: Final Result

```
Subarray indices: [1 .. 5]
Subarray: [6, 4, 8, 10, 9]
Length = 5 - 1 + 1 = 5
```

### Output

```
5
```

---

## 6. Why This Greedy Approach Works

* Any element **smaller than a previous max** must be repositioned.
* Any element **larger than a future min** must be repositioned.
* These two scans precisely capture the minimal unsorted window.
* No sorting required.
* Optimal time and space.

---

## 7. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** |

---

## 8. Interview Takeaway

> This problem is a classic example of using **prefix max** and **suffix min** to detect disorder without sorting.

If you want, I can also provide:

* Stack-based approach (alternative)
* Visual intuition diagram
* Comparison with sorting-based solution
* Similar pattern problems using prefix/suffix scans

Just tell me how deep you want to go.