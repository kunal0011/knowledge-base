---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 413: Arithmetic Slices"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 413: Arithmetic Slices

**LeetCode 413 – Arithmetic Slices**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## Problem Statement (Condensed)

Given an integer array `nums`, return the number of **arithmetic subarrays** of length **≥ 3**.

An array is **arithmetic** if the difference between consecutive elements is constant.

---

## Key Observation

An arithmetic slice must:

* Have **at least 3 elements**
* Maintain the **same difference** across adjacent elements

If we already know that an arithmetic slice **ends at index `i-1`**, then we only need to check whether `nums[i] - nums[i-1] == nums[i-1] - nums[i-2]` to extend it.

---

## DP State Definition

Let:

```
dp[i] = number of arithmetic subarrays that END at index i
```

Important:

* These subarrays **must include nums[i]**
* Length is **≥ 3**
* We do **not** store all subarrays — only the count ending at `i`

---

## DP Transition

For `i >= 2`:

```
if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
    dp[i] = dp[i-1] + 1
else:
    dp[i] = 0
```

### Why `dp[i-1] + 1`?

* `+1` → the new slice formed by `(nums[i-2], nums[i-1], nums[i])`
* `dp[i-1]` → all previous slices ending at `i-1` can be extended by `nums[i]`

---

## Base Case

```
dp[0] = 0
dp[1] = 0
```

Reason:  
A slice needs **at least 3 elements**, so nothing valid can end at indices `0` or `1`.

---

## Final Answer

```
sum(dp)
```

Because:

* Each `dp[i]` counts **distinct arithmetic subarrays ending at i**
* Summing gives the total count

---

## Example Walkthrough

### Input

```text
nums = [1, 2, 3, 4]
```

### Differences

```
2 - 1 = 1
3 - 2 = 1
4 - 3 = 1
```

### DP Table Construction

| i | nums[i] | Check | dp[i] | Explanation |
| --- | --- | --- | --- | --- |
| 0 | 1 | – | 0 | Too short |
| 1 | 2 | – | 0 | Too short |
| 2 | 3 | 1 == 1 | 1 | [1,2,3] |
| 3 | 4 | 1 == 1 | 2 | [2,3,4], [1,2,3,4] |

### DP Array

```
dp = [0, 0, 1, 2]
```

### Result

```
1 + 2 = 3
```

### Arithmetic Slices

```
[1,2,3]
[2,3,4]
[1,2,3,4]
```

---

## Python 3 (With Typing)

```python
from typing import List

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0

        dp = [0] * n
        total = 0

        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                dp[i] = dp[i - 1] + 1
                total += dp[i]

        return total
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | `O(n)` |
| Space | `O(n)` (can be optimized to `O(1)`) |

---

## Space-Optimized Insight (Conceptual)

Since `dp[i]` only depends on `dp[i-1]`, you can replace the array with a single variable:

```
curr = dp[i]
```

But the **state definition and transition remain identical**, which is critical for interviews.

---

## Summary

* **State**: `dp[i]` = arithmetic slices ending at `i`
* **Transition**: extend previous slices if difference matches
* **Answer**: sum of all `dp[i]`
* **Pattern**: DP on contiguous subarrays (extension-based)

If you want, I can next:

* Show the **O(1) space version**
* Compare with **sliding window (why it doesn’t fit cleanly)**
* Generalize to **non-contiguous arithmetic subsequences** (LeetCode 446)