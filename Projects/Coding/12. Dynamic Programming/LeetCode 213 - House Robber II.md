---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 213: House Robber II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 213: House Robber II

**LeetCode 213 – House Robber II**, with **proper state definition, transitions, DP table construction, and a worked example**.

---

## Problem Statement (LeetCode 213 – House Robber II)

You are given an integer array `nums` where `nums[i]` represents the amount of money in the `i`-th house.  
Houses are arranged **in a circle**, meaning:

* House `0` is adjacent to house `n-1`
* You **cannot rob two adjacent houses**

Return the **maximum amount of money** you can rob **without triggering the alarm**.

---

## Key Observation (Why this differs from LeetCode 198)

In **House Robber I**, houses are in a straight line.  
In **House Robber II**, the circular arrangement introduces **one extra constraint**:

> **You cannot rob both the first and the last house.**

So the problem splits naturally into **two mutually exclusive cases**:

1. **Rob houses from index `0` to `n-2`** (exclude last house)
2. **Rob houses from index `1` to `n-1`** (exclude first house)

The final answer is:

```
max(case1, case2)
```

Each case is a **standard House Robber (LeetCode 198)** problem.

---

## DP State Definition (for linear robber subproblem)

For a given subarray `nums[l...r]`:

### State

```
dp[i] = maximum money that can be robbed from houses l to i
```

### Transition

At house `i`, you have two choices:

1. **Do not rob house i**

   ```
   dp[i] = dp[i-1]
   ```
2. **Rob house i**

   ```
   dp[i] = dp[i-2] + nums[i]
   ```

### Final Transition Formula

```
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

---

## Base Cases

For the subarray starting at index `l`:

* `dp[l] = nums[l]`
* `dp[l+1] = max(nums[l], nums[l+1])`

---

## Complete Algorithm

1. If `n == 1`, return `nums[0]`
2. Compute:

   * `rob_linear(0, n-2)`
   * `rob_linear(1, n-1)`
3. Return the maximum of the two results

---

## Worked Example

### Input

```text
nums = [2, 3, 2]
```

### Case 1: Rob from index `0` to `1`

```
Subarray = [2, 3]
```

| i | nums[i] | dp[i] | Explanation |
| --- | --- | --- | --- |
| 0 | 2 | 2 | Only one house |
| 1 | 3 | 3 | max(2, 3) |

Result = **3**

---

### Case 2: Rob from index `1` to `2`

```
Subarray = [3, 2]
```

| i | nums[i] | dp[i] | Explanation |
| --- | --- | --- | --- |
| 1 | 3 | 3 | Only one house |
| 2 | 2 | 3 | max(3, 2) |

Result = **3**

---

### Final Answer

```
max(3, 3) = 3
```

---

## DP Table Example (Larger Input)

### Input

```text
nums = [1, 2, 3, 1]
```

---

### Case 1: Rob from `0` to `2` → `[1, 2, 3]`

| i | nums[i] | dp[i] |
| --- | --- | --- |
| 0 | 1 | 1 |
| 1 | 2 | 2 |
| 2 | 3 | max(2, 1+3) = 4 |

Result = **4**

---

### Case 2: Rob from `1` to `3` → `[2, 3, 1]`

| i | nums[i] | dp[i] |
| --- | --- | --- |
| 1 | 2 | 2 |
| 2 | 3 | 3 |
| 3 | 1 | max(3, 2+1) = 3 |

Result = **3**

---

### Final Answer

```
max(4, 3) = 4
```

---

## Python 3 DP Implementation (With Typing)

```python
from typing import List

class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]

        def rob_linear(start: int, end: int) -> int:
            if start == end:
                return nums[start]

            dp = [0] * n
            dp[start] = nums[start]
            dp[start + 1] = max(nums[start], nums[start + 1])

            for i in range(start + 2, end + 1):
                dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

            return dp[end]

        return max(
            rob_linear(0, n - 2),
            rob_linear(1, n - 1)
        )
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** (can be optimized to **O(1)**) |

---

## Why This DP Works

* Circular constraint is handled by **problem decomposition**
* Each subproblem is **independent and linear**
* DP captures the **optimal substructure**
* No overlapping circular conflict remains

---

If you want next:

* **Space-optimized O(1) DP**
* **State machine interpretation**
* **Comparison with greedy (why greedy fails)**

Tell me which one you want.