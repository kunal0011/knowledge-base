---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 494: Target Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 494: Target Sum

**LeetCode 494 – Target Sum**, focusing on **state definition, transition, DP table construction**, and a **worked example**.

---

## Problem Statement (LeetCode 494)

You are given an integer array `nums` and an integer `target`.

You can assign either `'+'` or `'-'` in front of each number.  
Return the **number of different expressions** that evaluate to `target`.

---

## Key Observation (Problem Transformation)

Let:

* `P` = sum of numbers assigned `'+'`
* `N` = sum of numbers assigned `'-'`

Then:

```
P - N = target
P + N = totalSum
```

Add both equations:

```
2P = target + totalSum
P = (target + totalSum) / 2
```

### Critical Conditions

1. `target + totalSum` must be **even**
2. `|target| <= totalSum`

If not satisfied → **answer = 0**

---

## Reformulated Problem

> Count the number of **subsets** of `nums` whose sum equals  
> `requiredSum = (target + totalSum) / 2`

This is now a **Subset Sum Count** problem.

---

## DP State Definition

Let:

```
dp[i][s] = number of ways to pick a subset
           from first i elements
           such that subset sum = s
```

* `i` → number of elements considered (0 to n)
* `s` → target subset sum (0 to requiredSum)

---

## DP Transition

For element `nums[i-1]`:

### Case 1: Do not include nums[i-1]

```
dp[i][s] += dp[i-1][s]
```

### Case 2: Include nums[i-1] (only if s >= nums[i-1])

```
dp[i][s] += dp[i-1][s - nums[i-1]]
```

### Final Transition Formula

```
dp[i][s] = dp[i-1][s]
         + dp[i-1][s - nums[i-1]]  (if s >= nums[i-1])
```

---

## Base Case (Very Important)

```
dp[0][0] = 1
```

Explanation:

* There is **1 way** to make sum `0` using `0` elements (empty subset)

All other `dp[0][s] = 0`

---

## DP Table Construction (Example)

### Input

```
nums   = [1, 1, 1, 1, 1]
target = 3
```

### Step 1: Compute requiredSum

```
totalSum = 5
requiredSum = (3 + 5) / 2 = 4
```

---

### DP Table Layout

`dp[i][s]` where:

* `i = 0..5`
* `s = 0..4`

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0 | 0 |

---

### After processing each `1`

#### i = 1 (nums[0] = 1)

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | 0 | 0 |

---

#### i = 2

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | 1 | 0 | 0 |

---

#### i = 3

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 3 | 1 | 3 | 3 | 1 | 0 |

---

#### i = 4

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 4 | 6 | 4 | 1 |

---

#### i = 5

| i \ s | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 5 | 1 | 5 | 10 | 10 | **5** |

---

### Final Answer

```
dp[5][4] = 5
```

There are **5 valid expressions**.

---

## Python 3 DP Implementation (Typed)

### Space-Optimized (1D DP)

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        
        # Edge case
        if abs(target) > total_sum or (target + total_sum) % 2 != 0:
            return 0
        
        required_sum = (target + total_sum) // 2
        
        dp = [0] * (required_sum + 1)
        dp[0] = 1  # base case
        
        for num in nums:
            # traverse backwards to avoid reuse
            for s in range(required_sum, num - 1, -1):
                dp[s] += dp[s - num]
        
        return dp[required_sum]
```

---

## Why Backward Iteration?

To ensure:

* Each number is used **at most once**
* Prevents overwriting values needed for current iteration

This preserves **0/1 knapsack behavior**.

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(n * requiredSum)` |
| Space | `O(requiredSum)` |

---

## Summary

* Convert sign problem → **subset sum count**
* Proper state: `dp[i][s] = number of ways`
* Transition: include / exclude
* Base case: `dp[0][0] = 1`
* Final answer: `dp[n][requiredSum]`

If you want next:

* recursive + memo version
* explanation for handling `0`s
* visualization as DP tree
* comparison with backtracking approach

Tell me which one you want.