---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 416: Partition Equal Subset Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 416: Partition Equal Subset Sum

**LeetCode 416 – Partition Equal Subset Sum**, with **formal state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 416 – Partition Equal Subset Sum

### Problem Statement

You are given an integer array `nums`.  
Return `true` if you can partition the array into **two subsets** such that the **sum of elements in both subsets is equal**.

---

## Key Observation

Let:

* `total = sum(nums)`

If `total` is **odd**, it is **impossible** to split it into two equal integers.

So the problem reduces to:

> **Can we find a subset of `nums` whose sum is `target = total / 2`?**

This is a classic **0/1 Knapsack (Subset Sum)** problem.

---

## DP Formulation (Core of the Solution)

### DP State Definition

We define a **boolean DP table**:

```
dp[i][s] = True if using the first i elements (nums[0..i-1]),
           we can form a subset with sum exactly s
```

* `i` → number of elements considered
* `s` → target sum

---

### DP Dimensions

```
n = len(nums)
target = total // 2

dp size = (n + 1) x (target + 1)
```

---

### Base Cases

1. **Zero sum is always possible** (choose nothing):

```
dp[i][0] = True   for all i
```

2. **No elements but positive sum is impossible**:

```
dp[0][s] = False  for s > 0
```

---

### State Transition

For element `nums[i-1]`, we have two choices:

#### Case 1: Do NOT take the element

```
dp[i][s] = dp[i-1][s]
```

#### Case 2: Take the element (only if s >= nums[i-1])

```
dp[i][s] = dp[i-1][s] OR dp[i-1][s - nums[i-1]]
```

---

### Final DP Transition Formula

```
if s < nums[i-1]:
    dp[i][s] = dp[i-1][s]
else:
    dp[i][s] = dp[i-1][s] or dp[i-1][s - nums[i-1]]
```

---

### Answer

```
return dp[n][target]
```

---

## DP Table Construction – Worked Example

### Input

```text
nums = [1, 5, 11, 5]
```

### Step 1: Compute Target

```
total = 22
target = 11
```

---

### Step 2: DP Table Layout

Rows → elements considered  
Columns → possible sums (0 → 11)

```
       0  1  2  3  4  5  6  7  8  9 10 11
dp[0]  T  F  F  F  F  F  F  F  F  F  F  F
dp[1]  T  T  F  F  F  F  F  F  F  F  F  F
dp[2]  T  T  F  F  F  T  T  F  F  F  F  F
dp[3]  T  T  F  F  F  T  T  F  F  F  F  T
dp[4]  T  T  F  F  F  T  T  F  F  F  T  T
```

---

### How the Table Was Built

#### Row 1 (element = 1)

* Sum `1` achievable by picking `1`

#### Row 2 (element = 5)

* Sum `5` achievable by `{5}`
* Sum `6` achievable by `{1,5}`

#### Row 3 (element = 11)

* Sum `11` achievable by `{11}`

#### Row 4 (element = 5)

* Sum `10` achievable by `{5,5}`
* Sum `11` still achievable (`{11}` or `{1,5,5}`)

---

### Final Cell

```
dp[4][11] = True
```

✅ Partition is possible.

---

## Python 3 DP Implementation (2D DP)

```python
from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False

        target = total // 2
        n = len(nums)

        dp = [[False] * (target + 1) for _ in range(n + 1)]

        # Base case: sum = 0 is always possible
        for i in range(n + 1):
            dp[i][0] = True

        for i in range(1, n + 1):
            for s in range(1, target + 1):
                if s < nums[i - 1]:
                    dp[i][s] = dp[i - 1][s]
                else:
                    dp[i][s] = dp[i - 1][s] or dp[i - 1][s - nums[i - 1]]

        return dp[n][target]
```

---

## Time and Space Complexity

* **Time:** `O(n * target)`
* **Space:** `O(n * target)`

---

## Optimization Note (Optional Insight)

This problem can be optimized to **1D DP** because each row depends only on the previous row, but the **2D DP** is the **best version for learning state definition and transitions**, which is why it is shown here.

---

If you want next:

* 1D DP version with reasoning
* Backtracking comparison
* Relation to Knapsack template
* Visualization of transitions for any custom input

State your preference.