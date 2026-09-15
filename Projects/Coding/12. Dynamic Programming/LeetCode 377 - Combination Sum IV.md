---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 377: Combination Sum IV"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 377: Combination Sum IV

**LeetCode 377 – Combination Sum IV**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 377 — Combination Sum IV

### Problem Statement

You are given an array of **distinct positive integers** `nums` and a target integer `target`.

Return the **number of possible combinations** that add up to `target`.

**Important:**

* **Order matters** (i.e., `[1,2]` and `[2,1]` are considered different).
* You may use the same number **multiple times**.

---

## Key Observation

This is a **counting DP problem with order sensitivity**.

* Since **order matters**, we build solutions **by target value**, not by index.
* For each intermediate sum, try **adding every number** in `nums` as the **last element**.

This directly leads to a **1D DP** approach.

---

## DP State Definition

Let:

```
dp[t] = number of ordered combinations that sum to t
```

Our final answer will be:

```
dp[target]
```

---

## Base Case

```
dp[0] = 1
```

Why?

* There is **exactly one way** to make sum `0`: choose nothing.
* This acts as the foundation for building larger sums.

---

## DP Transition (Core Logic)

To compute `dp[t]`:

```
dp[t] = Σ dp[t - num]   for all num ∈ nums where t - num ≥ 0
```

### Intuition

* Assume `num` is the **last element** used.
* The number of ways to reach `t` using `num` at the end is:

  ```
  dp[t - num]
  ```

---

## DP Table Construction Order

Because `dp[t]` depends on **smaller targets**, we iterate **in increasing order of t**:

```
for t = 1 to target:
    for num in nums:
        if t >= num:
            dp[t] += dp[t - num]
```

---

## Example Walkthrough

### Input

```text
nums = [1, 2, 3]
target = 4
```

---

### Step-by-Step DP Table

Initialize:

```
dp = [1, 0, 0, 0, 0]
      0  1  2  3  4
```

---

#### t = 1

* Use 1 → dp[1] += dp[0] = 1

```
dp = [1, 1, 0, 0, 0]
```

Combinations:

* [1]

---

#### t = 2

* Use 1 → dp[2] += dp[1] = 1
* Use 2 → dp[2] += dp[0] = 1

```
dp = [1, 1, 2, 0, 0]
```

Combinations:

* [1,1]
* [2]

---

#### t = 3

* Use 1 → dp[3] += dp[2] = 2
* Use 2 → dp[3] += dp[1] = 1
* Use 3 → dp[3] += dp[0] = 1

```
dp = [1, 1, 2, 4, 0]
```

Combinations:

* [1,1,1]
* [1,2]
* [2,1]
* [3]

---

#### t = 4

* Use 1 → dp[4] += dp[3] = 4
* Use 2 → dp[4] += dp[2] = 2
* Use 3 → dp[4] += dp[1] = 1

```
dp = [1, 1, 2, 4, 7]
```

---

### Final Answer

```
dp[4] = 7
```

---

### All Valid Combinations

```
[1,1,1,1]
[1,1,2]
[1,2,1]
[2,1,1]
[2,2]
[1,3]
[3,1]
```

---

## Python 3 Implementation (with typing)

```python
from typing import List

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1

        for t in range(1, target + 1):
            for num in nums:
                if t >= num:
                    dp[t] += dp[t - num]

        return dp[target]
```

---

## Complexity Analysis

* **Time Complexity:** `O(target × len(nums))`
* **Space Complexity:** `O(target)`

---

## Why Order Matters Here (Very Important Insight)

If the loops were reversed:

```
for num in nums:
    for t in range(num, target + 1):
        dp[t] += dp[t - num]
```

Then:

* You would count **combinations where order does NOT matter**
* That version corresponds to **Coin Change II (LeetCode 518)**

---

## Summary

* **DP Type:** Unbounded knapsack (order-sensitive)
* **State:** `dp[t]` → number of ways to form sum `t`
* **Transition:** `dp[t] += dp[t - num]`
* **Traversal:** target first → numbers inside
* **Key Trick:** `dp[0] = 1`

If you want, I can also:

* Draw the **DP tree expansion**
* Compare this with **LeetCode 39 / 518**
* Convert this into a **top-down memoized recursion**