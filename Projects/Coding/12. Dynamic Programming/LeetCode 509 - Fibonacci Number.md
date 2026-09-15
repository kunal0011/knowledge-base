---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 509: Fibonacci Number"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 509: Fibonacci Number

**LeetCode 509 – Fibonacci Number**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## Problem Statement (LeetCode 509)

The Fibonacci numbers are defined as:

* `F(0) = 0`
* `F(1) = 1`
* `F(n) = F(n - 1) + F(n - 2)` for `n ≥ 2`

**Given** an integer `n`, return `F(n)`.

---

## Dynamic Programming Approach

This problem is a canonical example of **1-D Dynamic Programming**.

---

## 1. DP State Definition

Let:

```
dp[i] = the i-th Fibonacci number, i.e., F(i)
```

Each state represents the answer to a **smaller subproblem**.

---

## 2. State Transition

From the problem definition:

```
dp[i] = dp[i - 1] + dp[i - 2]
```

This transition is valid because Fibonacci numbers are defined recursively using the previous two values.

---

## 3. Base Cases

The smallest subproblems are known upfront:

```
dp[0] = 0
dp[1] = 1
```

These act as the foundation of the DP table.

---

## 4. DP Table Construction (Bottom-Up)

We compute the DP table iteratively from `2` to `n`.

### Algorithm

1. Initialize a DP array of size `n + 1`
2. Fill base cases
3. Iterate and apply the transition

---

## 5. Example Walkthrough

### Input

```
n = 6
```

### Step-by-Step DP Table Filling

| i | dp[i] calculation | dp[i] |
| --- | --- | --- |
| 0 | base case | 0 |
| 1 | base case | 1 |
| 2 | dp[1] + dp[0] = 1 + 0 | 1 |
| 3 | dp[2] + dp[1] = 1 + 1 | 2 |
| 4 | dp[3] + dp[2] = 2 + 1 | 3 |
| 5 | dp[4] + dp[3] = 3 + 2 | 5 |
| 6 | dp[5] + dp[4] = 5 + 3 | 8 |

### Final DP Table

```
Index:  0  1  2  3  4  5  6
dp:     0  1  1  2  3  5  8
```

**Answer:** `dp[6] = 8`

---

## 6. Python 3 DP Implementation (With Typing)

```python
from typing import List

class Solution:
    def fib(self, n: int) -> int:
        # Base cases
        if n <= 1:
            return n

        # DP table
        dp: List[int] = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1

        # Fill DP table
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
```

---

## 7. Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** |

---

## 8. Optimization Insight (Why This Is DP)

* **Overlapping subproblems**: `F(n-1)` and `F(n-2)` are reused
* **Optimal substructure**: `F(n)` depends only on optimal results of smaller states
* DP avoids exponential recursion by **storing results**

> This problem also admits a **space-optimized DP** (`O(1)` space), but the above solution is the **cleanest for learning state definition and table construction**.

---

If you want, I can also:

* Show **space-optimized DP**
* Compare **recursion vs memoization vs tabulation**
* Generalize this to **climbing stairs / tribonacci / DP patterns**

Just tell me.