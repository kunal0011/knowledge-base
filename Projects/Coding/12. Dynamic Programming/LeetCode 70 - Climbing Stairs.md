---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 70: Climbing Stairs"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 70: Climbing Stairs

**LeetCode 70 – Climbing Stairs**, focusing explicitly on **state definition, transition, DP table creation, and a worked example**, as typically expected in interviews.

---

## LeetCode 70 — Climbing Stairs

### Problem Statement

You are climbing a staircase with `n` steps.  
Each time, you can climb either **1 step** or **2 steps**.

Return the **number of distinct ways** to reach the top.

---

## 1. Key Observation (Why DP?)

* At any step `i`, you could have arrived:

  * from step `i - 1` (taking 1 step)
  * from step `i - 2` (taking 2 steps)
* The total ways to reach step `i` is the **sum of ways** to reach those previous steps.

This exhibits:

* **Overlapping subproblems**
* **Optimal substructure**

Hence, **Dynamic Programming**.

---

## 2. DP State Definition

Let:

```
dp[i] = number of distinct ways to reach step i
```

Our goal is to compute `dp[n]`.

---

## 3. DP Transition (Recurrence Relation)

To reach step `i`:

* From step `i - 1`, take 1 step
* From step `i - 2`, take 2 steps

Therefore:

```
dp[i] = dp[i - 1] + dp[i - 2]
```

This is identical in structure to the **Fibonacci sequence**.

---

## 4. Base Cases

These define the foundation of the DP table:

| Step | Meaning | Value |
| --- | --- | --- |
| `dp[0]` | One way to stay at ground (do nothing) | `1` |
| `dp[1]` | Only one way: 1 step | `1` |

---

## 5. DP Table Creation (Bottom-Up)

We iteratively fill the DP table from `0` to `n`.

### Example: `n = 5`

| i (step) | dp[i] | Explanation |
| --- | --- | --- |
| 0 | 1 | Base case |
| 1 | 1 | Base case |
| 2 | 2 | `dp[1] + dp[0] = 1 + 1` |
| 3 | 3 | `dp[2] + dp[1] = 2 + 1` |
| 4 | 5 | `dp[3] + dp[2] = 3 + 2` |
| 5 | 8 | `dp[4] + dp[3] = 5 + 3` |

### Final Answer

```
dp[5] = 8
```

---

## 6. Python 3 DP Implementation (with typing)

```python
from typing import List

class Solution:
    def climbStairs(self, n: int) -> int:
        # dp[i] = number of ways to reach step i
        dp: List[int] = [0] * (n + 1)

        # base cases
        dp[0] = 1
        dp[1] = 1

        # fill dp table
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
```

---

## 7. Time and Space Complexity

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)` (can be optimized to `O(1)`)

---

## 8. (Optional) Space Optimization Insight

Since `dp[i]` only depends on the previous two states:

```
prev2 = dp[i - 2]
prev1 = dp[i - 1]
```

You can replace the DP array with two variables (Fibonacci optimization).

---

## Interview-Ready Summary

* **State:** `dp[i]` = ways to reach step `i`
* **Transition:** `dp[i] = dp[i-1] + dp[i-2]`
* **Base:** `dp[0] = 1`, `dp[1] = 1`
* **Pattern:** Fibonacci DP

If you want, I can also:

* Draw the **DP dependency tree**
* Show the **recursive + memoization version**
* Explain **why greedy does not work**
* Map this problem to **other DP staircase variants**