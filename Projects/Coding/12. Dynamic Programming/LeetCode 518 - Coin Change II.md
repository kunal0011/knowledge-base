---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 518: Coin Change II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 518: Coin Change II

**LeetCode 518 – Coin Change II**, with **precise state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 518 — Coin Change II

### Problem Statement

You are given:

* An integer `amount`
* An array of integers `coins` representing different coin denominations

Return the **number of combinations** that make up `amount`.  
You may use **unlimited coins of each denomination**.

**Important:**

* Order does **not** matter (i.e., `[1,2]` and `[2,1]` are the same combination).

---

## Key Observation

This is a **counting combinations** problem with **unlimited supply** of items.

Therefore:

* Each coin can be used **multiple times**
* To avoid counting permutations, we must **process coins in a fixed order**

This is a classic **Unbounded Knapsack (Combination variant)**.

---

## DP State Definition

We use a **2D DP table** for clarity.

### State

```
dp[i][j] = number of ways to make amount j
           using first i coins (coins[0 ... i-1])
```

Where:

* `i` → number of coin types considered
* `j` → target amount

---

## Base Case

1. **Amount = 0**

```
dp[i][0] = 1   for all i
```

There is exactly **one way** to make amount 0 → choose no coins.

2. **No coins**

```
dp[0][j] = 0   for all j > 0
```

With zero coins, you cannot form any positive amount.

---

## State Transition

For coin value `coins[i-1]`:

### Two choices

1. **Do not take the coin**

```
dp[i-1][j]
```

2. **Take the coin (unlimited times)**

```
dp[i][j - coins[i-1]]   if j >= coins[i-1]
```

### Transition Formula

```
dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]]
```

---

## DP Table Construction Order

* Outer loop → coins
* Inner loop → amount (0 → target)

This guarantees:

* Each combination is counted **once**
* No permutation duplication

---

## Example Walkthrough

### Input

```
amount = 5
coins = [1, 2, 5]
```

### DP Table

| i / j | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (no coins) | 1 | 0 | 0 | 0 | 0 | 0 |
| 1 (coin=1) | 1 | 1 | 1 | 1 | 1 | 1 |
| 2 (coin=2) | 1 | 1 | 2 | 2 | 3 | 3 |
| 3 (coin=5) | 1 | 1 | 2 | 2 | 3 | 4 |

---

### Explanation

#### Using coin `1`

Only one way to form any amount:

```
[1+1+1+...]
```

#### Using coins `1, 2`

For amount `4`:

```
[1+1+1+1]
[1+1+2]
[2+2]
```

#### Using coins `1, 2, 5`

For amount `5`, new combination:

```
[5]
```

Total = **4**

---

## Final Answer

```
dp[3][5] = 4
```

---

## Python 3 Implementation (with typing)

```python
from typing import List

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        n = len(coins)
        
        dp = [[0] * (amount + 1) for _ in range(n + 1)]
        
        # Base case: amount = 0
        for i in range(n + 1):
            dp[i][0] = 1
        
        for i in range(1, n + 1):
            for j in range(amount + 1):
                # Not take coin
                dp[i][j] = dp[i - 1][j]
                
                # Take coin (unbounded)
                if j >= coins[i - 1]:
                    dp[i][j] += dp[i][j - coins[i - 1]]
        
        return dp[n][amount]
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(n × amount)` |
| Space | `O(n × amount)` |
| Optimizable to | `O(amount)` (1D DP) |

---

## Interview Insight

> If the question says **“number of combinations”** and **order does not matter**,  
> think **Unbounded Knapsack (Combination DP)**.

If you want, I can also:

* Convert this to **1D DP**
* Draw a **DP transition tree**
* Compare it with **LeetCode 322 (minimum coins)**

Tell me how deep you want to go.