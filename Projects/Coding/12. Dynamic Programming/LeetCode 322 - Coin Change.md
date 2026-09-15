---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 322: Coin Change"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 322: Coin Change

**LeetCode 322 (Coin Change)** using **Dynamic Programming**, with explicit **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 322 — Coin Change

### Problem Statement

You are given an integer array `coins` representing coins of different denominations and an integer `amount`.  
Return the **fewest number of coins** needed to make up that amount.  
If it is not possible, return `-1`.

You may use **unlimited coins of each denomination**.

---

## Key Observation

* Order does **not** matter.
* Each coin can be used **multiple times**.
* This is a classic **unbounded knapsack / minimum optimization DP** problem.
* Greedy fails (e.g., coins `[1, 3, 4]`, amount `6`).

---

## DP State Definition

### 1D DP (Optimal and Standard)

Let:

```
dp[x] = minimum number of coins required to make amount x
```

Goal:

```
dp[amount]
```

---

## Base Case

```
dp[0] = 0
```

Reason:

* Zero coins are needed to make amount `0`.

For all other values:

```
dp[x] = ∞  (initially)
```

---

## State Transition

For every amount `x` from `1` to `amount`  
For every coin `c` in `coins`:

```
if x - c >= 0:
    dp[x] = min(dp[x], dp[x - c] + 1)
```

### Intuition

* To make amount `x`, try taking one coin `c`
* Remaining amount = `x - c`
* If we know the best way to make `x - c`, add one coin

---

## DP Table Creation Order

* Amount increases from `0 → amount`
* This ensures subproblems (`x - c`) are already solved

---

## Example Walkthrough

### Input

```
coins = [1, 2, 5]
amount = 11
```

---

### DP Table Initialization

```
dp = [0, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞]
index: 0  1  2  3  4  5  6  7  8  9 10 11
```

---

### Step-by-Step Filling

#### Amount = 1

* Using coin 1 → dp[1] = dp[0] + 1 = 1

```
dp = [0, 1, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞]
```

---

#### Amount = 2

* coin 1 → dp[1] + 1 = 2
* coin 2 → dp[0] + 1 = 1

```
dp = [0, 1, 1, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞]
```

---

#### Amount = 3

* coin 1 → dp[2] + 1 = 2
* coin 2 → dp[1] + 1 = 2

```
dp = [0, 1, 1, 2, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞]
```

---

#### Amount = 5

* coin 5 → dp[0] + 1 = 1 (best)

```
dp = [0, 1, 1, 2, 2, 1, ∞, ∞, ∞, ∞, ∞, ∞]
```

---

#### Amount = 11

* coin 1 → dp[10] + 1 = 3
* coin 2 → dp[9] + 1 = 4
* coin 5 → dp[6] + 1 = 3

```
dp[11] = 3
```

Final DP table:

```
dp = [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3]
```

---

## Final Answer

```
Minimum coins required = 3
```

(5 + 5 + 1)

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[x] = minimum coins to make amount x
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        for x in range(1, amount + 1):
            for coin in coins:
                if x - coin >= 0:
                    dp[x] = min(dp[x], dp[x - coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1
```

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(amount × len(coins))` |
| Space | `O(amount)` |

---

## Why This Works

* Each amount builds on **already solved smaller amounts**
* Unlimited coin usage handled naturally
* Guarantees global minimum due to DP optimization

---

If you want, next I can:

* Show **2D DP version** and why it’s worse
* Explain **why greedy fails**
* Derive this from **unbounded knapsack**
* Show **coin combination reconstruction**

Just tell me.