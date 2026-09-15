---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 123: Best Time to Buy and Sell Stock III"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 123: Best Time to Buy and Sell Stock III

**LeetCode 123 – Best Time to Buy and Sell Stock III**, focusing on **state definition, transitions, DP table construction, and a worked example**.

---

## Problem Statement

You are given an integer array `prices`, where `prices[i]` is the price of a stock on day `i`.

You may complete **at most two transactions**.

* One transaction = **buy once and sell once**
* You must **sell before you buy again**

Return the **maximum profit** you can achieve.

---

## Key Observation

* Each transaction has **two actions**: `buy → sell`
* With **at most 2 transactions**, we can model the problem as a **state machine**
* On any day, your state depends on:

  1. **Day index**
  2. **How many transactions have been completed**
  3. **Whether you are holding a stock or not**

This naturally leads to **Dynamic Programming**.

---

## DP State Definition (Core Part)

We define:

```
dp[i][k][h]
```

Where:

* `i` = day index (0-based)
* `k` = number of transactions completed (0 to 2)
* `h` = holding state

  * `0` → not holding stock
  * `1` → holding stock

### Meaning

> `dp[i][k][h]` = **maximum profit on day `i`**,  
> after completing `k` transactions,  
> and current holding state = `h`

---

## State Transitions

### Case 1: Not Holding Stock (`h = 0`)

Two possibilities:

1. **Do nothing**
2. **Sell today** (which completes one transaction)

```
dp[i][k][0] =
    max(
        dp[i-1][k][0],            // no action
        dp[i-1][k-1][1] + prices[i]   // sell today
    )
```

---

### Case 2: Holding Stock (`h = 1`)

Two possibilities:

1. **Do nothing**
2. **Buy today** (does NOT complete a transaction)

```
dp[i][k][1] =
    max(
        dp[i-1][k][1],            // no action
        dp[i-1][k][0] - prices[i] // buy today
    )
```

---

## Base Initialization

### Day 0 (`i = 0`)

* No stock, no transactions → profit = 0
* Buying stock on day 0 → profit = `-prices[0]`

```
dp[0][0][0] = 0
dp[0][0][1] = -prices[0]

dp[0][1][0] = dp[0][2][0] = -∞
dp[0][1][1] = dp[0][2][1] = -∞
```

(You cannot complete a transaction on day 0)

---

## DP Table Dimensions

```
Days = n
Transactions = 0..2
Holding = 0 or 1

Total states = O(n × 2 × 2)
```

---

## Example Walkthrough

### Input

```
prices = [3, 3, 5, 0, 0, 3, 1, 4]
```

---

### Compact DP Table (Key States)

| Day | Price | dp[i][0][1] | dp[i][1][0] | dp[i][1][1] | dp[i][2][0] |
| --- | --- | --- | --- | --- | --- |
| 0 | 3 | -3 | -∞ | -∞ | -∞ |
| 1 | 3 | -3 | 0 | -3 | -∞ |
| 2 | 5 | -3 | 2 | -3 | -∞ |
| 3 | 0 | 0 | 2 | 2 | -∞ |
| 4 | 0 | 0 | 2 | 2 | -∞ |
| 5 | 3 | 0 | 3 | 2 | 5 |
| 6 | 1 | 0 | 3 | 2 | 5 |
| 7 | 4 | 0 | 4 | 2 | **6** |

---

### Interpretation

* First transaction: buy at `3`, sell at `5` → profit `2`
* Second transaction: buy at `0`, sell at `4` → profit `4`
* Total profit = `2 + 4 = 6`

---

## Final Answer

We want **not holding stock** after **at most 2 transactions**:

```
answer = dp[n-1][2][0]
```

---

## Python 3 (Typed) DP Implementation

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0

        # dp[i][k][h]
        dp = [[[float('-inf')] * 2 for _ in range(3)] for _ in range(n)]

        # Base case
        dp[0][0][0] = 0
        dp[0][0][1] = -prices[0]

        for i in range(1, n):
            for k in range(3):
                # Not holding
                dp[i][k][0] = dp[i-1][k][0]
                if k > 0:
                    dp[i][k][0] = max(
                        dp[i][k][0],
                        dp[i-1][k-1][1] + prices[i]
                    )

                # Holding
                dp[i][k][1] = max(
                    dp[i-1][k][1],
                    dp[i-1][k][0] - prices[i]
                )

        return dp[n-1][2][0]
```

---

## Why This DP Works

* Explicitly models **transaction boundaries**
* Prevents overlapping buys/sells
* Guarantees **at most 2 completed transactions**
* Cleanly generalizes to **K transactions** problems

---

If you want next:

* **Space-optimized O(1) solution**
* **State-machine version**
* **Generalization to LeetCode 188**
* **Comparison with greedy approach**

Tell me which one to proceed with.