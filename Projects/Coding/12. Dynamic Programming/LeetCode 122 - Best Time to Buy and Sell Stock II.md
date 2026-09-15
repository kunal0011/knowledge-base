---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 122: Best Time to Buy and Sell Stock II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 122: Best Time to Buy and Sell Stock II

**LeetCode 122 – Best Time to Buy and Sell Stock II**, focusing on **state definition, transitions, DP table construction, and a worked example**.

---

## Problem Statement (LeetCode 122)

You are given an integer array `prices`, where `prices[i]` is the price of a stock on day `i`.

* You may complete **as many transactions as you like**
* You may **not hold more than one stock at a time**
* You must sell before you buy again

Return the **maximum profit** you can achieve.

---

## Key Observation

At any given day, your situation can be completely described by:

1. **The day index**
2. **Whether you are holding a stock or not**

This naturally leads to a **2-state DP formulation**.

---

## DP State Definition

Let:

```
dp[i][0] = Maximum profit on day i, when you do NOT hold a stock
dp[i][1] = Maximum profit on day i, when you DO hold a stock
```

Where:

* `i` ranges from `0` to `n-1`

---

## State Transitions

### 1. Not Holding a Stock (dp[i][0])

Two possibilities on day `i`:

1. You already did not hold a stock on day `i-1`
2. You sell the stock today (which you held on day `i-1`)

```
dp[i][0] = max(
    dp[i-1][0],          // do nothing
    dp[i-1][1] + prices[i]   // sell today
)
```

---

### 2. Holding a Stock (dp[i][1])

Two possibilities on day `i`:

1. You already held a stock on day `i-1`
2. You buy a stock today (you were not holding one yesterday)

```
dp[i][1] = max(
    dp[i-1][1],          // do nothing
    dp[i-1][0] - prices[i]   // buy today
)
```

---

## Base Case (Day 0)

On the very first day:

```
dp[0][0] = 0               // no stock, no profit
dp[0][1] = -prices[0]     // buy the stock
```

---

## Final Answer

At the end of the last day, the **maximum profit must be in a non-holding state**:

```
answer = dp[n-1][0]
```

---

## Example Walkthrough

### Input

```
prices = [7, 1, 5, 3, 6, 4]
```

---

### DP Table Construction

| Day (i) | Price | dp[i][0] (No Stock) | dp[i][1] (Holding Stock) |
| --- | --- | --- | --- |
| 0 | 7 | 0 | -7 |
| 1 | 1 | max(0, -7+1)=0 | max(-7, 0-1)=-1 |
| 2 | 5 | max(0, -1+5)=4 | max(-1, 0-5)=-1 |
| 3 | 3 | max(4, -1+3)=4 | max(-1, 4-3)=1 |
| 4 | 6 | max(4, 1+6)=7 | max(1, 4-6)=1 |
| 5 | 4 | max(7, 1+4)=7 | max(1, 7-4)=3 |

---

### Explanation of Key Days

* **Day 1**: Buying at price `1` is better than holding from day 0
* **Day 2**: Sell at price `5` → profit = `4`
* **Day 3**: Buy again at price `3`
* **Day 4**: Sell at price `6` → total profit = `7`
* **Day 5**: No further profitable sell

---

## Final Result

```
Maximum Profit = 7
```

---

## Python 3 DP Implementation (with typing)

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0

        dp = [[0] * 2 for _ in range(n)]

        dp[0][0] = 0
        dp[0][1] = -prices[0]

        for i in range(1, n):
            dp[i][0] = max(
                dp[i - 1][0],
                dp[i - 1][1] + prices[i]
            )
            dp[i][1] = max(
                dp[i - 1][1],
                dp[i - 1][0] - prices[i]
            )

        return dp[n - 1][0]
```

---

## Time & Space Complexity

* **Time Complexity**: `O(n)`
* **Space Complexity**: `O(n)` (can be optimized to `O(1)` using rolling variables)

---

If you want, I can next:

* Show **space-optimized DP**
* Compare this with **greedy solution**
* Derive this from **LeetCode 121**
* Provide a **DP state machine diagram**

Just specify.