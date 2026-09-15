---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 121: Best Time to Buy and Sell Stock"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 121: Best Time to Buy and Sell Stock

**LeetCode 121 (Best Time to Buy and Sell Stock)** with **explicit state definitions, transitions, DP table construction, and a worked example**. This presentation aligns with interview-grade dynamic programming reasoning rather than the shortcut greedy intuition.

---

## LeetCode 121 — Best Time to Buy and Sell Stock

### Problem Statement

You are given an array `prices`, where `prices[i]` is the price of a given stock on day `i`.

You are allowed to complete **at most one transaction** (i.e., buy once and sell once).

Return the **maximum profit** you can achieve.  
If no profit is possible, return `0`.

---

## Why Dynamic Programming Works Here

This problem is a **finite-state optimization over time**:

* Each day is a decision point
* You can be in one of a small number of states
* Each decision depends only on the previous day

This makes it a **classic DP on days + states**.

---

## DP State Definition

Let:

```
dp[i][0] = maximum profit on day i when you DO NOT hold a stock
dp[i][1] = maximum profit on day i when you DO hold a stock
```

### Interpretation

* `dp[i][0]` → best profit after possibly selling (or never buying)
* `dp[i][1]` → best profit after buying and holding stock

---

## Constraints Incorporated into State

* Only **one transaction allowed**
* You must **buy before you sell**
* Holding stock implies you have already used your one buy

This is why no transaction count dimension is required.

---

## State Transition Equations

### 1️⃣ Not holding stock on day `i`

Two possibilities:

* You did nothing today → still not holding
* You sold today → you must have held yesterday

```
dp[i][0] = max(
    dp[i-1][0],        # do nothing
    dp[i-1][1] + prices[i]   # sell today
)
```

---

### 2️⃣ Holding stock on day `i`

Two possibilities:

* You already held from yesterday
* You buy today (only once)

```
dp[i][1] = max(
    dp[i-1][1],        # keep holding
    -prices[i]         # buy today
)
```

> Note: Buying resets profit to `-prices[i]` because this is your **only allowed buy**.

---

## Base Case (Day 0)

On day `0`:

```
dp[0][0] = 0               # no stock, no profit
dp[0][1] = -prices[0]      # buy on day 0
```

---

## DP Table Creation (Worked Example)

### Input

```
prices = [7, 1, 5, 3, 6, 4]
```

---

### Step-by-Step DP Table

| Day (i) | Price | dp[i][0] (No Stock) | dp[i][1] (Hold Stock) |
| --- | --- | --- | --- |
| 0 | 7 | 0 | -7 |
| 1 | 1 | max(0, -7+1)=0 | max(-7, -1)= -1 |
| 2 | 5 | max(0, -1+5)=4 | max(-1, -5)= -1 |
| 3 | 3 | max(4, -1+3)=4 | max(-1, -3)= -1 |
| 4 | 6 | max(4, -1+6)=5 | max(-1, -6)= -1 |
| 5 | 4 | max(5, -1+4)=5 | max(-1, -4)= -1 |

---

### Final Answer

```
max profit = dp[n-1][0] = 5
```

---

## Visualization of State Evolution

* Best buy price tracked implicitly via `dp[i][1]`
* Best sell profit tracked via `dp[i][0]`
* Only one buy ever contributes because `dp[i][1] = max(dp[i-1][1], -price[i])`

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0

        dp = [[0, 0] for _ in range(n)]

        dp[0][0] = 0
        dp[0][1] = -prices[0]

        for i in range(1, n):
            dp[i][0] = max(
                dp[i-1][0],
                dp[i-1][1] + prices[i]
            )
            dp[i][1] = max(
                dp[i-1][1],
                -prices[i]
            )

        return dp[n-1][0]
```

---

## Time & Space Complexity

* **Time:** `O(n)`
* **Space:** `O(n)` (can be optimized to `O(1)`)

---

## Key Interview Takeaways

* This is the **base case of stock DP problems**
* Same state model extends to:

  * LeetCode 122 (unlimited transactions)
  * LeetCode 123 (at most 2 transactions)
  * LeetCode 188 (k transactions)
* Greedy solution is a **compressed DP**, not a different idea

---

If you want, I can:

* Derive the **O(1) space optimization**
* Generalize this to **k transactions**
* Show how this DP reduces to the **min-price greedy logic**

Tell me how deep you want to go next.