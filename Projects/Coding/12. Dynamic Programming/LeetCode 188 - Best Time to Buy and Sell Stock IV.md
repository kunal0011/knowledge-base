---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 188: Best Time to Buy and Sell Stock IV"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 188: Best Time to Buy and Sell Stock IV

**LeetCode 188 – Best Time to Buy and Sell Stock IV**, aligned with interview-grade expectations: **state definition → transitions → DP table → worked example**.

---

## LeetCode 188: Best Time to Buy and Sell Stock IV

### Problem Statement

You are given:

* An integer `k` — maximum number of transactions allowed.
* An array `prices`, where `prices[i]` is the stock price on day `i`.

Each transaction consists of **one buy followed by one sell**.  
You may not hold more than one stock at a time.

**Return the maximum profit you can achieve.**

---

## Key Observation

A transaction = **Buy → Sell**

At any day, your decision depends on:

1. Which **day** you are on
2. How many **transactions are remaining**
3. Whether you are **holding a stock or not**

This naturally leads to a **DP state with 3 dimensions**.

---

## DP State Definition

Let:

```
dp[day][transactions_left][holding]
```

Where:

* `day`: current index in prices array (0-based)
* `transactions_left`: number of transactions still allowed
* `holding`:

  * `0` → not holding stock
  * `1` → holding stock

### Meaning

`dp[d][t][h]` = **maximum profit achievable starting from day `d`**, with `t` transactions left, and `h` holding state.

---

## Base Conditions

1. **No more days**

```
if day == n → profit = 0
```

2. **No transactions left**

```
if transactions_left == 0 → profit = 0
```

---

## State Transitions

### Case 1: Not holding a stock (`holding = 0`)

You have **two choices**:

#### Option 1: Skip the day

```
profit = dp[day+1][t][0]
```

#### Option 2: Buy stock

Buying does **not** consume a transaction yet.

```
profit = -prices[day] + dp[day+1][t][1]
```

#### Transition

```
dp[day][t][0] = max(
    dp[day+1][t][0],
    -prices[day] + dp[day+1][t][1]
)
```

---

### Case 2: Holding a stock (`holding = 1`)

You have **two choices**:

#### Option 1: Hold stock

```
profit = dp[day+1][t][1]
```

#### Option 2: Sell stock

Selling **consumes one transaction**.

```
profit = prices[day] + dp[day+1][t-1][0]
```

#### Transition

```
dp[day][t][1] = max(
    dp[day+1][t][1],
    prices[day] + dp[day+1][t-1][0]
)
```

---

## DP Table Construction

### Dimensions

```
days: n
transactions: k + 1
holding states: 2
```

```
dp[n+1][k+1][2]
```

We fill the table **bottom-up**:

* Days: from `n-1 → 0`
* Transactions: from `1 → k`
* Holding: `0` and `1`

---

## Worked Example

### Input

```
k = 2
prices = [3, 2, 6, 5]
```

### Intuition

* Buy at 2 → Sell at 6 → Profit = 4
* Buy at 5 → Sell? No further gain

---

### DP Snapshot (Key States)

#### Day 3 (price = 5)

```
dp[3][1][0] = max(0, -5) = 0
dp[3][1][1] = max(0, 5) = 5
```

#### Day 2 (price = 6)

```
dp[2][1][0] = max(0, -6 + 5) = 0
dp[2][1][1] = max(5, 6) = 6
```

#### Day 1 (price = 2)

```
dp[1][2][0] = max(0, -2 + 6) = 4
dp[1][2][1] = max(6, 2) = 6
```

#### Day 0 (price = 3)

```
dp[0][2][0] = max(4, -3 + 6) = 4
```

---

### Final Answer

```
dp[0][k][0] = 4
```

---

## Python 3 Implementation (Typed, Bottom-Up)

```python
from typing import List

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        if n == 0 or k == 0:
            return 0

        # Optimization: k >= n//2 becomes unlimited transactions
        if k >= n // 2:
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit

        dp = [[[0] * 2 for _ in range(k + 1)] for _ in range(n + 1)]

        for day in range(n - 1, -1, -1):
            for t in range(1, k + 1):
                # Not holding
                dp[day][t][0] = max(
                    dp[day + 1][t][0],
                    -prices[day] + dp[day + 1][t][1]
                )

                # Holding
                dp[day][t][1] = max(
                    dp[day + 1][t][1],
                    prices[day] + dp[day + 1][t - 1][0]
                )

        return dp[0][k][0]
```

---

## class SolutionFull2D: def maxProfit(self, k: int, prices: List[int]) -> int: if not prices or k == 0: return 0 n = len(prices) # dp[i][j][0/1] = max profit at day i, j transactions, not/holding dp = [[[0, float('-inf')] for \_ in range(k + 1)] for \_ in range(n + 1)] # Base case: day 0, no transactions, not holding = 0 (already set) # Forward loop for i in range(1, n + 1): price = prices[i - 1] for j in range(k + 1): # Not holding: either rest or sold today dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + price) # Holding: either rest or bought today if j > 0: dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - price) else: dp[i][j][1] = dp[i - 1][j][1] # Can't buy if no transactions left return dp[n][k][0]

## Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(n * k)` |
| Space | `O(n * k * 2)` (can be optimized to `O(k)`) |

---

## Interview Takeaways

* Transaction count decreases **only on SELL**
* Holding state prevents multiple simultaneous buys
* This DP pattern generalizes to **all stock buy/sell problems**
* LeetCode 121, 122, 123, 188, 309 all reduce to this model

If you want, I can:

* Derive the **space-optimized O(k)** version
* Show **state compression to 1D**
* Compare this with **LeetCode 123**
* Draw a **DP state transition diagram**

State which one you want next.