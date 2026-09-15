---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 714: Best Time to Buy and Sell Stock with Transaction Fee"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 714: Best Time to Buy and Sell Stock with Transaction Fee

**LeetCode 714 – Best Time to Buy and Sell Stock with Transaction Fee**, structured exactly as requested.

---

## LeetCode 714 — Best Time to Buy and Sell Stock with Transaction Fee

---

## 1. Problem Statement

You are given an integer array `prices` where `prices[i]` is the price of a stock on day `i`, and an integer `fee` representing a **transaction fee**.

You may complete as many transactions as you like, but **you must pay the transaction fee for each sale**.

**Constraints**

* You may not engage in multiple transactions at the same time.
* You must sell the stock before you buy again.

**Goal**  
Return the **maximum profit** you can achieve.

---

## 2. Key Observation

This is an **unlimited transactions** stock problem with a **cost per transaction**.

### Without fee:

* We greedily take every upward movement.

### With fee:

* Small profits can become **unprofitable**.
* We must **delay selling** until profit exceeds the transaction fee.

---

## 3. Greedy Insight (Critical Trick)

Instead of tracking every buy/sell explicitly, we track **two states**:

### State Definitions

* `cash`: Maximum profit if **we do not hold a stock** at the end of day `i`
* `hold`: Maximum profit if **we hold a stock** at the end of day `i`

---

## 4. Greedy Transitions

For each price `p`:

### Option 1: Do nothing

* Keep current state

### Option 2: Perform an action

* Buy (from `cash`)
* Sell (from `hold` and pay fee)

---

### Transition Formulas

```
hold = max(hold, cash - price)
cash = max(cash, hold + price - fee)
```

### Why this works

* `hold` represents the **best effective buy price so far**
* `cash` represents realized profit after selling (with fee applied)
* This implicitly **merges multiple transactions** into a single greedy process

---

## 5. Initialization

```
cash = 0
hold = -prices[0]
```

* On day 0:

  * No stock → profit = 0
  * Buy stock → profit = -prices[0]

---

## 6. Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash = 0
        hold = -prices[0]

        for price in prices[1:]:
            # store previous cash before update
            prev_cash = cash

            # sell or do nothing
            cash = max(cash, hold + price - fee)

            # buy or do nothing
            hold = max(hold, prev_cash - price)

        return cash
```

---

## 7. Complete Worked Example (Step-by-Step)

### Input

```
prices = [1, 3, 2, 8, 4, 9]
fee = 2
```

---

### Day-by-Day State Evolution

| Day | Price | Action Considered | hold | cash |
| --- | --- | --- | --- | --- |
| 0 | 1 | Buy | -1 | 0 |
| 1 | 3 | Sell? → profit = 3 - 1 - 2 = 0 | -1 | 0 |
| 2 | 2 | No sell (profit negative) | -1 | 0 |
| 3 | 8 | Sell → profit = 8 - 1 - 2 = 5 | -1 | 5 |
| 4 | 4 | Buy again | 1 | 5 |
| 5 | 9 | Sell → profit = 9 - 4 - 2 = 3 | 1 | 8 |

---

### Explanation of Key Steps

#### Day 3 (Price = 8)

* Selling now yields:

  ```
  profit = 8 - 1 - 2 = 5
  ```
* This is better than holding → sell

#### Day 4 (Price = 4)

* Buying again:

  ```
  effective cost = 5 - 4 = 1
  ```

#### Day 5 (Price = 9)

* Selling yields:

  ```
  profit = 9 - 4 - 2 = 3
  total profit = 5 + 3 = 8
  ```

---

## 8. Final Answer

```
Maximum Profit = 8
```

---

## 9. Why This Is Greedy (Not DP Table)

* We **collapse DP states** into two rolling variables
* We **delay selling** until fee is covered
* We **reuse profits** to simulate continuous reinvestment
* Time Complexity: **O(n)**
* Space Complexity: **O(1)**

---

## 10. Interview Summary (One-Liner)

> Treat the transaction fee as part of the sell operation and maintain the best effective buy price using `hold`, while `cash` tracks realized profit greedily.

---

If you want, I can also provide:

* DP table version (for intuition)
* Visual state machine diagram
* Comparison with LeetCode 122 and 309
* Common pitfalls and wrong greedy approaches

Just tell me.