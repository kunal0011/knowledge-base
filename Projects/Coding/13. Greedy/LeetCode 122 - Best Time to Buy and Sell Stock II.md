---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 122: Best Time to Buy and Sell Stock II"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 122: Best Time to Buy and Sell Stock II

**LeetCode 122 – Best Time to Buy and Sell Stock II**, aligned with interview expectations.

---

## 1. Problem Statement

You are given an integer array `prices`, where `prices[i]` is the price of a given stock on day `i`.

You may complete **as many transactions as you like** (i.e., buy one and sell one share of the stock multiple times).

### Constraints

* You **cannot** hold more than one share at a time.
* You **must sell** the stock before you buy again.

### Goal

Return the **maximum profit** you can achieve.

---

## 2. Key Observation

This problem allows **unlimited transactions**, which fundamentally changes the strategy compared to LeetCode 121.

### Core Insight

> Any increasing price sequence can be split into multiple profitable transactions **without affecting total profit**.

Example:

```
Buy at 1 → Sell at 5
Profit = 4

Equivalent to:
Buy at 1 → Sell at 3 (profit = 2)
Buy at 3 → Sell at 5 (profit = 2)
Total profit = 4
```

### Critical Observation

* If `prices[i] > prices[i-1]`, then that **difference is pure profit**
* There is **no advantage** in waiting to find a global peak

---

## 3. Greedy Strategy (Why It Works)

### Greedy Rule

> **Take every upward price movement.**

For every adjacent day pair:

* If `prices[i] > prices[i-1]`, add

  ```
  prices[i] - prices[i-1]
  ```

  to profit.

### Why This Is Optimal

* Transactions are independent
* No transaction fee or cooldown
* Unlimited buy/sell allowed
* Local profits sum up to global maximum profit

This greedy approach effectively simulates:

* Buying at every local minimum
* Selling at every local maximum

---

## 4. Greedy Algorithm (Step-by-Step)

1. Initialize `profit = 0`
2. Traverse prices from day `1` to `n-1`
3. For each day:

   * If today’s price is higher than yesterday’s:

     * Add the difference to `profit`
4. Return `profit`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit: int = 0

        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += prices[i] - prices[i - 1]

        return profit
```

---

## 6. Complete Worked Example (All Steps)

### Input

```
prices = [7, 1, 5, 3, 6, 4]
```

---

### Day-by-Day Processing

| Day | Price Yesterday | Price Today | Action | Profit Added | Total Profit |
| --- | --- | --- | --- | --- | --- |
| 0 | — | 7 | — | 0 | 0 |
| 1 | 7 | 1 | Down | 0 | 0 |
| 2 | 1 | 5 | Up | +4 | 4 |
| 3 | 5 | 3 | Down | 0 | 4 |
| 4 | 3 | 6 | Up | +3 | 7 |
| 5 | 6 | 4 | Down | 0 | 7 |

---

### Transaction Interpretation

* Buy at **1**, Sell at **5** → Profit = 4
* Buy at **3**, Sell at **6** → Profit = 3

Total Profit = `4 + 3 = 7`

---

## 7. Edge Case Example

### Input

```
prices = [1, 2, 3, 4, 5]
```

### Processing

```
(2 - 1) + (3 - 2) + (4 - 3) + (5 - 4)
= 1 + 1 + 1 + 1
= 4
```

### Output

```
4
```

---

## 8. Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** |

---

## 9. Interview Takeaway

* This is a **pure greedy problem**
* No DP required because:

  * Unlimited transactions
  * No transaction cost
* Think in terms of **capturing every upward slope**

---

If you want, I can also:

* Contrast this with **LeetCode 121 / 123 / 188**
* Show the **DP state machine** version and then reduce it to greedy
* Explain why greedy fails when **transaction fee or cooldown** is introduced