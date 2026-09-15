---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 714: Best Time to Buy and Sell Stock with Transaction Fee"
tags:
  - leetcode
  - coding
  - greedy
  - dynamic-programming
  - state-machine
  - array
  - amazon
  - google
---

# LeetCode 714: Best Time to Buy and Sell Stock with Transaction Fee

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / State Machine Dynamic Programming / Array  

---

### Problem Statement

You are given an array `prices` where `prices[i]` is the price of a given stock on the $i$-th day, and an integer `fee` representing a transaction fee.

Find the maximum profit you can achieve. You may complete as many transactions as you like, but you need to pay the transaction fee for each transaction.

Note:
- You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
- The transaction fee is only charged once for each stock purchase and sale.

---

### Input & Output Formats & Constraints

- **Input:**
  - `prices`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{prices.length} \le 5 \times 10^4$).
  - `fee`: `int` ($0 \le fee \le 5 \times 10^4$).
- **Output:**
  - `int` — maximum total profit after deducting transaction fees.
- **Constraints:**
  - $1 \le \text{prices.length} \le 5 \times 10^4$
  - $1 \le \text{prices}[i] < 5 \times 10^4$
  - $0 \le \text{fee} < 5 \times 10^4$

---

### Key Idea & Intuition

Unlike LeetCode 122 (where transactions are free and we can capture every single positive daily price difference), each transaction here costs `fee`. Micro-fluctuations where the price increase is $\le fee$ are unprofitable and should not trigger transactions.

We can view this problem through two equivalent paradigms:

#### Approach 1: State Machine DP with $\mathcal{O}(1)$ Space
At the end of each day $i$, we are in one of two states:
1. `cash`: Maximum profit achievable if we **do not hold** stock at the end of day $i$.
2. `hold`: Maximum profit achievable if we **are holding** 1 share of stock at the end of day $i$.

Transitions:
- `cash = max(cash, hold + prices[i] - fee)`:
  Either stay without stock, or sell the stock held previously for `prices[i]` minus `fee`.
- `hold = max(hold, cash - prices[i])`:
  Either keep holding the stock, or buy today's stock at cost `prices[i]` using existing cash.

#### Approach 2: Pure Greedy with Virtual Buy Price
Maintain an effective purchase cost:
$$\text{buy} = \text{prices}[0] + \text{fee}$$
For each day $i$:
1. If $\text{prices}[i] + \text{fee} < \text{buy}$:
   - A cheaper buying opportunity has appeared $\rightarrow$ lower $\text{buy} = \text{prices}[i] + \text{fee}$.
2. Else if $\text{prices}[i] > \text{buy}$:
   - Selling today is profitable:
     $$\text{profit} \mathrel{+}= \text{prices}[i] - \text{buy}$$
   - **The Greedy Rollback Trick:** Set $\text{buy} = \text{prices}[i]$!
     - Why? If the price increases even further tomorrow to $\text{prices}[i+1]$, selling tomorrow instead of today would yield $\text{prices}[i+1] - \text{prices}[i]$ additional profit without paying another fee.
     - By setting $\text{buy} = \text{prices}[i]$ (without fee), subsequent gains accumulate continuously as part of the same transaction!

Both approaches achieve strict $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ auxiliary space.

---

### Solution Approach (Step-by-Step: State Machine)

1. Initialize `cash = 0` and `hold = -prices[0]`.
2. Iterate `price` across `prices[1:]`:
   - `cash = max(cash, hold + price - fee)`
   - `hold = max(hold, cash - price)`
3. Return `cash`.

---

### Visual Algorithm Walkthrough

For `prices = [1, 3, 2, 8, 4, 9]`, `fee = 2`:

```
Day 0 (price = 1):
  cash = 0
  hold = -1

Day 1 (price = 3):
  cash = max(0, -1 + 3 - 2) = max(0, 0) = 0
  hold = max(-1, 0 - 3) = -1

Day 2 (price = 2):
  cash = max(0, -1 + 2 - 2) = 0
  hold = max(-1, 0 - 2) = -1

Day 3 (price = 8):
  cash = max(0, -1 + 8 - 2) = 5 (Sell held stock at 8!)
  hold = max(-1, 5 - 8) = -1

Day 4 (price = 4):
  cash = max(5, -1 + 4 - 2) = 5
  hold = max(-1, 5 - 4) = 1 (Buy stock at 4 with existing 5 profit!)

Day 5 (price = 9):
  cash = max(5, 1 + 9 - 2) = 8 (Sell held stock at 9!)
  hold = max(1, 8 - 9) = 1

Final Result: cash = 8.
Trades: Buy at 1, sell at 8 (profit = 8 - 1 - 2 = 5); Buy at 4, sell at 9 (profit = 9 - 4 - 2 = 3). Total = 8.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `prices = [1, 3, 2, 8, 4, 9]`, `fee = 2`
- **Output:** `8`

#### Example 2:
- **Input:** `prices = [1, 3, 7, 5, 10, 3]`, `fee = 3`
- **Tracing:** Buy at 1, sell at 10 $\rightarrow$ profit $10 - 1 - 3 = 6$.
- **Output:** `6`

#### Example 3 (No Profitable Trade):
- **Input:** `prices = [1, 2, 3]`, `fee = 5`
- **Tracing:** Max increase is $2 < 5$. Zero transactions made.
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3 (State Machine DP)
```python
from typing import List

class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash = 0             # Max profit without holding stock
        hold = -prices[0]    # Max profit while holding stock
        
        for price in prices[1:]:
            cash = max(cash, hold + price - fee)
            hold = max(hold, cash - price)
            
        return cash
```

#### C++17 (State Machine DP)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxProfit(const std::vector<int>& prices, int fee) {
        int cash = 0;
        int hold = -prices[0];
        int n = static_cast<int>(prices.size());
        
        for (int i = 1; i < n; ++i) {
            cash = std::max(cash, hold + prices[i] - fee);
            hold = std::max(hold, cash - prices[i]);
        }
        
        return cash;
    }
};
```

#### Java 17 (State Machine DP)
```java
class Solution {
    public int maxProfit(int[] prices, int fee) {
        int cash = 0;
        int hold = -prices[0];
        
        for (int i = 1; i < prices.length; i++) {
            cash = Math.max(cash, hold + prices[i] - fee);
            hold = Math.max(hold, cash - prices[i]);
        }
        
        return cash;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear scan through `prices`.
  - Constant number of arithmetic operations and max comparisons per day.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only two primitive integer variables `cash` and `hold` are tracked.

---

### Takeaway Pattern & Interview Traps

- **Deducting the Fee:** You can deduct the fee upon selling (`hold + price - fee`) OR upon buying (`cash - price - fee`), but **never both**. Deducting on selling is standard because selling is when the transaction completes.
- **Selling and Re-buying on Same Day:** Notice in DP transitions, updating `cash` then `hold` using the new `cash` represents selling and immediately buying again. Since the fee is subtracted, same-day buy-sell is never strictly optimal, so the order of updates does not produce invalid profits.