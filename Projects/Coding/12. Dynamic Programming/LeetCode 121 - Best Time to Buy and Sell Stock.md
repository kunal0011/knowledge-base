---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 121: Best Time to Buy and Sell Stock"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - greedy
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 121: Best Time to Buy and Sell Stock

**Target Companies:** Amazon, Apple, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Dynamic Programming / Array / Greedy  

---

### Problem Statement

You are given an array `prices` where `prices[i]` is the price of a given stock on the $i^{\text{th}}$ day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return *the maximum profit you can achieve from this transaction*. If you cannot achieve any profit, return `0`.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `prices` ($1 \le |prices| \le 10^5$).
- **Output:** An integer denoting the maximum profit possible.
- **Constraints:**
  - `1 <= prices.length <= 10^5`
  - `0 <= prices[i] <= 10^4`

---

### Key Idea & Intuition

#### 1. Prefix Minimum Viewpoint (Greedy)
If we decide to sell on day $i$, to maximize profit we must have bought on the day with the minimum price prior to day $i$:
$$\text{Profit on day } i = prices[i] - \min_{0 \le k < i} prices[k]$$
By maintaining a running prefix minimum `min_price`, we can evaluate the optimal sell profit for each day in a single linear pass.

#### 2. Finite-State Dynamic Programming Viewpoint
This problem is the foundational base case of the entire LeetCode Buy/Sell Stock series (LC 121, 122, 123, 188, 309, 714).
On any day $i$, an agent is in one of two states:
- State 0 (`hold = 0`): Not holding any stock.
- State 1 (`hold = 1`): Holding one stock.

Because we are permitted **at most one transaction** (at most one buy):
- $dp[i][0] = \max(dp[i-1][0], dp[i-1][1] + prices[i])$
  *(Stay without stock, or sell the stock held from yesterday)*
- $dp[i][1] = \max(dp[i-1][1], -prices[i])$
  *(Keep holding stock from yesterday, or buy today. Buying today starts from a baseline of $0$ profit, hence $-prices[i]$ rather than $dp[i-1][0] - prices[i]$)*

Both formulations simplify to tracking two variables in $\mathcal{O}(1)$ space and $\mathcal{O}(N)$ time.

---

### Solution Approach (Step-by-Step)

1. **Initialize Running State:**
   - `min_price = infinity` (or `prices[0]`)
   - `max_profit = 0`
2. **One-Pass Linear Scan:**
   - For each price $p$ in `prices`:
     - If $p < min\_price$:
       - Update $min\_price = p$.
     - Else:
       - Update $max\_profit = \max(max\_profit, p - min\_price)$.
3. **Return Result:**
   - Return `max_profit`.

---

### Visual Algorithm Walkthrough

#### Trace for `prices = [7, 1, 5, 3, 6, 4]`
```
Day 0: price = 7
  min_price = 7
  profit = 7 - 7 = 0
  max_profit = 0

Day 1: price = 1
  price < min_price (1 < 7) -> update min_price = 1
  max_profit = 0

Day 2: price = 5
  profit = 5 - 1 = 4
  max_profit = max(0, 4) = 4

Day 3: price = 3
  profit = 3 - 1 = 2
  max_profit = max(4, 2) = 4

Day 4: price = 6
  profit = 6 - 1 = 5
  max_profit = max(4, 5) = 5

Day 5: price = 4
  profit = 4 - 1 = 3
  max_profit = max(5, 3) = 5

Final Answer: max_profit = 5 (Buy at 1 on Day 1, Sell at 6 on Day 4).
```

---

### Solved Examples with Multiple Inputs

| Prices Array | `min_price` Evolution | Potential Daily Profits | Output |
|---|---|---|---|
| `[7, 1, 5, 3, 6, 4]` | $7 \to 1 \to 1 \to 1 \to 1 \to 1$ | $0, 0, 4, 2, 5, 3$ | `5` |
| `[7, 6, 4, 3, 1]` | $7 \to 6 \to 4 \to 3 \to 1$ | Strictly decreasing: all profits $\le 0$ | `0` |
| `[2, 4, 1]` | $2 \to 2 \to 1$ | $0, 2, 0$ | `2` |
| `[5]` | $5$ | Single day, cannot sell | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price: float = float('inf')
        max_profit: int = 0
        
        for price in prices:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, price - min_price)
                
        return max_profit
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int maxProfit(const std::vector<int>& prices) {
        int min_price = INT_MAX;
        int max_profit = 0;
        
        for (int price : prices) {
            if (price < min_price) {
                min_price = price;
            } else {
                max_profit = std::max(max_profit, price - min_price);
            }
        }
        
        return max_profit;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        
        for (int price : prices) {
            if (price < minPrice) {
                minPrice = price;
            } else {
                maxProfit = Math.max(maxProfit, price - minPrice);
            }
        }
        
        return maxProfit;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |prices|$. We traverse the array in a single linear pass with constant-time updates.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only two scalar integer variables (`min_price`, `max_profit`) are stored.

---

### Takeaway Pattern & Interview Traps

1. **Future Knowledge (No Time Travel):** You cannot sell before you buy. A common brute-force trap is trying $\max(prices) - \min(prices)$, which fails when the minimum appears after the maximum (e.g., `[7, 6, 4, 3, 1]`).
2. **Kadane's Algorithm Equivalence:** If you take the difference array $\Delta[i] = prices[i] - prices[i - 1]$, this problem is mathematically identical to the Maximum Subarray problem (LC 53), solvable by Kadane's algorithm.
3. **Bridge to General Stock DP:** In multi-transaction variants (LC 122, 123, 188), buying retains accumulated profits ($dp[i-1][0] - prices[i]$). In LC 121, buying ignores prior profits and resets to $-prices[i]$ because at most one transaction is permitted.