---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 279: Perfect Squares"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 279: Perfect Squares

**LeetCode 279 – Perfect Squares**, with explicit **state definition, transition, DP table construction, and a worked example**.

---

## Problem Statement

Given an integer `n`, return the **least number of perfect square numbers** (e.g., `1, 4, 9, 16, ...`) whose **sum equals `n`**.

**Example**

```text
Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4
```

---

## Key Observation

This is an **unbounded knapsack / coin change (min coins)** problem:

* Perfect squares are the “coins”
* You can use each square **multiple times**
* Goal: minimize the number of squares whose sum is `n`

---

## DP State Definition

### State

```
dp[i] = minimum number of perfect squares required to sum to i
```

### DP Array Size

```
dp[0...n]
```

---

## Base Case

```
dp[0] = 0
```

Zero requires zero numbers.

All other values are initialized to a large number (∞).

---

## State Transition

For each number `i` from `1` to `n`  
Try all perfect squares `sq = j*j` such that `sq ≤ i`

```
dp[i] = min(dp[i], dp[i - sq] + 1)
```

### Why this works

* If you pick square `sq`, you need:

  * `1` (for current square)
  * `dp[i - sq]` to build the remaining sum
* Choose the minimum across all possible squares

---

## Order of Computation

* Outer loop: `i = 1 → n`
* Inner loop: all perfect squares `≤ i`

This ensures `dp[i - sq]` is already computed.

---

## Python 3 Implementation (Typed)

```python
from typing import List
import math

class Solution:
    def numSquares(self, n: int) -> int:
        dp: List[int] = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(1, int(math.sqrt(i)) + 1):
                square = j * j
                dp[i] = min(dp[i], dp[i - square] + 1)

        return dp[n]
```

---

## Example Walkthrough: `n = 12`

### Perfect Squares ≤ 12

```
1, 4, 9
```

---

### Step-by-Step DP Table

| i | dp[i] | Explanation |
| --- | --- | --- |
| 0 | 0 | Base case |
| 1 | 1 | 1 |
| 2 | 2 | 1 + 1 |
| 3 | 3 | 1 + 1 + 1 |
| 4 | 1 | 4 |
| 5 | 2 | 4 + 1 |
| 6 | 3 | 4 + 1 + 1 |
| 7 | 4 | 4 + 1 + 1 + 1 |
| 8 | 2 | 4 + 4 |
| 9 | 1 | 9 |
| 10 | 2 | 9 + 1 |
| 11 | 3 | 9 + 1 + 1 |
| 12 | 3 | 4 + 4 + 4 |

---

### DP Transition Example (`i = 12`)

```
dp[12] =
min(
    dp[12 - 1] + 1 = dp[11] + 1 = 4,
    dp[12 - 4] + 1 = dp[8]  + 1 = 3,
    dp[12 - 9] + 1 = dp[3]  + 1 = 4
)
= 3
```

---

## Final DP Table (Index → Value)

```
Index:  0  1  2  3  4  5  6  7  8  9 10 11 12
dp:     0  1  2  3  1  2  3  4  2  1  2  3  3
```

---

## Time & Space Complexity

### Time

```
O(n * sqrt(n))
```

### Space

```
O(n)
```

---

## Pattern Classification

* Dynamic Programming
* Unbounded Knapsack
* Coin Change (Minimum Coins)

---

If you want, I can next provide:

* BFS / graph shortest-path solution
* Mathematical (Lagrange’s Four Square Theorem) approach
* Comparison with Coin Change DP
* Visualization of transitions as a decision tree

State your preference.