---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 375: Guess Number Higher or Lower II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 375: Guess Number Higher or Lower II

**LeetCode 375 – Guess Number Higher or Lower II**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 375 – Guess Number Higher or Lower II

### Problem Statement

You are playing a guessing game with numbers from **1 to n**.

* Every time you guess a number **x**:

  * If it is **wrong**, you pay **x dollars**
  * You are told whether the correct number is **higher or lower**

Your goal is to **guarantee a win** while **minimizing the maximum amount of money** you might have to pay.

Return the **minimum money required to guarantee a win**.

---

## Key Insight (Why DP?)

This is a **minimax** problem:

* You choose a number `x`
* The adversary (worst case) forces you into the **more expensive side**
* You want to minimize the **maximum loss**

Hence:

> **Minimize over guesses, maximize over outcomes**

---

## DP State Definition

Let:

```
dp[l][r] = minimum money required to guarantee a win
           if the secret number is in range [l, r]
```

### Base Case

* If `l >= r` → only one or zero numbers

```
dp[l][r] = 0
```

No cost because you already know the number.

---

## DP Transition (Core Logic)

If you guess number `x` (where `l <= x <= r`):

* You pay `x`
* Worst-case future cost:

  ```
  max(dp[l][x-1], dp[x+1][r])
  ```

So total cost for guessing `x`:

```
cost(x) = x + max(dp[l][x-1], dp[x+1][r])
```

You choose `x` that **minimizes** this cost:

```
dp[l][r] = min over x in [l..r] of
           ( x + max(dp[l][x-1], dp[x+1][r]) )
```

---

## Order of DP Computation

* Smaller ranges must be solved first
* Use **interval DP**
* Increase range length from `2 → n`

---

## Example Walkthrough (n = 4)

### Step 1: Initialize DP Table

`dp[i][i] = 0`

| l\ r | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | 0 |  |  |  |
| 2 |  | 0 |  |  |
| 3 |  |  | 0 |  |
| 4 |  |  |  | 0 |

---

### Step 2: Length = 2

#### dp[1][2]

* Guess 1 → cost = `1 + dp[2][2] = 1`
* Guess 2 → cost = `2 + dp[1][1] = 2`

```
dp[1][2] = 1
```

#### dp[2][3] = 2

#### dp[3][4] = 3

---

### Step 3: Length = 3

#### dp[1][3]

* Guess 1 → `1 + dp[2][3] = 3`
* Guess 2 → `2 + max(dp[1][1], dp[3][3]) = 2`
* Guess 3 → `3 + dp[1][2] = 4`

```
dp[1][3] = 2
```

#### dp[2][4]

* Guess 2 → `2 + dp[3][4] = 5`
* Guess 3 → `3 + max(dp[2][2], dp[4][4]) = 3`
* Guess 4 → `4 + dp[2][3] = 6`

```
dp[2][4] = 3
```

---

### Step 4: Length = 4

#### dp[1][4]

* Guess 1 → `1 + dp[2][4] = 4`
* Guess 2 → `2 + max(dp[1][1], dp[3][4]) = 5`
* Guess 3 → `3 + max(dp[1][2], dp[4][4]) = 4`
* Guess 4 → `4 + dp[1][3] = 6`

```
dp[1][4] = 4
```

---

### Final DP Table

| l\ r | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| 1 | 0 | 1 | 2 | 4 |
| 2 |  | 0 | 2 | 3 |
| 3 |  |  | 0 | 3 |
| 4 |  |  |  | 0 |

### Answer

```
dp[1][n] = dp[1][4] = 4
```

---

## Python 3 DP Solution (with typing)

```python
from typing import List

class Solution:
    def getMoneyAmount(self, n: int) -> int:
        dp: List[List[int]] = [[0] * (n + 2) for _ in range(n + 2)]

        for length in range(2, n + 1):
            for l in range(1, n - length + 2):
                r = l + length - 1
                dp[l][r] = float('inf')

                for x in range(l, r + 1):
                    cost = x + max(dp[l][x - 1], dp[x + 1][r])
                    dp[l][r] = min(dp[l][r], cost)

        return dp[1][n]
```

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(n³)` |
| Space | `O(n²)` |

---

## Mental Model to Remember

* Interval DP
* Guess `x`
* Pay `x`
* Adversary chooses worse side
* Minimize the worst loss

If you want, I can next:

* Optimize explanation using **decision tree intuition**
* Compare with **binary search**
* Show **why greedy fails**
* Visualize the **game tree for n = 5**