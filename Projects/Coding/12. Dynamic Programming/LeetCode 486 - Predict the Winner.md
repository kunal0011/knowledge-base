---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 486: Predict the Winner"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 486: Predict the Winner

Below is a **clean, interview-grade DP explanation** for **LeetCode 486 – Predict the Winner**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 486 – Predict the Winner

### Problem Statement

You are given an integer array `nums`. Two players take turns picking a number from **either end** of the array. Each player plays **optimally**.  
Return `true` if **Player 1** can win (or tie), otherwise return `false`.

---

## Key Observation (Game Theory → DP)

* Both players play **optimally**
* This is a **two-player zero-sum game**
* Instead of tracking absolute scores, track **score difference**

> **Score difference = (current player’s score − opponent’s score)**

If Player 1 can guarantee a **non-negative score difference**, Player 1 wins.

---

## DP State Definition

### State

```
dp[i][j] = maximum score difference the current player can achieve
           from subarray nums[i...j]
```

* `i` = left index
* `j` = right index
* Current player is whoever’s turn it is on subarray `[i, j]`

---

## Base Case

When there is **only one element**:

```
dp[i][i] = nums[i]
```

Reason:

* Current player picks the only number
* Opponent gets nothing

---

## State Transition

From subarray `[i, j]`, the current player has **two choices**:

### 1. Pick left element `nums[i]`

After picking:

* Opponent plays optimally on `[i+1, j]`
* Opponent’s advantage = `dp[i+1][j]`

So net gain:

```
nums[i] - dp[i+1][j]
```

---

### 2. Pick right element `nums[j]`

After picking:

* Opponent plays optimally on `[i, j-1]`

Net gain:

```
nums[j] - dp[i][j-1]
```

---

### Transition Formula

```
dp[i][j] = max(
    nums[i] - dp[i+1][j],
    nums[j] - dp[i][j-1]
)
```

This captures **optimal play on both sides**.

---

## DP Table Construction Order

* We need `dp[i+1][j]` and `dp[i][j-1]`
* So fill table by **increasing subarray length**

### Order

1. Length = 1 (base case)
2. Length = 2
3. ...
4. Length = n

---

## Final Decision

After filling the table:

```
return dp[0][n-1] >= 0
```

* `>= 0` → Player 1 can win or tie
* `< 0` → Player 1 loses

---

## Example Walkthrough

### Input

```text
nums = [1, 5, 2]
```

---

### Step 1: Initialize DP Table

`dp[i][i] = nums[i]`

| i\j | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 1 |  |  |
| 1 |  | 5 |  |
| 2 |  |  | 2 |

---

### Step 2: Length = 2

#### Subarray [0,1]

```
dp[0][1] = max(
    1 - dp[1][1],   // 1 - 5 = -4
    5 - dp[0][0]    // 5 - 1 = 4
) = 4
```

#### Subarray [1,2]

```
dp[1][2] = max(
    5 - dp[2][2],   // 5 - 2 = 3
    2 - dp[1][1]    // 2 - 5 = -3
) = 3
```

| i\j | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 1 | 4 |  |
| 1 |  | 5 | 3 |
| 2 |  |  | 2 |

---

### Step 3: Length = 3

#### Subarray [0,2]

```
dp[0][2] = max(
    1 - dp[1][2],   // 1 - 3 = -2
    2 - dp[0][1]    // 2 - 4 = -2
) = -2
```

| i\j | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | 1 | 4 | -2 |
| 1 |  | 5 | 3 |
| 2 |  |  | 2 |

---

### Step 4: Final Result

```
dp[0][2] = -2 < 0
```

➡ **Player 1 loses**

Return:

```
false
```

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [[0] * n for _ in range(n)]

        # Base case
        for i in range(n):
            dp[i][i] = nums[i]

        # Fill DP table by increasing length
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = max(
                    nums[i] - dp[i + 1][j],
                    nums[j] - dp[i][j - 1]
                )

        return dp[0][n - 1] >= 0
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n²)** |
| Space Complexity | **O(n²)** |
| DP Type | Interval DP / Game DP |

---

## Why This DP Works

* Converts a **two-player optimal game** into a **single-player maximization**
* Tracks **relative advantage**, not absolute scores
* Cleanly models **minimax behavior** using subtraction

---

If you want next:

* **Space-optimized O(n) DP**
* **Minimax recursion vs DP comparison**
* **Backtracking game tree visualization**
* **Similar problems (877, 1140, 1406)**

Tell me which direction to continue.