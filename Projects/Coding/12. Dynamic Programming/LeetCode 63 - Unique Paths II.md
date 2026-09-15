---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 63: Unique Paths II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 63: Unique Paths II

**LeetCode 63 – Unique Paths II**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 63 – Unique Paths II

### Problem Statement

You are given an `m x n` grid where:

* `0` represents an empty cell
* `1` represents an obstacle

A robot starts at the **top-left corner** `(0,0)` and wants to reach the **bottom-right corner** `(m-1,n-1)`.

The robot can move **only right or down**.

Return the **number of unique paths** that avoid obstacles.

---

## Key Observation

* If a cell contains an **obstacle**, it **cannot be part of any path**.
* Otherwise, the number of ways to reach a cell depends on:

  * the number of ways to reach the **cell above**
  * the number of ways to reach the **cell to the left**

This is a classic **2D Dynamic Programming grid problem**.

---

## DP State Definition

Let:

```
dp[i][j] = number of unique paths to reach cell (i, j)
```

---

## DP Transition

For a normal cell (`obstacleGrid[i][j] == 0`):

```
dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

For an obstacle (`obstacleGrid[i][j] == 1`):

```
dp[i][j] = 0
```

---

## Base Conditions

1. **Start Cell**

   ```
   dp[0][0] = 1   if obstacleGrid[0][0] == 0
   dp[0][0] = 0   if obstacleGrid[0][0] == 1
   ```
2. **First Row**

   * Only reachable from the **left**
   * If an obstacle appears, all cells after it become unreachable
3. **First Column**

   * Only reachable from **above**
   * Same obstacle rule applies

---

## Example Walkthrough

### Input Grid

```
obstacleGrid =
[
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 0]
]
```

---

### Step 1: Initialize DP Table

```
dp =
[
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]
]
```

---

### Step 2: Fill DP Table Cell by Cell

#### (0,0)

```
dp[0][0] = 1   (start cell, no obstacle)
```

```
[1, 0, 0]
[0, 0, 0]
[0, 0, 0]
```

---

#### First Row

* (0,1): no obstacle → from left

```
dp[0][1] = dp[0][0] = 1
```

* (0,2): no obstacle → from left

```
dp[0][2] = dp[0][1] = 1
```

```
[1, 1, 1]
[0, 0, 0]
[0, 0, 0]
```

---

#### Second Row

* (1,0): no obstacle → from top

```
dp[1][0] = dp[0][0] = 1
```

* (1,1): obstacle

```
dp[1][1] = 0
```

* (1,2): no obstacle

```
dp[1][2] = dp[0][2] + dp[1][1]
         = 1 + 0
         = 1
```

```
[1, 1, 1]
[1, 0, 1]
[0, 0, 0]
```

---

#### Third Row

* (2,0): from top

```
dp[2][0] = dp[1][0] = 1
```

* (2,1): no obstacle

```
dp[2][1] = dp[1][1] + dp[2][0]
         = 0 + 1
         = 1
```

* (2,2): destination

```
dp[2][2] = dp[1][2] + dp[2][1]
         = 1 + 1
         = 2
```

```
[1, 1, 1]
[1, 0, 1]
[1, 1, 2]
```

---

### Final Answer

```
2 unique paths
```

---

## Python 3 DP Solution (with typing)

```python
from typing import List

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        
        # If starting cell is blocked
        if obstacleGrid[0][0] == 1:
            return 0
        
        dp = [[0] * n for _ in range(m)]
        dp[0][0] = 1
        
        # First column
        for i in range(1, m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = dp[i - 1][0]
        
        # First row
        for j in range(1, n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = dp[0][j - 1]
        
        # Fill rest of dp table
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[m - 1][n - 1]
```

---

## Time & Space Complexity

* **Time:** `O(m × n)`
* **Space:** `O(m × n)`  
  (Can be optimized to `O(n)` using a 1D DP array)

---

## Pattern Recognition (Interview Insight)

* **Category:** Grid DP
* **State:** `dp[i][j]`
* **Transition:** top + left
* **Obstacle handling:** zero out the state

If you want, I can also:

* Convert this to **1D DP**
* Compare it with **LeetCode 62**
* Draw a **DP dependency graph**
* Show **edge-case traps interviewers test**

Just say which one.