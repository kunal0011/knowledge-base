---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 64: Minimum Path Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 64: Minimum Path Sum

**LeetCode 64 – Minimum Path Sum**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 64 – Minimum Path Sum

### Problem Statement

You are given an `m x n` grid filled with **non-negative integers**.  
Starting from the **top-left cell (0,0)**, you want to reach the **bottom-right cell (m-1,n-1)**.

**Constraints**

* You may only move **right** or **down** at any point.
* The **path sum** is the sum of all numbers along the path.

Return the **minimum path sum**.

---

## Key Observation

To reach cell `(i, j)`, you can only come from:

* the **top** cell `(i-1, j)`
* the **left** cell `(i, j-1)`

This makes the problem **optimal substructure + overlapping subproblems**, which is ideal for **Dynamic Programming**.

---

## DP State Definition

Let:

```
dp[i][j] = minimum path sum to reach cell (i, j) from (0, 0)
```

This state fully captures the problem because:

* Once you know the minimum cost to reach `(i, j)`,
* future decisions do not depend on how you arrived there.

---

## State Transition

For any cell `(i, j)`:

```
dp[i][j] = grid[i][j] + min(
    dp[i-1][j],   # from top
    dp[i][j-1]    # from left
)
```

### Boundary Conditions

* If `i == 0` (first row), you **can only come from left**
* If `j == 0` (first column), you **can only come from top**

---

## Base Case

```
dp[0][0] = grid[0][0]
```

This is the starting cell.

---

## DP Table Construction (Worked Example)

### Input Grid

```
grid =
[
  [1, 3, 1],
  [1, 5, 1],
  [4, 2, 1]
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

### Step 2: Fill the DP Table

#### Cell (0,0)

```
dp[0][0] = 1
```

#### First Row (only from left)

```
dp[0][1] = 1 + 3 = 4
dp[0][2] = 4 + 1 = 5
```

#### First Column (only from top)

```
dp[1][0] = 1 + 1 = 2
dp[2][0] = 2 + 4 = 6
```

---

#### Remaining Cells

```
dp[1][1] = 5 + min(4, 2) = 7
dp[1][2] = 1 + min(5, 7) = 6
dp[2][1] = 2 + min(7, 6) = 8
dp[2][2] = 1 + min(6, 8) = 7
```

---

### Final DP Table

```
dp =
[
  [1, 4, 5],
  [2, 7, 6],
  [6, 8, 7]
]
```

---

## Final Answer

```
Minimum Path Sum = dp[m-1][n-1] = 7
```

---

## Python 3 DP Solution (With Typing)

```python
from typing import List

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        
        dp = [[0] * n for _ in range(m)]
        
        # Base case
        dp[0][0] = grid[0][0]
        
        # First row
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        
        # First column
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        
        # Fill the rest
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = grid[i][j] + min(
                    dp[i - 1][j],
                    dp[i][j - 1]
                )
        
        return dp[m - 1][n - 1]
```

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | `O(m × n)` |
| Space Complexity | `O(m × n)` |
| Optimizable | Yes → `O(n)` using 1D DP |

---

## DP Pattern Classification

* **Grid DP**
* **2D DP with directional constraints**
* Foundation for problems like:

  * Unique Paths
  * Dungeon Game
  * Cherry Pickup

---

If you want, I can also provide:

* **1D space-optimized DP**
* **Recursive + memoized version**
* **Path reconstruction**
* **Comparison with BFS / Dijkstra**
* **DP tree / dependency graph**

Just tell me.