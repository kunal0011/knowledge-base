---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 62: Unique Paths"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 62: Unique Paths

Below is a **precise, DP-first explanation** of **LeetCode 62 – Unique Paths**, focusing on **state definition, transition, DP table construction, and a concrete example**.

---

## LeetCode 62 – Unique Paths

### Problem Statement

You are given an `m x n` grid.  
A robot starts at the **top-left corner** `(0, 0)` and wants to reach the **bottom-right corner** `(m-1, n-1)`.

Constraints:

* The robot can move **only right or down**.
* Count the **total number of unique paths**.

---

## Key Observation

At any cell `(i, j)`, the robot can arrive **only from**:

1. The **top cell** `(i-1, j)`
2. The **left cell** `(i, j-1)`

There are no other possibilities.

---

## DP State Definition

Let:

```
dp[i][j] = number of unique paths to reach cell (i, j)
```

---

## Base Cases

1. **Start cell**

```
dp[0][0] = 1
```

2. **First row (`i = 0`)**

* Can only move **right**

```
dp[0][j] = 1
```

3. **First column (`j = 0`)**

* Can only move **down**

```
dp[i][0] = 1
```

---

## State Transition

For all other cells:

```
dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
```

Explanation:

* Paths coming from **above**
* Paths coming from **left**

---

## DP Table Creation (Bottom-Up)

We fill the DP table **row by row**, **left to right**, ensuring required states are already computed.

---

## Example Walkthrough

### Input

```
m = 3, n = 4
```

Grid indices:

```
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
```

---

### Step 1: Initialize DP Table

```
dp = [
 [1, 1, 1, 1],
 [1, 0, 0, 0],
 [1, 0, 0, 0]
]
```

First row and first column are all `1`.

---

### Step 2: Fill Remaining Cells

#### Cell (1,1)

```
dp[1][1] = dp[0][1] + dp[1][0]
         = 1 + 1
         = 2
```

#### Cell (1,2)

```
dp[1][2] = dp[0][2] + dp[1][1]
         = 1 + 2
         = 3
```

#### Cell (1,3)

```
dp[1][3] = dp[0][3] + dp[1][2]
         = 1 + 3
         = 4
```

---

#### Cell (2,1)

```
dp[2][1] = dp[1][1] + dp[2][0]
         = 2 + 1
         = 3
```

#### Cell (2,2)

```
dp[2][2] = dp[1][2] + dp[2][1]
         = 3 + 3
         = 6
```

#### Cell (2,3)

```
dp[2][3] = dp[1][3] + dp[2][2]
         = 4 + 6
         = 10
```

---

### Final DP Table

```
dp = [
 [1,  1,  1,  1],
 [1,  2,  3,  4],
 [1,  3,  6, 10]
]
```

---

## Final Answer

```
dp[m - 1][n - 1] = dp[2][3] = 10
```

There are **10 unique paths**.

---

## Python 3 Implementation (Typed)

```python
from typing import List

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp: List[List[int]] = [[0] * n for _ in range(m)]

        # Base cases
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1

        # Fill dp table
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]
```

---

## Time & Space Complexity

* **Time:** `O(m × n)`
* **Space:** `O(m × n)`

---

If you want next:

* Space-optimized `O(n)` DP
* Backtracking vs DP comparison
* Mathematical (combinatorics) solution
* DP table visualization as a tree or path graph

Tell me which one you want next.