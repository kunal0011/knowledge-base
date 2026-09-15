---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 542: 01 Matrix"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 542: 01 Matrix

**LeetCode 542 – 01 Matrix**, focusing explicitly on **state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 542 – 01 Matrix

### Problem Statement

You are given an `m x n` binary matrix `mat` where:

* `mat[i][j] == 0` or `1`
* For each cell containing `1`, compute the **distance to the nearest `0`**
* Distance is **Manhattan distance** (up, down, left, right)

Return a matrix of the same size with these distances.

---

## Key Observation (DP Insight)

For any cell `(i, j)`:

* If `mat[i][j] == 0`, distance = `0`
* If `mat[i][j] == 1`, its distance depends on **neighboring cells**
* The distance is:

```
1 + min(distance of top, bottom, left, right neighbor)
```

However, **a single DP pass is insufficient**, because some neighbors may not yet have correct values.

Hence:

> **We use a 2-pass DP approach to propagate distances from all directions.**

---

## DP State Definition

Let:

```
dp[i][j] = minimum distance from cell (i, j) to the nearest 0
```

---

## Initialization (DP Table Creation)

* Create `dp` matrix of size `m x n`
* Initialize:

  * `dp[i][j] = 0` if `mat[i][j] == 0`
  * `dp[i][j] = INF` (a very large number) if `mat[i][j] == 1`

This reflects:

* Known distances for `0`
* Unknown distances for `1`

---

## DP Transitions

### Pass 1: Top-Left → Bottom-Right

This pass accounts for **top and left neighbors**.

For each cell `(i, j)`:

```
dp[i][j] = min(
    dp[i][j],
    dp[i-1][j] + 1   (if i > 0),
    dp[i][j-1] + 1   (if j > 0)
)
```

### Pass 2: Bottom-Right → Top-Left

This pass accounts for **bottom and right neighbors**.

For each cell `(i, j)`:

```
dp[i][j] = min(
    dp[i][j],
    dp[i+1][j] + 1   (if i < m-1),
    dp[i][j+1] + 1   (if j < n-1)
)
```

---

## Why Two Passes Are Required

* First pass propagates distances **from top-left**
* Second pass propagates distances **from bottom-right**
* Together, they cover **all four directions**

This ensures every cell sees the closest `0`, regardless of its location.

---

## Example Walkthrough

### Input

```
mat =
[
  [0, 0, 0],
  [0, 1, 0],
  [1, 1, 1]
]
```

---

### Step 1: Initialize DP

```
dp =
[
  [0, 0, 0],
  [0, ∞, 0],
  [∞, ∞, ∞]
]
```

---

### Step 2: First Pass (Top-Left → Bottom-Right)

```
dp =
[
  [0, 0, 0],
  [0, 1, 0],
  [1, 2, 1]
]
```

Explanation:

* `(1,1)` → min(top=0, left=0) + 1 = 1
* `(2,0)` → from top `(1,0)` → 1
* `(2,1)` → from left `(2,0)` → 2
* `(2,2)` → from top `(1,2)` → 1

---

### Step 3: Second Pass (Bottom-Right → Top-Left)

```
dp =
[
  [0, 0, 0],
  [0, 1, 0],
  [1, 2, 1]
]
```

No changes required because optimal distances are already found.

---

### Final Output

```
[
  [0, 0, 0],
  [0, 1, 0],
  [1, 2, 1]
]
```

---

## Python 3 DP Solution (With Typing)

```python
from typing import List

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        INF = 10**9

        # DP table initialization
        dp = [[INF] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    dp[i][j] = 0

        # First pass: top-left → bottom-right
        for i in range(m):
            for j in range(n):
                if i > 0:
                    dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
                if j > 0:
                    dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)

        # Second pass: bottom-right → top-left
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i < m - 1:
                    dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
                if j < n - 1:
                    dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)

        return dp
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(m × n)** |
| Space Complexity | **O(m × n)** |
| Technique | Dynamic Programming (2-pass) |

---

## When to Prefer This DP Approach

* When grid size is large
* When BFS queue overhead is undesirable
* When you want deterministic directional propagation

If you want, I can also:

* Compare **DP vs Multi-Source BFS**
* Show **why single-pass DP fails**
* Draw a **DP dependency graph**
* Convert this into **space-optimized in-place DP**

State your preference.