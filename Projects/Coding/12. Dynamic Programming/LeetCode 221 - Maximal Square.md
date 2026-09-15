---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 221: Maximal Square"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 221: Maximal Square

**LeetCode 221 – Maximal Square**, focused on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 221 – Maximal Square

### Problem Statement

You are given a `rows × cols` binary matrix filled with `'0'` and `'1'`.  
Find the **largest square containing only `'1'`** and return its **area**.

---

## Key Observation (Why DP Works)

A square of size `k × k` **ending at a cell (i, j)** exists **iff**:

* The cell itself is `'1'`
* The **top**, **left**, and **top-left diagonal** neighbors can all form squares of at least size `k-1`

Thus, the largest square at `(i, j)` depends on **three previously solved subproblems**.

---

## DP State Definition

Let:

```
dp[i][j] = side length of the largest square
           whose bottom-right corner is at (i, j)
```

Important:

* We track **side length**, not area
* Final answer = `(max side length)²`

---

## DP Transition (Core Formula)

If `matrix[i][j] == '1'`:

```
dp[i][j] = 1 + min(
    dp[i-1][j],     # top
    dp[i][j-1],     # left
    dp[i-1][j-1]    # top-left diagonal
)
```

If `matrix[i][j] == '0'`:

```
dp[i][j] = 0
```

### Why `min(...)`?

To form a larger square, **all three directions must support it**.  
The smallest neighboring square limits expansion.

---

## Base Case

* First row or first column:

  * If `matrix[i][j] == '1'`, then `dp[i][j] = 1`
  * Because no room to expand further

---

## Example Walkthrough

### Input Matrix

```
matrix =
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
```

---

### DP Table Construction (step-by-step)

We build `dp` row by row.

```
dp =
[
  [1, 0, 1, 0, 0],
  [1, 0, 1, 1, 1],
  [1, 1, 1, 2, 2],
  [1, 0, 0, 1, 0]
]
```

### Explanation of a Key Cell

At position `(2,3)` (value `'1'`):

```
top        = dp[1][3] = 1
left       = dp[2][2] = 1
top-left   = dp[1][2] = 1

dp[2][3] = 1 + min(1,1,1) = 2
```

This means a **2×2 square** ends here.

---

## Final Result

* Maximum side length = `2`
* Area = `2 × 2 = 4`

---

## Python 3 (Typed, Clean DP)

```python
from typing import List

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        rows, cols = len(matrix), len(matrix[0])
        dp = [[0] * cols for _ in range(rows)]
        max_side = 0

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == "1":
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = 1 + min(
                            dp[i-1][j],
                            dp[i][j-1],
                            dp[i-1][j-1]
                        )
                    max_side = max(max_side, dp[i][j])

        return max_side * max_side
```

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(rows × cols)** |
| Space | **O(rows × cols)** |
| Optimizable Space | **O(cols)** using rolling array |

---

## DP Pattern Recognition

This problem is a classic example of:

* **2D DP**
* **DP on matrix**
* **Square expansion using neighbors**
* Similar to histogram / largest square problems

---

If you want next:

* **Space-optimized version**
* **Why greedy fails**
* **Visual DP animation**
* **Comparison with LeetCode 1277 (count squares)**

Tell me how deep you want to go.