---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 120: Triangle"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 120: Triangle

**LeetCode 120 – Triangle**, with **explicit state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 120 – Triangle

### Problem Statement

You are given a triangle array. Starting from the top, you may move to **adjacent numbers** on the row below.

Return the **minimum path sum** from top to bottom.

**Adjacent rule**:  
From `triangle[i][j]`, you may move to:

* `triangle[i+1][j]`
* `triangle[i+1][j+1]`

---

## Key Observation

At every cell `(i, j)`:

* The minimum path to reach this cell depends **only on the minimum paths to its parents** from the previous row.
* This naturally fits **Dynamic Programming on a triangle grid**.

---

## DP State Definition

Let:

```
dp[i][j] = minimum path sum to reach cell (i, j)
```

Where:

* `i` → row index (0-based)
* `j` → column index (0 ≤ j ≤ i)

---

## Base Case

Top of the triangle:

```
dp[0][0] = triangle[0][0]
```

---

## State Transition

For row `i > 0`:

### Case 1: Left edge (`j == 0`)

Can only come from directly above:

```
dp[i][0] = dp[i-1][0] + triangle[i][0]
```

---

### Case 2: Right edge (`j == i`)

Can only come from top-left:

```
dp[i][i] = dp[i-1][i-1] + triangle[i][i]
```

---

### Case 3: Middle cells (`0 < j < i`)

Can come from **two parents**:

```
dp[i][j] = min(
    dp[i-1][j-1],
    dp[i-1][j]
) + triangle[i][j]
```

---

## Final Answer

The path may end at **any cell in the last row**, so:

```
answer = min(dp[n-1])
```

---

## DP Table Construction (Example)

### Input Triangle

```
[
 [2],
 [3, 4],
 [6, 5, 7],
 [4, 1, 8, 3]
]
```

---

### Step-by-Step DP Table

Initialize DP table with same structure:

```
dp = [
 [2],
 [0, 0],
 [0, 0, 0],
 [0, 0, 0, 0]
]
```

---

### Row 1 (`i = 1`)

```
dp[1][0] = 2 + 3 = 5
dp[1][1] = 2 + 4 = 6
```

```
[
 [2],
 [5, 6],
 [0, 0, 0],
 [0, 0, 0, 0]
]
```

---

### Row 2 (`i = 2`)

```
dp[2][0] = 5 + 6 = 11
dp[2][1] = min(5, 6) + 5 = 10
dp[2][2] = 6 + 7 = 13
```

```
[
 [2],
 [5, 6],
 [11, 10, 13],
 [0, 0, 0, 0]
]
```

---

### Row 3 (`i = 3`)

```
dp[3][0] = 11 + 4 = 15
dp[3][1] = min(11, 10) + 1 = 11
dp[3][2] = min(10, 13) + 8 = 18
dp[3][3] = 13 + 3 = 16
```

```
[
 [2],
 [5, 6],
 [11, 10, 13],
 [15, 11, 18, 16]
]
```

---

## Final Answer

```
min(dp[3]) = min(15, 11, 18, 16) = 11
```

### Minimum Path:

```
2 → 3 → 5 → 1 = 11
```

---

## Python 3 DP Solution (With Typing)

```python
from typing import List

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0] * (i + 1) for i in range(n)]

        dp[0][0] = triangle[0][0]

        for i in range(1, n):
            for j in range(i + 1):
                if j == 0:
                    dp[i][j] = dp[i - 1][j] + triangle[i][j]
                elif j == i:
                    dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j - 1],
                        dp[i - 1][j]
                    ) + triangle[i][j]

        return min(dp[-1])
```

---

## Complexity Analysis

* **Time Complexity:** `O(n²)`
* **Space Complexity:** `O(n²)`

  * Can be optimized to `O(n)` using bottom-up 1D DP if required.

---

If you want, I can next provide:

* Space-optimized 1D DP version
* Bottom-up (reverse traversal) explanation
* Recursion → Memoization → DP transformation  
  Just tell me which one you want.