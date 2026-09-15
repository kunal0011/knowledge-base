---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 85: Maximal Rectangle"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 85: Maximal Rectangle

**LeetCode 85 (Maximal Rectangle)** with **explicit state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 85 — Maximal Rectangle

### Problem Statement

Given a binary matrix filled with `'0'` and `'1'`, find the **largest rectangle containing only `1`s** and return its area.

---

## Core DP Insight

A maximal rectangle in a 2D binary matrix can be reduced to a sequence of **Largest Rectangle in Histogram** problems by processing **row by row**.

Each row acts as the **base of a histogram**, where column heights represent **consecutive 1s above (including current row)**.

---

## DP State Definition

We define a **1D DP array per row**:

```
dp[j] = number of consecutive '1's in column j ending at current row
```

* This DP is updated row by row
* For each row, `dp` forms a histogram
* We compute the **largest rectangle in this histogram**

---

## DP Transition

For each cell `(i, j)`:

```
if matrix[i][j] == '1':
    dp[j] = dp[j] + 1
else:
    dp[j] = 0
```

This transition is **pure DP accumulation**.

---

## Overall Algorithm (DP + Stack)

1. Initialize `dp` with zeros (size = number of columns)
2. For each row:

   * Update `dp` using the transition above
   * Compute **largest rectangle area in histogram `dp`**
3. Return maximum area found

---

## Why This Is DP (Not Greedy)

* `dp[j]` **depends on the previous row's dp[j]**
* Each row reuses prior computation
* Histogram area calculation uses DP values as input

---

## Largest Rectangle in Histogram (Per Row)

We use a **monotonic increasing stack**.

### Idea

For each bar:

* Find nearest **smaller bar on left**
* Find nearest **smaller bar on right**
* Compute width × height

---

## Python 3 Solution (Typed)

```python
from typing import List

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0

        cols = len(matrix[0])
        dp = [0] * cols
        max_area = 0

        for row in matrix:
            # DP transition
            for j in range(cols):
                dp[j] = dp[j] + 1 if row[j] == '1' else 0

            # Largest Rectangle in Histogram
            stack = []
            for i in range(cols + 1):
                cur_height = dp[i] if i < cols else 0
                while stack and cur_height < dp[stack[-1]]:
                    h = dp[stack.pop()]
                    w = i if not stack else i - stack[-1] - 1
                    max_area = max(max_area, h * w)
                stack.append(i)

        return max_area
```

---

## DP Table Construction (Worked Example)

### Input

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

### Row 0

```
dp = [1,0,1,0,0]
Max area = 1
```

---

### Row 1

```
dp = [2,0,2,1,1]
Histogram max = 3  (columns 2–4 with height 1)
```

---

### Row 2

```
dp = [3,1,3,2,2]
Histogram max = 6
Rectangle:
height = 2
width  = 3
```

---

### Row 3

```
dp = [4,0,0,3,0]
Histogram max = 4
```

---

### Final Answer

```
Maximum Rectangle Area = 6
```

---

## Visual Interpretation of DP

Each row builds **vertical height DP**:

```
Row 2 dp: [3,1,3,2,2]

Histogram:
|
|      █ █
| █    █ █
| █ █  █ █
+----------------
  0 1  2 3 4
```

Largest rectangle spans columns `[2..4]`.

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | `O(R * C)` |
| Space | `O(C)` |

---

## Key Takeaways

1. **DP State**: vertical height accumulation
2. **Transition**: depends on previous row
3. **2D → 1D reduction** is the core trick
4. Histogram stack is **deterministic**, not greedy

---

If you want, I can also explain:

* **Pure DP with left/right boundaries**
* **Why brute force fails**
* **Comparison with LeetCode 84**
* **Backtracking vs DP contrast**

Just specify.