---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 119: Pascal’s Triangle II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 119: Pascal’s Triangle II

**LeetCode 119 – Pascal’s Triangle II**, focusing strictly on **state definition, transition, DP table construction, and optimization**, with a worked example.

---

## LeetCode 119: Pascal’s Triangle II

### Problem Statement

Given an integer `rowIndex` (0-indexed), return the `rowIndex`-th row of Pascal’s Triangle.

**Pascal’s Triangle rule**

```
triangle[i][j] = triangle[i-1][j-1] + triangle[i-1][j]
```

with boundary values always equal to `1`.

---

## Dynamic Programming Formulation

### 1. DP State Definition

Let:

```
dp[i][j] = value at row i and column j of Pascal’s Triangle
```

Where:

* `0 ≤ j ≤ i`
* Rows are **0-indexed**

---

### 2. Base Case

For every row `i`:

```
dp[i][0] = 1
dp[i][i] = 1
```

First and last elements of every row are `1`.

---

### 3. State Transition

For internal elements:

```
dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
```

This directly follows Pascal’s Triangle definition.

---

### 4. DP Table Construction (Conceptual)

To compute row `k`, we only need rows `0 → k`.

#### Example: `rowIndex = 4`

| i (row) | dp[i] |
| --- | --- |
| 0 | [1] |
| 1 | [1, 1] |
| 2 | [1, 2, 1] |
| 3 | [1, 3, 3, 1] |
| 4 | [1, 4, 6, 4, 1] |

Final Answer:

```
[1, 4, 6, 4, 1]
```

---

## Space Optimization (Key Insight)

### Observation

To compute row `i`, we only need row `i-1`.

So instead of a **2D DP table**, we can use a **1D DP array** and update it **right-to-left**.

This avoids overwriting values that are still needed.

---

## Optimized DP State (1D)

```
dp[j] = value at current row and column j
```

### Transition (Right-to-Left Update)

```
dp[j] = dp[j] + dp[j-1]
```

---

## Step-by-Step Example (rowIndex = 4)

Initialize:

```
dp = [1]
```

Row 1:

```
dp = [1, 1]
```

Row 2:

```
dp = [1, 2, 1]
```

Row 3:

```
dp = [1, 3, 3, 1]
```

Row 4:

```
dp = [1, 4, 6, 4, 1]
```

---

## Python 3 Solution (DP, Optimized)

```python
from typing import List

class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [1]

        for i in range(1, rowIndex + 1):
            dp.append(1)  # last element is always 1
            for j in range(i - 1, 0, -1):
                dp[j] = dp[j] + dp[j - 1]

        return dp
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n²)** |
| Space | **O(n)** |

---

## Why Right-to-Left Update Is Mandatory

If we update **left-to-right**, `dp[j-1]` would already be modified for the current row, corrupting the computation.

Right-to-left preserves the **previous row values** until they are used.

---

## Summary

* **DP State**: `dp[i][j]`
* **Transition**: `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`
* **Optimization**: Use 1D DP updated from right to left
* **Final Output**: The last constructed row

If you want, I can also provide:

* Mathematical (nCr) solution
* Comparison between 2D vs 1D DP
* Visualization of in-place overwrite behavior

Just tell me.