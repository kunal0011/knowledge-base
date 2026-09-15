---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 118: Pascal’s Triangle"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 118: Pascal’s Triangle

**LeetCode 118 – Pascal’s Triangle**, with **clear state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 118 – Pascal’s Triangle

### Problem Statement

Given an integer `numRows`, return the first `numRows` of Pascal’s Triangle.

**Pascal’s Triangle rules**

* First and last element of every row is `1`
* Any inner element is the **sum of the two elements directly above it**

---

## Why this is a Dynamic Programming problem

Each value depends on **previously computed values** (from the row above).  
Hence:

* **Overlapping subproblems**
* **Optimal substructure**

---

## DP Formulation

### 1. State Definition

Let

```
dp[i][j] = value at row i and column j (0-indexed)
```

Where:

* `0 ≤ i < numRows`
* `0 ≤ j ≤ i`

---

### 2. Base Case

```
dp[i][0] = 1
dp[i][i] = 1
```

First and last elements of every row are always `1`.

---

### 3. State Transition

For inner elements:

```
dp[i][j] = dp[i-1][j-1] + dp[i-1][j]
```

This follows directly from Pascal’s Triangle definition.

---

### 4. DP Table Size

The DP table is **triangular**:

```
Row 0 → 1 element
Row 1 → 2 elements
Row 2 → 3 elements
...
Row n-1 → n elements
```

Total elements ≈ `numRows * (numRows + 1) / 2`

---

## Example Walkthrough (numRows = 5)

### Step-by-step DP table construction

Initialize empty DP table:

```
dp = []
```

---

### Row 0

```
dp[0] = [1]
```

---

### Row 1

```
dp[1][0] = 1
dp[1][1] = 1

dp[1] = [1, 1]
```

---

### Row 2

```
dp[2][0] = 1
dp[2][1] = dp[1][0] + dp[1][1] = 1 + 1 = 2
dp[2][2] = 1

dp[2] = [1, 2, 1]
```

---

### Row 3

```
dp[3][0] = 1
dp[3][1] = dp[2][0] + dp[2][1] = 1 + 2 = 3
dp[3][2] = dp[2][1] + dp[2][2] = 2 + 1 = 3
dp[3][3] = 1

dp[3] = [1, 3, 3, 1]
```

---

### Row 4

```
dp[4][0] = 1
dp[4][1] = 1 + 3 = 4
dp[4][2] = 3 + 3 = 6
dp[4][3] = 3 + 1 = 4
dp[4][4] = 1

dp[4] = [1, 4, 6, 4, 1]
```

---

### Final DP Table

```
[
 [1],
 [1, 1],
 [1, 2, 1],
 [1, 3, 3, 1],
 [1, 4, 6, 4, 1]
]
```

---

## Python 3 DP Implementation (with typing)

```python
from typing import List

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        dp: List[List[int]] = []

        for i in range(numRows):
            # initialize row with 1s
            row = [1] * (i + 1)

            # fill inner elements using DP transition
            for j in range(1, i):
                row[j] = dp[i - 1][j - 1] + dp[i - 1][j]

            dp.append(row)

        return dp
```

---

## Time & Space Complexity

### Time Complexity

```
O(numRows²)
```

Each row computes up to `i` elements.

### Space Complexity

```
O(numRows²)
```

Full DP table stored (required by problem output).

---

## Key Takeaways

* **State:** `dp[i][j]`
* **Transition:** `dp[i-1][j-1] + dp[i-1][j]`
* **Edges:** Always `1`
* Classic example of **bottom-up DP with table construction**

---

If you want next:

* Space-optimized explanation
* Relation to **binomial coefficients**
* Visual dependency graph
* Or contrast with **LeetCode 119 (Pascal’s Triangle II)**