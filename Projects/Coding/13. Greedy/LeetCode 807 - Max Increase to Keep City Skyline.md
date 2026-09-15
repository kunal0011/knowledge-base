---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 807: Max Increase to Keep City Skyline"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 807: Max Increase to Keep City Skyline

Below is a **complete, structured explanation** of **LeetCode 807 – Max Increase to Keep City Skyline**, aligned with your usual DSA-learning format.

---

## LeetCode 807 — Max Increase to Keep City Skyline

---

## 1. Problem Statement

You are given an `n x n` integer matrix `grid` where:

* `grid[i][j]` represents the height of the building located at row `i`, column `j`.

A city's **skyline** is defined as:

* **Top view**: the maximum height in each column.
* **Side view**: the maximum height in each row.

You may increase the height of any building **any number of times**, but **you must not change the skyline** when viewed from the top or the side.

### Objective

Return the **maximum total sum** by which the height of the buildings can be increased **without affecting the skyline**.

---

## 2. Key Observations

1. **Row skyline constraint**

   * For row `i`, the maximum allowed height for any building in that row is:

     ```
     rowMax[i] = max(grid[i][*])
     ```
2. **Column skyline constraint**

   * For column `j`, the maximum allowed height for any building in that column is:

     ```
     colMax[j] = max(grid[*][j])
     ```
3. **Cell-wise maximum possible height**

   * For building at `(i, j)`, the height **cannot exceed either skyline**:

     ```
     allowedHeight(i, j) = min(rowMax[i], colMax[j])
     ```
4. **Increase at a cell**

   * If current height is `grid[i][j]`, the increase is:

     ```
     increase = allowedHeight(i, j) - grid[i][j]
     ```
5. **Greedy Insight**

   * Since each cell is independently constrained by its row and column skyline,  
     **we greedily increase every cell to its maximum allowed height**.

This greedy choice is always optimal because:

* Increasing one building does **not affect** constraints of others.
* Constraints are **fixed and independent**.

---

## 3. Greedy Strategy (Why It Works)

* Compute row-wise maximums.
* Compute column-wise maximums.
* For every cell:

  * Raise it to the **minimum of its row and column max**.
  * Accumulate the increase.

Time and space complexity remain optimal.

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n = len(grid)

        # Step 1: Compute row maximums
        row_max = [max(row) for row in grid]

        # Step 2: Compute column maximums
        col_max = [max(grid[i][j] for i in range(n)) for j in range(n)]

        # Step 3: Compute total increase
        total_increase = 0
        for i in range(n):
            for j in range(n):
                allowed_height = min(row_max[i], col_max[j])
                total_increase += allowed_height - grid[i][j]

        return total_increase
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input Grid

```
grid = [
  [3, 0, 8, 4],
  [2, 4, 5, 7],
  [9, 2, 6, 3],
  [0, 3, 1, 0]
]
```

---

### Step 1: Compute Row Maximums

| Row | Values | rowMax |
| --- | --- | --- |
| 0 | [3, 0, 8, 4] | 8 |
| 1 | [2, 4, 5, 7] | 7 |
| 2 | [9, 2, 6, 3] | 9 |
| 3 | [0, 3, 1, 0] | 3 |

```
rowMax = [8, 7, 9, 3]
```

---

### Step 2: Compute Column Maximums

| Column | Values | colMax |
| --- | --- | --- |
| 0 | [3, 2, 9, 0] | 9 |
| 1 | [0, 4, 2, 3] | 4 |
| 2 | [8, 5, 6, 1] | 8 |
| 3 | [4, 7, 3, 0] | 7 |

```
colMax = [9, 4, 8, 7]
```

---

### Step 3: Cell-by-Cell Processing

| Cell (i,j) | Current | min(rowMax, colMax) | Increase |
| --- | --- | --- | --- |
| (0,0) | 3 | min(8,9)=8 | +5 |
| (0,1) | 0 | min(8,4)=4 | +4 |
| (0,2) | 8 | min(8,8)=8 | +0 |
| (0,3) | 4 | min(8,7)=7 | +3 |
| (1,0) | 2 | min(7,9)=7 | +5 |
| (1,1) | 4 | min(7,4)=4 | +0 |
| (1,2) | 5 | min(7,8)=7 | +2 |
| (1,3) | 7 | min(7,7)=7 | +0 |
| (2,0) | 9 | min(9,9)=9 | +0 |
| (2,1) | 2 | min(9,4)=4 | +2 |
| (2,2) | 6 | min(9,8)=8 | +2 |
| (2,3) | 3 | min(9,7)=7 | +4 |
| (3,0) | 0 | min(3,9)=3 | +3 |
| (3,1) | 3 | min(3,4)=3 | +0 |
| (3,2) | 1 | min(3,8)=3 | +2 |
| (3,3) | 0 | min(3,7)=3 | +3 |

---

### Step 4: Sum of All Increases

```
Total Increase = 35
```

---

## 6. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n²)** |
| Space Complexity | **O(n)** (rowMax + colMax) |

---

## 7. Key Takeaways (Greedy Pattern)

* **Independent local decisions** with **global constraints**.
* Skyline constraints are **fixed**, making greedy optimal.
* Common pattern:  
  **Precompute constraints → maximize each cell independently**.

If you want, I can also:

* Reframe this as a **matrix greedy pattern**
* Compare with similar problems (e.g., trapping rainwater 2D intuition)
* Provide a dry-run template you can reuse in interviews