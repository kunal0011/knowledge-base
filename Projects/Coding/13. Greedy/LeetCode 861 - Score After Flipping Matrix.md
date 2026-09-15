---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 861: Score After Flipping Matrix"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 861: Score After Flipping Matrix

Below is a **complete, structured, interview-grade explanation** of **LeetCode 861 – Score After Flipping Matrix**, aligned with your usual preparation style.

---

## LeetCode 861 – Score After Flipping Matrix

### Problem Statement

You are given a binary matrix `grid` of size `m x n`.

You may perform the following operations any number of times:

1. **Flip a row**: Toggle every value in that row (0 → 1, 1 → 0)
2. **Flip a column**: Toggle every value in that column

After performing any number of flips, **each row is interpreted as a binary number**, and the **score** is the sum of these numbers.

Return the **maximum possible score**.

---

### Key Observations

#### 1. Most Significant Bit (MSB) dominates the value

* Each row is treated as a binary number.
* The **leftmost column (column 0)** represents the **highest value**:  
  contribution = `2^(n-1)`
* Therefore:

  > It is always optimal to make the **first column all 1s**

#### 2. Row flips should be decided first

* If `grid[i][0] == 0`, flip the entire row.
* This guarantees:

  * Every row starts with `1`
  * We maximize the MSB contribution for every row

#### 3. Column-wise greedy choice

* After fixing rows:

  * For each column `j` (from 1 to n-1)
  * Count number of `1`s
* If `zeros > ones`, flip the column.

  * Because each column contributes independently

#### 4. No need to actually mutate the grid

* After row normalization:

  * Value at `grid[i][j]` becomes:

    ```
    grid[i][j] XOR (grid[i][0] == 0)
    ```
* We only need counts, not real flips.

---

### Greedy Strategy Summary

1. **Force MSB to 1** for all rows (row flip if needed)
2. **For each column**, maximize number of 1s
3. **Compute contribution directly**

This greedy works because:

* MSB has the highest weight
* Each column’s contribution is independent after row normalization

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        total_score = 0

        # Step 1: MSB column contribution (all rows forced to 1)
        total_score += m * (1 << (n - 1))

        # Step 2: Process remaining columns
        for col in range(1, n):
            ones = 0
            for row in range(m):
                # If grid[row][0] == 0, row is flipped
                val = grid[row][col]
                if grid[row][0] == 0:
                    val ^= 1
                ones += val

            max_ones = max(ones, m - ones)
            total_score += max_ones * (1 << (n - col - 1))

        return total_score
```

---

## Complete Worked Example (Step-by-Step)

### Input

```
grid = [
  [0,0,1,1],
  [1,0,1,0],
  [1,1,0,0]
]
```

Matrix size: `m = 3`, `n = 4`  
Column weights:

* col0 → 8
* col1 → 4
* col2 → 2
* col3 → 1

---

### Step 1: Fix First Column (Row Flips)

We want column 0 to be all `1`s.

| Row | Original | grid[i][0] | Action | Result |
| --- | --- | --- | --- | --- |
| 0 | 0 0 1 1 | 0 | flip row | 1 1 0 0 |
| 1 | 1 0 1 0 | 1 | no flip | 1 0 1 0 |
| 2 | 1 1 0 0 | 1 | no flip | 1 1 0 0 |

MSB contribution:

```
3 rows × 8 = 24
```

---

### Step 2: Column-wise Greedy Optimization

#### Column 1 (weight = 4)

Values after row normalization:

```
Row 0: 1
Row 1: 0
Row 2: 1
```

* ones = 2
* zeros = 1
* keep as is

Contribution:

```
2 × 4 = 8
```

---

#### Column 2 (weight = 2)

Values:

```
Row 0: 0
Row 1: 1
Row 2: 0
```

* ones = 1
* zeros = 2 → flip column

Contribution:

```
2 × 2 = 4
```

---

#### Column 3 (weight = 1)

Values:

```
Row 0: 0
Row 1: 0
Row 2: 0
```

* ones = 0
* zeros = 3 → flip column

Contribution:

```
3 × 1 = 3
```

---

### Final Score Calculation

| Column | Contribution |
| --- | --- |
| col0 | 24 |
| col1 | 8 |
| col2 | 4 |
| col3 | 3 |
| **Total** | **39** |

---

### Final Answer

```
39
```

---

### Why This Greedy Is Optimal

* MSB dominates all lower bits combined
* Row flips affect all columns uniformly
* After row normalization, each column is independent
* Always maximize count of `1`s per column

---

If you want:

* Decision tree / greedy proof sketch
* Equivalent bit-manipulation formulation
* Time–space complexity analysis
* Follow-up variants (constraints changed)

Tell me which one to expand.