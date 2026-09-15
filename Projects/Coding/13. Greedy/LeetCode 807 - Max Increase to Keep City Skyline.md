---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 807: Max Increase to Keep City Skyline"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - matrix
  - google
  - amazon
  - apple
---

# LeetCode 807: Max Increase to Keep City Skyline

**Target Companies:** Google, Amazon, Adobe, Apple  
**Difficulty:** Medium  
**Topic:** Greedy / Array / Matrix  

---

### Problem Statement

There is a city composed of $n \times n$ blocks, where each block contains a single building whose height is given in an $n \times n$ integer matrix `grid` where `grid[r][c]` represents the height of the building at row `r` and column `c`.

A city's **skyline** is the outer contour formed by all the buildings when looking at the city from a distance:
- The **north/south skyline** is the maximum height in each column.
- The **east/west skyline** is the maximum height in each row.

You are allowed to increase the height of any number of buildings by any amount (the amount can be different for each building). The height of a building cannot be decreased, and **you must not change the city's skyline** from any cardinal direction.

Return *the **maximum total sum** by which the height of the buildings can be increased*.

---

### Input & Output Formats & Constraints

- **Input:** An $n \times n$ integer matrix `grid`.
- **Output:** An integer representing the maximum total increase in building heights across all cells.
- **Constraints:**
  - $n == grid.length$
  - $n == grid[i].length$
  - $2 \le n \le 50$
  - $0 \le grid[r][c] \le 100$

---

### Key Idea & Intuition

#### Skyline Invariants
For any building at cell $(r, c)$:
1. The row skyline viewed from East or West is determined by the maximum building height in row $r$:
   $$\text{row\_max}[r] = \max_{0 \le j < n} \text{grid}[r][j]$$
2. The column skyline viewed from North or South is determined by the maximum building height in column $c$:
   $$\text{col\_max}[c] = \max_{0 \le i < n} \text{grid}[i][c]$$

#### The Safe Upper Bound
If we increase $\text{grid}[r][c]$ to a new height $H$:
- To preserve the row skyline, we must have $H \le \text{row\_max}[r]$.
- To preserve the column skyline, we must have $H \le \text{col\_max}[c]$.

Together, this gives the strict bounding condition:
$$H \le \min(\text{row\_max}[r], \text{col\_max}[c])$$

#### Independence of Decisions (Greedy Choice)
Because $\text{row\_max}[r]$ and $\text{col\_max}[c]$ are determined by the *existing original* maximums of each row and column, raising $\text{grid}[r][c]$ to exactly $\min(\text{row\_max}[r], \text{col\_max}[c])$:
- Will never exceed the existing maximum of row $r$.
- Will never exceed the existing maximum of column $c$.
- Operates completely independently for every cell $(r, c)$.

Thus, greedily increasing every cell to its maximal permissible height $\min(\text{row\_max}[r], \text{col\_max}[c])$ achieves the global maximum total increase.

---

### Solution Approach (Step-by-Step)

1. **Precompute Skylines:**
   - Compute `row_max[r]` for each row $r \in [0, n - 1]$.
   - Compute `col_max[c]` for each column $c \in [0, n - 1]$.
2. **Compute Total Height Increase:**
   - Initialize `total_increase = 0`.
   - Traverse each cell $(r, c)$:
     - $\text{allowed\_height} = \min(\text{row\_max}[r], \text{col\_max}[c])$.
     - $\text{total\_increase} += \text{allowed\_height} - \text{grid}[r][c]$.
3. **Return Output:**
   - Return `total_increase`.

---

### Visual Algorithm Walkthrough

#### Trace for `grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]`
```
Original Grid:
   Col:  0   1   2   3    -> row_max
Row 0:  [3,  0,  8,  4]   -> 8
Row 1:  [2,  4,  5,  7]   -> 7
Row 2:  [9,  2,  6,  3]   -> 9
Row 3:  [0,  3,  1,  0]   -> 3
         |   |   |   |
col_max: 9   4   8   7

Allowed Max Height Matrix: M[r][c] = min(row_max[r], col_max[c])
Row 0: [min(8,9)=8, min(8,4)=4, min(8,8)=8, min(8,7)=7] = [8, 4, 8, 7]
Row 1: [min(7,9)=7, min(7,4)=4, min(7,8)=7, min(7,7)=7] = [7, 4, 7, 7]
Row 2: [min(9,9)=9, min(9,4)=4, min(9,8)=8, min(9,7)=7] = [9, 4, 8, 7]
Row 3: [min(3,9)=3, min(3,4)=3, min(3,8)=3, min(3,7)=3] = [3, 3, 3, 3]

Cell Differences (Allowed - Original):
Row 0: (8-3=5) + (4-0=4) + (8-8=0) + (7-4=3) = 12
Row 1: (7-2=5) + (4-4=0) + (7-5=2) + (7-7=0) = 7
Row 2: (9-9=0) + (4-2=2) + (8-6=2) + (7-3=4) = 8
Row 3: (3-0=3) + (3-3=0) + (3-1=2) + (3-0=3) = 8

Total Increase = 12 + 7 + 8 + 8 = 35.
```

---

### Solved Examples with Multiple Inputs

| Input `grid` | `row_max` | `col_max` | Cell Height Increases Sum | Output |
|---|---|---|---|---|
| `[[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]` | `[8, 7, 9, 3]` | `[9, 4, 8, 7]` | $12 + 7 + 8 + 8 = 35$ | `35` |
| `[[0,0,0],[0,0,0],[0,0,0]]` | `[0, 0, 0]` | `[0, 0, 0]` | All cells allowed height 0 $\to 0$ | `0` |
| `[[1,2],[3,4]]` | `[2, 4]` | `[3, 4]` | $(2-1)+(2-2)+(3-3)+(4-4) = 1$ | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: list[list[int]]) -> int:
        n: int = len(grid)
        row_max: list[int] = [max(row) for row in grid]
        col_max: list[int] = [max(grid[r][c] for r in range(n)) for c in range(n)]
        
        total_increase: int = 0
        for r in range(n):
            for c in range(n):
                total_increase += min(row_max[r], col_max[c]) - grid[r][c]
                
        return total_increase
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxIncreaseKeepingSkyline(const std::vector<std::vector<int>>& grid) {
        int n = static_cast<int>(grid.size());
        std::vector<int> row_max(n, 0);
        std::vector<int> col_max(n, 0);
        
        // Find row maximums and column maximums
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                row_max[r] = std::max(row_max[r], grid[r][c]);
                col_max[c] = std::max(col_max[c], grid[r][c]);
            }
        }
        
        int total_increase = 0;
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                total_increase += std::min(row_max[r], col_max[c]) - grid[r][c];
            }
        }
        
        return total_increase;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int n = grid.length;
        int[] rowMax = new int[n];
        int[] colMax = new int[n];
        
        // Precalculate maximums
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                rowMax[r] = Math.max(rowMax[r], grid[r][c]);
                colMax[c] = Math.max(colMax[c], grid[r][c]);
            }
        }
        
        int totalIncrease = 0;
        for (int r = 0; r < n; r++) {
            for (int c = 0; c < n; c++) {
                totalIncrease += Math.min(rowMax[r], colMax[c]) - grid[r][c];
            }
        }
        
        return totalIncrease;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n^2)$, where $n$ is the dimension of the grid. Computing the row and column maximums requires iterating through all $n^2$ elements, and the subsequent summation pass checks each element once.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space to store the `row_max` and `col_max` arrays of size $n$.

---

### Takeaway Pattern & Interview Traps

1. **Cell-Wise Independence:** Because the limits are dictated by the initial maximums of the row and column (which are already achieved by at least one building in each row and column), modifications to other cells never perturb the skyline.
2. **Avoid Repeated Max Scans:** Do not compute `max(row)` or `max(col)` inside the nested loop; doing so increases time complexity from $\mathcal{O}(n^2)$ to $\mathcal{O}(n^3)$. Precomputation keeps it linear in the number of cells.