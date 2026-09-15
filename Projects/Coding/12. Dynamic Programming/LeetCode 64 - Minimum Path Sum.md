---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 64: Minimum Path Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - grid
  - pathfinding
  - amazon
  - google
  - meta
---

# LeetCode 64: Minimum Path Sum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** 2D/1D Dynamic Programming / Grid Pathfinding / 1D Space Optimization  

---

### Problem Statement

Given a `m x n` `grid` filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

**Note:** You can only move either down or right at any point in time.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[int]]` — 2D matrix of non-negative integers.
- **Output:** `int` — Minimum cumulative path sum from $(0, 0)$ to $(m-1, n-1)$.
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 200$
  - $0 \le \text{grid}[i][j] \le 200$

---

### Key Idea & Intuition

1. **Optimal Substructure:**
   - To arrive at cell $(i, j)$, the robot can only step from:
     - The cell above: $(i - 1, j)$
     - The cell to the left: $(i, j - 1)$
   - By the Principle of Optimality, the minimum cost to reach $(i, j)$ is the cost of cell $(i, j)$ plus the minimum of the optimal costs to reach those two immediate predecessors:
     $$\text{dp}[i][j] = \text{grid}[i][j] + \min(\text{dp}[i - 1][j], \ \text{dp}[i][j - 1])$$
   - Base Case: $\text{dp}[0][0] = \text{grid}[0][0]$.

2. **1D Space Optimization ($\mathcal{O}(n)$ Space):**
   - When evaluating row $i$:
     - $\text{dp}[j]$ currently holds the minimum path sum to reach cell $(i - 1, j)$ from the row above.
     - $\text{dp}[j - 1]$ holds the newly updated minimum path sum to reach cell $(i, j - 1)$ from the left.
   - We update in-place:
     $$\text{dp}[j] = \text{grid}[i][j] + \min(\text{dp}[j], \ \text{dp}[j - 1])$$
   - This compresses matrix storage down to a single 1D array of length $n$.

---

### Solution Approach (Step-by-Step)

1. **Initialize 1D DP Array:**
   - Let $m = \text{len}(grid)$ and $n = \text{len}(grid[0])$.
   - Create array `dp` of size $n$.
   - Base case row 0:
     - `dp[0] = grid[0][0]`.
     - For $j$ from $1$ to $n - 1$: `dp[j] = dp[j - 1] + grid[0][j]`.
2. **Iterate Subsequent Rows:**
   - For $i$ from $1$ to $m - 1$:
     - Column 0 can only be reached from above: `dp[0] += grid[i][0]`.
     - For $j$ from $1$ to $n - 1$:
       - `dp[j] = grid[i][j] + min(dp[j], dp[j - 1])`.
3. **Return:**
   - Return `dp[n - 1]`.

---

### Visual Algorithm Walkthrough

For input `grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]`:

```
Initial Grid:
[ 1, 3, 1 ]
[ 1, 5, 1 ]
[ 4, 2, 1 ]

Row 0 (can only move right):
  dp = [1, 1+3=4, 4+1=5] -> [1, 4, 5]

Row 1:
  j = 0: dp[0] += grid[1][0] = 1 + 1 = 2
  j = 1: dp[1] = 5 + min(top=4, left=2) = 5 + 2 = 7
  j = 2: dp[2] = 1 + min(top=5, left=7) = 1 + 5 = 6
  dp = [2, 7, 6]

Row 2:
  j = 0: dp[0] += grid[2][0] = 2 + 4 = 6
  j = 1: dp[1] = 2 + min(top=7, left=6) = 2 + 6 = 8
  j = 2: dp[2] = 1 + min(top=6, left=8) = 1 + 6 = 7
  dp = [6, 8, 7]

Final Minimum Path Sum: dp[2] = 7.
Path: 1 -> 3 -> 1 -> 1 -> 1 (Sum = 7).
```

---

### Solved Examples with Multiple Inputs

| Case | `grid` | Minimum Path | Result | Explanation |
|---|---|---|---|---|
| **Standard 3x3** | `[[1,3,1],[1,5,1],[4,2,1]]` | $1 \to 3 \to 1 \to 1 \to 1$ | `7` | Optimal route avoids center $5$ |
| **Standard 2x3** | `[[1,2,3],[4,5,6]]` | $1 \to 2 \to 3 \to 6$ | `12` | Straight right then down |
| **Single Element** | `[[5]]` | Only cell | `5` | Start is destination |
| **Single Column** | `[[1],[2],[3]]` | Straight down | `6` | $1 + 2 + 3 = 6$ |
| **All Zeroes** | `[[0,0],[0,0]]` | Any path | `0` | Sum of all zeroes |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — 1D Space Optimized)
```python
from typing import List

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dp = [0] * n
        
        # Initialize row 0
        dp[0] = grid[0][0]
        for j in range(1, n):
            dp[j] = dp[j - 1] + grid[0][j]
            
        # Process rows 1 to m - 1
        for i in range(1, m):
            dp[0] += grid[i][0]
            for j in range(1, n):
                dp[j] = grid[i][j] + min(dp[j], dp[j - 1])
                
        return dp[n - 1]
```

#### 2. C++ (C++17 / STL — 1D Space Optimized)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int minPathSum(std::vector<std::vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();
        std::vector<int> dp(n, 0);

        dp[0] = grid[0][0];
        for (int j = 1; j < n; ++j) {
            dp[j] = dp[j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; ++i) {
            dp[0] += grid[i][0];
            for (int j = 1; j < n; ++j) {
                dp[j] = grid[i][j] + std::min(dp[j], dp[j - 1]);
            }
        }

        return dp[n - 1];
    }
};
```

#### 3. Java (Modern, Typed — 1D Space Optimized)
```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] dp = new int[n];

        dp[0] = grid[0][0];
        for (int j = 1; j < n; j++) {
            dp[j] = dp[j - 1] + grid[0][j];
        }

        for (int i = 1; i < m; i++) {
            dp[0] += grid[i][0];
            for (int j = 1; j < n; j++) {
                dp[j] = grid[i][j] + Math.min(dp[j], dp[j - 1]);
            }
        }

        return dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  Every cell in the matrix is processed exactly once, executing $\mathcal{O}(1)$ basic arithmetic and min operations. For $m, n \le 200$, operations $\le 4 \times 10^4$ ($< 2$ ms).
- **Space Complexity:** $\mathcal{O}(n)$  
  Only a single 1D vector of length $n$ is maintained in memory. (Can be reduced to $\mathcal{O}(1)$ if modifying the input `grid` in-place is permitted).

---

### Takeaway Pattern & Interview Traps

1. **In-Place Modification Caution:**
   - In interviews, candidates often propose overwriting `grid[i][j]` directly to claim $\mathcal{O}(1)$ space. Always ask the interviewer whether mutating the input matrix is acceptable. A separate 1D array of size $n$ is safe, clean, and avoids modifying client data.
2. **Dijkstra vs. DP:**
   - If moves in all 4 directions were allowed, Dijkstra's algorithm ($\mathcal{O}(V \log V)$) would be required. Because moves are strictly restricted to **Down** and **Right**, the grid is a Directed Acyclic Graph (DAG), making topological Dynamic Programming $\mathcal{O}(m \times n)$ optimal.