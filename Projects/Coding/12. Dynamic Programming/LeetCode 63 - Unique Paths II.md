---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 63: Unique Paths II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - grid
  - obstacles
  - amazon
  - google
  - meta
---

# LeetCode 63: Unique Paths II

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Grid Traversal with Obstacles / 1D Space Optimization  

---

### Problem Statement

You are given an `m x n` integer array `obstacleGrid`. There is a robot initially located at the **top-left corner** (i.e., `grid[0][0]`). The robot tries to move to the **bottom-right corner** (i.e., `grid[m - 1][n - 1]`). The robot can only move either **down** or **right** at any point in time.

An obstacle and space are marked as `1` or `0` respectively in `grid`. A path that the robot takes cannot include any square that is an obstacle.

Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The testcases are generated so that the answer will be less than or equal to $2 \times 10^9$.

---

### Input & Output Formats & Constraints

- **Input:** `obstacleGrid: List[List[int]]` — 2D matrix where `1` = obstacle, `0` = open path.
- **Output:** `int` — Count of valid unique obstacle-free paths.
- **Constraints:**
  - $m == \text{obstacleGrid.length}$
  - $n == \text{obstacleGrid}[i].\text{length}$
  - $1 \le m, n \le 100$
  - `obstacleGrid[i][j]` is `0` or `1`.

---

### Key Idea & Intuition

1. **Obstacle Nullification:**
   - Any cell with an obstacle (`obstacleGrid[i][j] == 1`) cannot be traversed. Its contribution to any future path is strictly $0$:
     $$\text{dp}[i][j] = 0 \quad \text{if } \text{obstacleGrid}[i][j] == 1$$
   - For an open cell (`obstacleGrid[i][j] == 0`), paths can arrive from the top or left:
     $$\text{dp}[i][j] = \text{dp}[i - 1][j] + \text{dp}[i][j - 1]$$

2. **Corner Obstacle Edge Cases:**
   - If the starting cell has an obstacle (`obstacleGrid[0][0] == 1`), the robot cannot even start $\implies$ return `0`.
   - If the ending cell has an obstacle (`obstacleGrid[m - 1][n - 1] == 1`), the destination cannot be entered $\implies$ return `0`.

3. **1D Space Optimization:**
   - We maintain a single array `dp` of length $n$.
   - For each cell $(i, j)$:
     - If `obstacleGrid[i][j] == 1`: `dp[j] = 0` (cannot pass through).
     - Else if $j > 0$: `dp[j] += dp[j - 1]`.
   - The memory drops from $\mathcal{O}(m \times n)$ to $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. **Check Starting/Ending Obstacle:**
   - If `obstacleGrid[0][0] == 1` or `obstacleGrid[m - 1][n - 1] == 1`, return `0`.
2. **Initialize 1D DP Array:**
   - `dp = [0] * n`
   - Set `dp[0] = 1`.
3. **Iterate Grid:**
   - For $i$ from $0$ to $m - 1$:
     - For $j$ from $0$ to $n - 1$:
       - If `obstacleGrid[i][j] == 1`:
         - `dp[j] = 0`
       - Else if $j > 0$:
         - `dp[j] += dp[j - 1]`
4. **Return:**
   - Return `dp[n - 1]`.

---

### Visual Algorithm Walkthrough

For `obstacleGrid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]`:

```
Initial: dp = [1, 0, 0]

Row 0 ([0, 0, 0]):
  j = 0: open -> dp[0] = 1
  j = 1: open -> dp[1] += dp[0] = 1
  j = 2: open -> dp[2] += dp[1] = 1
  dp = [1, 1, 1]

Row 1 ([0, 1, 0]):
  j = 0: open -> dp[0] = 1 (from row 0)
  j = 1: OBSTACLE! -> dp[1] = 0
  j = 2: open -> dp[2] += dp[1] = 1 + 0 = 1
  dp = [1, 0, 1]

Row 2 ([0, 0, 0]):
  j = 0: open -> dp[0] = 1
  j = 1: open -> dp[1] += dp[0] = 0 + 1 = 1
  j = 2: open -> dp[2] += dp[1] = 1 + 1 = 2
  dp = [1, 1, 2]

Final Result: dp[2] = 2 unique paths.
```

---

### Solved Examples with Multiple Inputs

| Case | `obstacleGrid` | DP Execution | Result | Explanation |
|---|---|---|---|---|
| **Center Obstacle** | `[[0,0,0],[0,1,0],[0,0,0]]` | Path navigates around center obstacle | `2` | Two paths: Right-Down or Down-Right |
| **Start Blocked** | `[[1,0],[0,0]]` | Start cell has obstacle | `0` | Cannot take any step |
| **End Blocked** | `[[0,0],[0,1]]` | End cell has obstacle | `0` | Destination blocked |
| **Full Blockade** | `[[0,1],[1,0]]` | Both exit branches blocked | `0` | Robot is trapped at (0,0) |
| **Single Open Cell** | `[[0]]` | No moves needed | `1` | Start is destination |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — 1D Space Optimized)
```python
from typing import List

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        if not obstacleGrid or obstacleGrid[0][0] == 1:
            return 0
            
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [0] * n
        dp[0] = 1
        
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[j] = 0
                elif j > 0:
                    dp[j] += dp[j - 1]
                    
        return dp[n - 1]
```

#### 2. C++ (C++17 / STL — 1D Space Optimized)
```cpp
#include <vector>

class Solution {
public:
    int uniquePathsWithObstacles(std::vector<std::vector<int>>& obstacleGrid) {
        if (obstacleGrid.empty() || obstacleGrid[0][0] == 1) return 0;

        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        // Use long long to safely avoid signed 32-bit overflow in intermediate states
        std::vector<long long> dp(n, 0);
        dp[0] = 1;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else if (j > 0) {
                    dp[j] += dp[j - 1];
                }
            }
        }

        return static_cast<int>(dp[n - 1]);
    }
};
```

#### 3. Java (Modern, Typed — 1D Space Optimized)
```java
class Solution {
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0][0] == 1) {
            return 0;
        }

        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        long[] dp = new long[n];
        dp[0] = 1;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else if (j > 0) {
                    dp[j] += dp[j - 1];
                }
            }
        }

        return (int) dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m \times n)$  
  A single pass visiting each of the $m \times n$ cells once, with $\mathcal{O}(1)$ operations per cell. For $m, n \le 100$, operations $\le 10^4$ ($< 0.5$ ms).
- **Space Complexity:** $\mathcal{O}(n)$  
  A single 1D array of length $n$ stores the accumulated path counts.

---

### Takeaway Pattern & Interview Traps

1. **Why `dp[j] = 0` on Obstacles:**
   - In 1D DP compression, when an obstacle is hit, setting `dp[j] = 0` ensures that paths from the row above are nullified and do not propagate to subsequent rows or rightwards.
2. **Initial and Final Cell Checks:**
   - Many candidates forget that `obstacleGrid[0][0]` or `obstacleGrid[m-1][n-1]` can themselves be obstacles. Always check these boundary cells.