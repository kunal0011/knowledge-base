---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 329: Longest Increasing Path in a Matrix"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - dynamic-programming
  - memoization
  - amazon
  - google
---

# LeetCode 329: Longest Increasing Path in a Matrix

**Target Companies:** Google (All-Time Top #1 Signature Hard), Amazon, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** Directed Acyclic Graph (DAG) / DFS with Memoization / Longest Path in Grid

---

### Problem Statement

Given an `m x n` integers `matrix`, return the length of the **longest increasing path** in `matrix`.

From each cell, you can either move in four directions: left, right, up, or down. You **may not** move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

---

### Input & Output Formats & Constraints

- **Input:** `matrix: List[List[int]]`
- **Output:** `int` — Length of the longest strictly increasing path (in number of cells).
- **Constraints:**
  - $m == \text{matrix.length}$
  - $n == \text{matrix}[i].\text{length}$
  - $1 \le m, n \le 200$
  - $0 \le \text{matrix}[i][j] \le 2^{31} - 1$

---

### Key Idea & Intuition

- **Inherent DAG Property (Cycle-Free Guarantee):**
  - Any valid move must transition from cell `(r, c)` to a neighbor `(nr, nc)` satisfying:
    $$\text{matrix}[nr][nc] > \text{matrix}[r][c]$$
  - Because values along a path are **strictly increasing**, a path can never loop back to an earlier cell ($v_1 < v_2 < \dots < v_k < v_1$ is mathematically impossible).
  - Hence, the grid forms an implicit **Directed Acyclic Graph (DAG)**!
- **DFS with Memoization:**
  - On any DAG, the longest path starting from a vertex can be computed via Dynamic Programming with Memoization:
    $$\text{dp}[r][c] = 1 + \max_{(nr, nc)} \left\{ \text{dfs}(nr, nc) \mid \text{matrix}[nr][nc] > \text{matrix}[r][c] \right\}$$
  - If a cell has no strictly larger neighbors, it is a sink with path length $1$.
  - Because cycles are impossible, **no `visited` set is required**!
  - Caching results in `memo[r][c]` guarantees each cell is computed exactly once in $\mathcal{O}(1)$ amortized time.

---

### Solution Approach (Step-by-Step)

1. Let $m = \text{len}(matrix), n = \text{len}(matrix[0])$.
2. Initialize 2D table `memo` of size $m \times n$ filled with $0$.
3. Define `dfs(r, c)`:
   - If `memo[r][c] != 0`: return `memo[r][c]`.
   - Initialize `max_len = 1`.
   - For `(dr, dc)` in `[(0, 1), (0, -1), (1, 0), (-1, 0)]`:
     - `nr = r + dr, nc = c + dc`.
     - If $0 \le nr < m$ and $0 \le nc < n$ and `matrix[nr][nc] > matrix[r][c]`:
       - `max_len = max(max_len, 1 + dfs(nr, nc))`.
   - `memo[r][c] = max_len`.
   - Return `max_len`.
4. Scan all cells `(r, c)` in the matrix, computing $\max_{r, c}(\text{dfs}(r, c))$.
5. Return the global maximum.

---

### Visual Algorithm Walkthrough

```
Matrix:
9  9  4
6  6  8
2  1  1

DAG Transitions (directed toward strictly larger neighbors):
(2, 1) [1] -> (1, 0) [6] -> (0, 0) [9]
(2, 1) [1] -> (2, 0) [2] -> (1, 0) [6] -> (0, 0) [9]
(2, 2) [1] -> (1, 2) [8]

Longest Path Search:
Start at (2, 1) [val = 1]:
  -> neighbor (2, 0) [val = 2, 2 > 1]
    -> neighbor (1, 0) [val = 6, 6 > 2]
      -> neighbor (0, 0) [val = 9, 9 > 6]
        -> no neighbor > 9 -> returns 1
      -> (1, 0) returns 1 + 1 = 2
    -> (2, 0) returns 1 + 2 = 3
  -> (2, 1) returns 1 + 3 = 4

Path: 1 -> 2 -> 6 -> 9 has length 4.
Global maximum length = 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Matrix
- **Input:**
  ```python
  matrix = [
    [9, 9, 4],
    [6, 6, 8],
    [2, 1, 1]
  ]
  ```
- **Longest Path:** `[1, 2, 6, 9]`
- **Output:** `4`

#### Example 2: Decreasing Strip
- **Input:**
  ```python
  matrix = [
    [3, 4, 5],
    [3, 2, 6],
    [2, 2, 1]
  ]
  ```
- **Longest Path:** `[1, 2, 3, 4, 5, 6]`
- **Output:** `4` (e.g. `3 -> 4 -> 5 -> 6`)

#### Example 3: Single Cell
- **Input:** `matrix = [[1]]`
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        if not matrix or not matrix[0]:
            return 0
            
        m, n = len(matrix), len(matrix[0])
        memo = [[0] * n for _ in range(m)]
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        def dfs(r: int, c: int) -> int:
            if memo[r][c] != 0:
                return memo[r][c]
                
            max_len = 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                    max_len = max(max_len, 1 + dfs(nr, nc))
                    
            memo[r][c] = max_len
            return max_len
            
        return max(dfs(r, c) for r in range(m) for c in range(n))
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestIncreasingPath(std::vector<std::vector<int>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size();
        std::vector<std::vector<int>> memo(m, std::vector<int>(n, 0));

        int maxPath = 0;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                maxPath = std::max(maxPath, dfs(r, c, matrix, memo, m, n));
            }
        }
        return maxPath;
    }

private:
    int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    int dfs(int r, int c, const std::vector<std::vector<int>>& matrix,
            std::vector<std::vector<int>>& memo, int m, int n) {
        if (memo[r][c] != 0) {
            return memo[r][c];
        }

        int maxLen = 1;
        for (auto& d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && matrix[nr][nc] > matrix[r][c]) {
                maxLen = std::max(maxLen, 1 + dfs(nr, nc, matrix, memo, m, n));
            }
        }

        memo[r][c] = maxLen;
        return maxLen;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    private int[][] memo;
    private int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    private int m, n;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }

        m = matrix.length;
        n = matrix[0].length;
        memo = new int[m][n];

        int maxPath = 0;
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                maxPath = Math.max(maxPath, dfs(r, c, matrix));
            }
        }
        return maxPath;
    }

    private int dfs(int r, int c, int[][] matrix) {
        if (memo[r][c] != 0) {
            return memo[r][c];
        }

        int maxLen = 1;
        for (int[] d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && matrix[nr][nc] > matrix[r][c]) {
                maxLen = Math.max(maxLen, 1 + dfs(nr, nc, matrix));
            }
        }

        memo[r][c] = maxLen;
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \times N)$ — Each cell `(r, c)` is computed via `dfs` once and cached. Across the entire traversal, each directed edge between adjacent cells is traversed at most once. Total operations are strictly bounded by $4 \times M \times N = \mathcal{O}(M \times N)$.
- **Space Complexity:** $\mathcal{O}(M \times N)$ — The 2D `memo` table takes $\mathcal{O}(M \times N)$ space. In the worst case (e.g. a strictly ascending spiral matrix), the recursion call stack can reach depth $M \times N$.

---

### Takeaway Pattern & Interview Traps

1. **Why No `visited` Array is Needed:** In general graph DFS, a `visited` set prevents cycles. But because transitions require `matrix[nr][nc] > matrix[r][c]`, a path cannot visit the same node twice without violating monotonicity. Omitting `visited` avoids unnecessary state resets and backtracking.
2. **Topological Sort Alternative:** Since this is a DAG, you can also solve it using Kahn's algorithm: calculate out-degrees (number of strictly greater neighbors), start BFS from cells with out-degree 0 (local peaks), and work backwards in layers to find the maximum layer count.