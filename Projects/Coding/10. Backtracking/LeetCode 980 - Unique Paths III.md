---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 980: Unique Paths III"
tags:
  - leetcode
  - coding
  - backtracking
  - matrix
  - array
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 980: Unique Paths III

**Target Companies:** Amazon, Google, Uber, Apple  
**Difficulty:** Hard  
**Topic:** Backtracking / Grid Hamiltonian Path / In-Place State Marking  

---

### Problem Statement

You are given an `m x n` integer array `grid` where:
- `1` represents the starting square. There is exactly one starting square.
- `2` represents the ending square. There is exactly one ending square.
- `0` represents empty squares we can walk over.
- `-1` represents obstacles that we cannot walk over.

Return *the number of 4-directional walks from the starting square to the ending square, that walk over every non-obstacle square exactly once*.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[int]]` of dimensions $m \times n$
- **Output:** `int` (number of valid Hamiltonian paths)
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i]\text{.length}$
  - $1 \le m, n \le 20$
  - $1 \le m \times n \le 20$
  - `-1 <= grid[i][j] <= 2`
  - There is exactly one starting cell and one ending cell.

---

### Key Idea & Intuition

- **Hamiltonian Path Problem on Grid Graphs:**
  - Unlike LeetCode 62/63 (Unique Paths I & II) which are DAGs with right-and-down moves solvable via Dynamic Programming, LeetCode 980 allows moves in all **4 directions** and requires visiting **every single non-obstacle cell**.
  - Finding a path that visits all vertices exactly once in a general graph is the NP-complete **Hamiltonian Path** problem.
  - Because $m \times n \le 20$, the grid graph has at most 20 cells, which is tailored for state-space DFS backtracking.
- **Tracking Remaining Empty Cells:**
  - Count the total number of empty cells `empty_cells` (cells with value `0`).
  - Including the starting cell, the total number of steps required before entering the destination cell `2` is `empty_cells + 1`.
  - When reaching destination cell `2`:
    - The path is valid if and only if `remaining_steps == 0`.
- **In-Place Grid Modification:**
  - When stepping onto cell `(r, c)`, mark `grid[r][c] = -1` (temporarily treating it as an obstacle).
  - Recurse in 4 directions.
  - Restore `grid[r][c] = 0` (or `1`) upon backtracking.

---

### Solution Approach (Step-by-Step)

1. Traverse `grid` to locate starting coordinates `(start_r, start_c)` and count total empty cells `empty_cells`.
2. Initialize `valid_paths = 0`.
3. Define `dfs(r, c, remaining)`:
   - If `grid[r][c] == 2`:
     - If `remaining == 0`:
       - `valid_paths += 1`
     - Return.
   - Save original cell value: `temp = grid[r][c]`.
   - Mark cell as visited: `grid[r][c] = -1`.
   - For `(dr, dc)` in `[(1, 0), (-1, 0), (0, 1), (0, -1)]`:
     - `nr, nc = r + dr, c + dc`
     - If $0 \le nr < m$ and $0 \le nc < n$ and `grid[nr][nc] != -1`:
       - `dfs(nr, nc, remaining - 1)`
   - Backtrack: `grid[r][c] = temp`.
4. Call `dfs(start_r, start_c, empty_cells + 1)`.
5. Return `valid_paths`.

---

### Visual Algorithm Walkthrough

Let `grid` =
```
[
  [1,  0,  0,  0],
  [0,  0,  0,  0],
  [0,  0,  2, -1]
]
```
- Total cells: 12.
- Obstacles (`-1`): 1 cell.
- Start (`1`) and End (`2`): 2 cells.
- Empty cells (`0`): 9 cells.
- Total steps required before reaching `2` = $9 + 1 = 10$.

```
Path 1:
(0,0) -> (0,1) -> (0,2) -> (0,3)
                         |
(1,0) <- (1,1) <- (1,2) <- (1,3)
  |
(2,0) -> (2,1) -> (2,2) [End!]
Step count = 10 == required. All 9 empty cells visited!
Valid Path Count: +1
```

Any path that wanders into `2` prematurely (e.g. `(0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)`) arrives with `remaining > 0` and is rejected!

---

### Solved Examples with Multiple Inputs

| Test Case | Grid Dimensions | Empty Cells | Obstacles | Total Valid Paths | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | $3 \times 4$ | 9 | 1 obstacle at `(2,3)` | 2 symmetrical full-grid snake paths | `2` |
| **Example 2** | $3 \times 4$ | 8 | 2 obstacles | 4 paths | `4` |
| **Example 3** | $3 \times 3$ | 6 | Trapped end | 0 paths (impossible to visit all without blocking) | `0` |
| **Minimal Grid** | $1 \times 2$ (`[[1, 2]]`) | 0 | None | Directly step to 2 | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        """
        Finds the number of unique Hamiltonian paths visiting every non-obstacle cell.
        Uses in-place backtracking with remaining empty cell count.
        """
        rows, cols = len(grid), len(grid[0])
        start_r, start_c = 0, 0
        empty_count = 0

        # Find starting point and count walkable empty squares
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    start_r, start_c = r, c
                elif grid[r][c] == 0:
                    empty_count += 1

        valid_paths = 0

        def dfs(r: int, c: int, remaining: int) -> None:
            nonlocal valid_paths

            if grid[r][c] == 2:
                # Valid path only if all empty cells have been walked over
                if remaining == 0:
                    valid_paths += 1
                return

            temp = grid[r][c]
            grid[r][c] = -1  # Mark visited in-place

            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != -1:
                    dfs(nr, nc, remaining - 1)

            grid[r][c] = temp  # Backtrack

        # Remaining steps from start to destination is empty_count + 1
        dfs(start_r, start_c, empty_count + 1)
        return valid_paths
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int uniquePathsIII(std::vector<std::vector<int>>& grid) {
        int m = static_cast<int>(grid.size());
        int n = static_cast<int>(grid[0].size());
        int start_r = 0, start_c = 0;
        int empty_count = 0;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (grid[r][c] == 1) {
                    start_r = r;
                    start_c = c;
                } else if (grid[r][c] == 0) {
                    empty_count++;
                }
            }
        }

        int valid_paths = 0;
        dfs(start_r, start_c, empty_count + 1, m, n, grid, valid_paths);
        return valid_paths;
    }

private:
    void dfs(int r, int c, int remaining, int m, int n,
             std::vector<std::vector<int>>& grid, int& valid_paths) {
        if (grid[r][c] == 2) {
            if (remaining == 0) {
                valid_paths++;
            }
            return;
        }

        int temp = grid[r][c];
        grid[r][c] = -1; // Mark visited

        static const int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (const auto& d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] != -1) {
                dfs(nr, nc, remaining - 1, m, n, grid, valid_paths);
            }
        }

        grid[r][c] = temp; // Backtrack
    }
};
```

#### Java
```java
class Solution {
    private int validPaths = 0;

    public int uniquePathsIII(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int startR = 0, startC = 0;
        int emptyCount = 0;
        validPaths = 0;

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 1) {
                    startR = r;
                    startC = c;
                } else if (grid[r][c] == 0) {
                    emptyCount++;
                }
            }
        }

        dfs(startR, startC, emptyCount + 1, m, n, grid);
        return validPaths;
    }

    private void dfs(int r, int c, int remaining, int m, int n, int[][] grid) {
        if (grid[r][c] == 2) {
            if (remaining == 0) {
                validPaths++;
            }
            return;
        }

        int temp = grid[r][c];
        grid[r][c] = -1; // Mark visited

        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] d : dirs) {
            int nr = r + d[0];
            int nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] != -1) {
                dfs(nr, nc, remaining - 1, m, n, grid);
            }
        }

        grid[r][c] = temp; // Backtrack
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(4^{M \times N})$.
  - In the worst case, every cell branches in 4 directions.
  - However, because cells cannot be revisited, each step after the first has at most 3 directions. Furthermore, dead-ends terminate immediately without reaching $2$.
  - For $M \times N \le 20$, the maximum path length is 20, executing in under $10 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(M \times N)$ auxiliary space for the recursion call stack depth (at most 20 stack frames). Grid state is updated in-place without auxiliary arrays.

---

### Takeaway Pattern & Interview Traps

- **Premature Destination Reaching:** The destination square `2` must not simply be reached; it must be reached with **zero unvisited empty cells left**! If you return `remaining == 0` without checking `grid[r][c] == 2`, you'll count paths that end in dead-end empty squares.
- **Counting the Start Cell:** Don't forget that moving off `1` consumes 1 step. That's why initial remaining steps is `empty_count + 1`.