---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 417: Pacific Atlantic Water Flow"
tags:
  - leetcode
  - coding
  - graphs
  - dfs
  - bfs
  - matrix
  - amazon
  - google
---

# LeetCode 417: Pacific Atlantic Water Flow

**Target Companies:** Google (All-Time Signature Grid Flow Classic), Amazon, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Multi-Source Reverse Graph Traversal / DFS & BFS / Grid Reachability

---

### Problem Statement

There is an `m x n` rectangular island that borders both the **Pacific Ocean** and **Atlantic Ocean**. The **Pacific Ocean** touches the island's top and left edges, and the **Atlantic Ocean** touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an `m x n` integer matrix `heights` where `heights[r][c]` represents the **height above sea level** of the cell at coordinate `(r, c)`.

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is **less than or equal to** the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a **2D list of grid coordinates** `result` where `result[i] = [r_i, c_i]` denotes that rain water can flow from cell `(r_i, c_i)` to **both** the Pacific and Atlantic oceans.

---

### Input & Output Formats & Constraints

- **Input:** `heights: List[List[int]]`
- **Output:** `List[List[int]]` — List of `[r, c]` coordinates capable of reaching both oceans.
- **Constraints:**
  - $m == \text{heights.length}$
  - $n == \text{heights}[r].\text{length}$
  - $1 \le m, n \le 200$
  - $0 \le \text{heights}[r][c] \le 10^5$

---

### Key Idea & Intuition

- **The Forward Simulation Trap ($\mathcal{O}((MN)^2)$):**
  - Checking each cell individually by simulating downhill flow to see if it reaches both borders takes $\mathcal{O}(M \times N)$ per cell, resulting in an unacceptable $\mathcal{O}((MN)^2)$ time complexity.
- **The Reverse Uphill Strategy ($\mathcal{O}(MN)$):**
  - Invert the perspective: instead of asking *"can water flow downhill to the ocean?"*, ask:
    > *"Which cells can be reached by climbing **uphill** from the ocean shores?"*
  - An uphill move from `(r, c)` to neighbor `(nr, nc)` is valid if:
    $$\text{heights}[nr][nc] \ge \text{heights}[r][c]$$
  - **Two Independent Searches:**
    1. **Pacific Traversal:** Start multi-source DFS/BFS from all top-row cells `(0, c)` and left-column cells `(r, 0)`. Mark all reachable cells in `pacific_reachable`.
    2. **Atlantic Traversal:** Start multi-source DFS/BFS from all bottom-row cells `(m - 1, c)` and right-column cells `(r, n - 1)`. Mark all reachable cells in `atlantic_reachable`.
  - The answer is simply the **set intersection**:
    $$\text{Result} = \text{pacific\_reachable} \cap \text{atlantic\_reachable}$$

---

### Solution Approach (Step-by-Step)

1. Let $m = \text{len}(heights), n = \text{len}(heights[0])$.
2. Initialize two sets: `pacific = set()`, `atlantic = set()`.
3. Define helper `dfs(r, c, reachable)`:
   - Add `(r, c)` to `reachable`.
   - For each 4-directional neighbor `(nr, nc)`:
     - If $0 \le nr < m$ and $0 \le nc < n$ and `(nr, nc) not in reachable`:
       - If `heights[nr][nc] >= heights[r][c]`: (uphill flow condition)
         - `dfs(nr, nc, reachable)`
4. Launch Pacific DFS:
   - For all $c \in [0, n - 1]$: `dfs(0, c, pacific)`
   - For all $r \in [0, m - 1]$: `dfs(r, 0, pacific)`
5. Launch Atlantic DFS:
   - For all $c \in [0, n - 1]$: `dfs(m - 1, c, atlantic)`
   - For all $r \in [0, m - 1]$: `dfs(r, n - 1, atlantic)`
6. Return `[[r, c] for (r, c) in (pacific & atlantic)]`.

---

### Visual Algorithm Walkthrough

```
Grid:
Pacific ~   ~   ~   ~   ~
~  [1,  2,  2,  3, (5)] ~ Atlantic
~  [3,  2,  3, (4),(4)] ~ Atlantic
~  [2,  4, (5), 3,  1] ~ Atlantic
~  [(6),(7), 1,  4,  5] ~ Atlantic
~  [(5), 1,  1,  2,  4] ~ Atlantic
    ~   ~   ~   ~   ~ Atlantic

Pacific Uphill DFS:
Starts from top row and left col. Climbs to higher or equal neighbors.
Reaches: (0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0), etc.

Atlantic Uphill DFS:
Starts from bottom row and right col. Climbs to higher or equal neighbors.
Reaches: (0, 4), (1, 3), (1, 4), (2, 2), (3, 0), (3, 1), (4, 0), etc.

Intersection (cells marked in parentheses):
[[0,4], [1,3], [1,4], [2,2], [3,0], [3,1], [4,0]]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Island
- **Input:**
  ```python
  heights = [
    [1,2,2,3,5],
    [3,2,3,4,4],
    [2,4,5,3,1],
    [6,7,1,4,5],
    [5,1,1,2,4]
  ]
  ```
- **Overlap Cells:** `[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]`
- **Output:** `[[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]`

#### Example 2: Uniform Height Grid
- **Input:** `heights = [[1]]`
- **Analysis:** Single cell touches both Pacific and Atlantic borders simultaneously.
- **Output:** `[[0,0]]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
            
        m, n = len(heights), len(heights[0])
        pac_reachable = set()
        atl_reachable = set()
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        def dfs(r: int, c: int, visited: set) -> None:
            visited.add((r, c))
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and (nr, nc) not in visited:
                    # Water climbs uphill to higher or equal heights
                    if heights[nr][nc] >= heights[r][c]:
                        dfs(nr, nc, visited)
                        
        # Pacific borders: row 0 and col 0
        for c in range(n):
            dfs(0, c, pac_reachable)
        for r in range(m):
            dfs(r, 0, pac_reachable)
            
        # Atlantic borders: row m-1 and col n-1
        for c in range(n):
            dfs(m - 1, c, atl_reachable)
        for r in range(m):
            dfs(r, n - 1, atl_reachable)
            
        return [[r, c] for (r, c) in (pac_reachable & atl_reachable)]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> pacificAtlantic(std::vector<std::vector<int>>& heights) {
        if (heights.empty() || heights[0].empty()) return {};
        int m = heights.size(), n = heights[0].size();

        std::vector<std::vector<bool>> pac(m, std::vector<bool>(n, false));
        std::vector<std::vector<bool>> atl(m, std::vector<bool>(n, false));

        // Pacific borders
        for (int c = 0; c < n; ++c) dfs(0, c, pac, heights);
        for (int r = 0; r < m; ++r) dfs(r, 0, pac, heights);

        // Atlantic borders
        for (int c = 0; c < n; ++c) dfs(m - 1, c, atl, heights);
        for (int r = 0; r < m; ++r) dfs(r, n - 1, atl, heights);

        std::vector<std::vector<int>> result;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (pac[r][c] && atl[r][c]) {
                    result.push_back({r, c});
                }
            }
        }
        return result;
    }

private:
    int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

    void dfs(int r, int c, std::vector<std::vector<bool>>& visited, 
             const std::vector<std::vector<int>>& heights) {
        visited[r][c] = true;
        int m = heights.size(), n = heights[0].size();

        for (auto& d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc]) {
                if (heights[nr][nc] >= heights[r][c]) {
                    dfs(nr, nc, visited, heights);
                }
            }
        }
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    private int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    private int m, n;

    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        List<List<Integer>> result = new ArrayList<>();
        if (heights == null || heights.length == 0) return result;

        m = heights.length;
        n = heights[0].length;

        boolean[][] pac = new boolean[m][n];
        boolean[][] atl = new boolean[m][n];

        // Pacific borders (top row and left column)
        for (int c = 0; c < n; c++) dfs(0, c, pac, heights);
        for (int r = 0; r < m; r++) dfs(r, 0, pac, heights);

        // Atlantic borders (bottom row and right column)
        for (int c = 0; c < n; c++) dfs(m - 1, c, atl, heights);
        for (int r = 0; r < m; r++) dfs(r, n - 1, atl, heights);

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (pac[r][c] && atl[r][c]) {
                    result.add(Arrays.asList(r, c));
                }
            }
        }

        return result;
    }

    private void dfs(int r, int c, boolean[][] visited, int[][] heights) {
        visited[r][c] = true;

        for (int[] d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && !visited[nr][nc]) {
                if (heights[nr][nc] >= heights[r][c]) {
                    dfs(nr, nc, visited, heights);
                }
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \times N)$ — Each cell in the grid is visited at most once during the Pacific DFS and at most once during the Atlantic DFS. Finding the intersection takes $\mathcal{O}(M \times N)$. Total runtime is strictly linear $\mathcal{O}(M \times N)$.
- **Space Complexity:** $\mathcal{O}(M \times N)$ — Boolean arrays or sets storing visited cells take $\mathcal{O}(M \times N)$ space. In the worst case, the recursion stack depth is $\mathcal{O}(M \times N)$.

---

### Takeaway Pattern & Interview Traps

1. **Reverse Flow Condition ($\ge$):** In the original problem, water flows downhill (`neighbor <= current`). When traversing *uphill* from the ocean borders, the condition flips to `neighbor >= current`.
2. **Equal Heights Allow Flow:** If adjacent cells have the exact same height (`heights[nr][nc] == heights[r][c]`), water can flow between them. Using strict inequality ($>$) would fail on flat plateaus.
3. **Set Intersection vs Boolean Array:** In C++ and Java, 2D boolean matrices `pac[m][n]` and `atl[m][n]` have significantly better cache locality and lower memory overhead than hash sets.