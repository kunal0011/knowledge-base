---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 200: Number of Islands"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - dfs
  - amazon
  - google
---

# LeetCode 200: Number of Islands

**Target Companies:** Amazon (Top #1 Graph Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** BFS / DFS Flood Fill Connected Components

---

### Problem Statement

Given an $m \times n$ 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the **number of islands**.

An **island** is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[str]]` where `grid[r][c]` is `'1'` or `'0'`.
- **Output:** `int` (count of distinct connected components).
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 300$
  - $\text{grid}[i][j]$ is `'0'` or `'1'`.

---

### Key Idea & Intuition

- **Connected Component Count:**
  - Each island corresponds to a maximal connected component of `'1'`s under 4-directional adjacency.
- **Flood Fill Traversal:**
  - Scan the grid cell by cell `(r, c)`.
  - When encountering unvisited land (`grid[r][c] == '1'`):
    - Increment `island_count += 1`.
    - Trigger a BFS or DFS flood-fill starting at `(r, c)` to mark all reachable land cells as visited (`'0'`).
    - Mutating the grid in-place from `'1'` to `'0'` eliminates the need for an auxiliary `visited` set.

---

### Solution Approach (Step-by-Step - BFS)

1. Check edge cases: if not `grid` or not `grid[0]`, return `0`.
2. Let `m = len(grid)`, `n = len(grid[0])`, `islands = 0`.
3. For `r` in range $m$:
   - For `c` in range $n$:
     - If `grid[r][c] == '1'`:
       - `islands += 1`.
       - `grid[r][c] = '0'` (mark visited immediately).
       - Initialize queue with `(r, c)`.
       - While queue is not empty:
         - `row, col = queue.popleft()`.
         - For `(dr, dc)` in `[(0, 1), (0, -1), (1, 0), (-1, 0)]`:
           - `nr, nc = row + dr, col + dc`.
           - If $0 \le nr < m$ and $0 \le nc < n$ and `grid[nr][nc] == '1'`:
             - `grid[nr][nc] = '0'` (mark visited upon enqueueing).
             - `queue.append((nr, nc))`.
4. Return `islands`.

---

### Visual Algorithm Walkthrough

```
Grid:
1 1 0 0 0
1 1 0 0 0
0 0 1 0 0
0 0 0 1 1

Encounter (0, 0) == '1':
Flood fill island 1 -> marks (0,0), (0,1), (1,0), (1,1) as '0'
Grid becomes:
0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 1 1

Encounter (2, 2) == '1':
Flood fill island 2 -> marks (2, 2) as '0'

Encounter (3, 3) == '1':
Flood fill island 3 -> marks (3, 3), (3, 4) as '0'

Total Islands = 3
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
            
        m, n = len(grid), len(grid[0])
        islands = 0
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    islands += 1
                    grid[r][c] = '0'
                    queue = deque([(r, c)])
                    
                    while queue:
                        row, col = queue.popleft()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                                grid[nr][nc] = '0'
                                queue.append((nr, nc))
                                
        return islands
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    int numIslands(std::vector<std::vector<char>>& grid) {
        if (grid.empty()) return 0;
        int m = grid.size(), n = grid[0].size();
        int islands = 0;
        int dirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (grid[r][c] == '1') {
                    islands++;
                    grid[r][c] = '0';
                    std::queue<std::pair<int, int>> q;
                    q.push({r, c});

                    while (!q.empty()) {
                        auto [row, col] = q.front();
                        q.pop();

                        for (auto& d : dirs) {
                            int nr = row + d[0], nc = col + d[1];
                            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == '1') {
                                grid[nr][nc] = '0';
                                q.push({nr, nc});
                            }
                        }
                    }
                }
            }
        }
        return islands;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) return 0;
        int m = grid.length, n = grid[0].length;
        int islands = 0;
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == '1') {
                    islands++;
                    grid[r][c] = '0';
                    Queue<int[]> q = new ArrayDeque<>();
                    q.offer(new int[]{r, c});

                    while (!q.isEmpty()) {
                        int[] curr = q.poll();
                        for (int[] d : dirs) {
                            int nr = curr[0] + d[0], nc = curr[1] + d[1];
                            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == '1') {
                                grid[nr][nc] = '0';
                                q.offer(new int[]{nr, nc});
                            }
                        }
                    }
                }
            }
        }
        return islands;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(M \times N)$ — Every cell is checked once by the outer loop, and every `'1'` is enqueued and dequeued at most once.
- **Space Complexity:** $O(\min(M, N))$ — Max queue memory in BFS on an $M \times N$ grid is proportional to the diagonal.
