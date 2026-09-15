---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 994: Rotting Oranges"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - matrix
  - amazon
  - google
---

# LeetCode 994: Rotting Oranges

**Target Companies:** Amazon (All-Time #1 Most Frequently Asked Graph Problem), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Multi-Source Breadth-First Search (BFS) / Level-Order Propagation / Matrix Simulation

---

### Problem Statement

You are given an `m x n` `grid` where each cell can have one of three values:
- `0` representing an empty cell,
- `1` representing a fresh orange, or
- `2` representing a rotten orange.

Every minute, any fresh orange that is **4-directionally adjacent** to a rotten orange becomes rotten.

Return the **minimum number of minutes** that must elapse until no cell has a fresh orange. If this is impossible, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[int]]`
- **Output:** `int` — Minimum minutes elapsed, or `-1`.
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 10$
  - `grid[i][j]` is `0`, `1`, or `2`.

---

### Key Idea & Intuition

- **Simultaneous Contagion (Multi-Source BFS):**
  - All initially rotten oranges spread rot at the exact same time ($t = 0$).
  - Starting a separate single-source BFS from each rotten orange would simulate independent timelines and complicate timestamp synchronization.
  - Instead, **Multi-Source BFS** enqueues all initially rotten oranges together into a shared queue at level $0$.
- **Level-by-Level Minute Tick:**
  - Each iteration of the outer BFS loop represents exactly **one minute of elapsed time**.
  - During that minute, all rotten oranges at the current frontier rot their 4-directional fresh neighbors.
  - Newly rotted oranges are marked `grid[nr][nc] = 2` (preventing double counting) and enqueued for the next minute.
- **Termination Guard (`fresh > 0`):**
  - Track the count of remaining fresh oranges `fresh`.
  - If `fresh == 0` at the very start, return `0` (no time needed).
  - Only increment `minutes` if `fresh > 0` before processing the level. This avoids adding an extra minute when the last batch of rotten oranges has no fresh neighbors left to infect.

---

### Solution Approach (Step-by-Step)

1. Let $m = \text{len}(grid), n = \text{len}(grid[0])$.
2. Scan the grid:
   - Count initial `fresh` oranges.
   - Enqueue all coordinates `(r, c)` where `grid[r][c] == 2`.
3. If `fresh == 0`: return `0`.
4. Initialize `minutes = 0`.
5. While `queue` is not empty and `fresh > 0`:
   - `minutes += 1`.
   - `level_size = len(queue)`.
   - For `_` in range(`level_size`):
     - Pop `r, c = queue.popleft()`.
     - For `dr, dc` in `[(0, 1), (0, -1), (1, 0), (-1, 0)]`:
       - `nr, nc = r + dr, c + dc`.
       - If $0 \le nr < m$ and $0 \le nc < n$ and `grid[nr][nc] == 1`:
         - `grid[nr][nc] = 2` (mark rotted).
         - `fresh -= 1`.
         - `queue.append((nr, nc))`.
6. Return `minutes` if `fresh == 0` else `-1`.

---

### Visual Algorithm Walkthrough

```
Initial Grid (t = 0):
2  1  1
1  1  0
0  1  1

Initial Queue: [(0, 0)]
Fresh Oranges: 6

Minute 1 (t = 1):
Pop (0, 0) -> infects (0, 1) and (1, 0)
Grid:
2  2  1
2  1  0
0  1  1
Queue: [(0, 1), (1, 0)], fresh = 4

Minute 2 (t = 2):
Pop (0, 1) -> infects (0, 2)
Pop (1, 0) -> infects (1, 1)
Grid:
2  2  2
2  2  0
0  1  1
Queue: [(0, 2), (1, 1)], fresh = 2

Minute 3 (t = 3):
Pop (1, 1) -> infects (2, 1)
Grid:
2  2  2
2  2  0
0  2  1
Queue: [(2, 1)], fresh = 1

Minute 4 (t = 4):
Pop (2, 1) -> infects (2, 2)
Grid:
2  2  2
2  2  0
0  2  2
Queue: [(2, 2)], fresh = 0

Fresh == 0! Loop terminates.
Total Minutes = 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Rotting Progression
- **Input:** `grid = [[2,1,1],[1,1,0],[0,1,1]]`
- **Output:** `4`

#### Example 2: Disconnected Fresh Orange (Impossible)
- **Input:** `grid = [[2,1,1],[0,1,1],[1,0,1]]`
- **Analysis:** The fresh orange at `(2, 0)` is completely blocked by `0`s and can never be reached by rot.
- **Output:** `-1`

#### Example 3: No Fresh Oranges Initially
- **Input:** `grid = [[0,2]]`
- **Analysis:** `fresh == 0` at $t = 0$.
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        
        # 1. Enqueue all initial rotten oranges and count fresh ones
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
                    
        if fresh == 0:
            return 0
            
        minutes = 0
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # 2. Multi-source BFS
        while queue and fresh > 0:
            minutes += 1
            level_size = len(queue)
            
            for _ in range(level_size):
                r, c = queue.popleft()
                
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                        grid[nr][nc] = 2  # rot the orange
                        fresh -= 1
                        queue.append((nr, nc))
                        
        return minutes if fresh == 0 else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

class Solution {
public:
    int orangesRotting(std::vector<std::vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        std::queue<std::pair<int, int>> q;
        int fresh = 0;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (grid[r][c] == 2) {
                    q.push({r, c});
                } else if (grid[r][c] == 1) {
                    fresh++;
                }
            }
        }

        if (fresh == 0) return 0;

        int minutes = 0;
        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!q.empty() && fresh > 0) {
            minutes++;
            int levelSize = q.size();

            for (int i = 0; i < levelSize; ++i) {
                auto [r, c] = q.front();
                q.pop();

                for (auto& d : dirs) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 1) {
                        grid[nr][nc] = 2;
                        fresh--;
                        q.push({nr, nc});
                    }
                }
            }
        }

        return fresh == 0 ? minutes : -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

class Solution {
    public int orangesRotting(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        Queue<int[]> queue = new ArrayDeque<>();
        int fresh = 0;

        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 2) {
                    queue.offer(new int[]{r, c});
                } else if (grid[r][c] == 1) {
                    fresh++;
                }
            }
        }

        if (fresh == 0) {
            return 0;
        }

        int minutes = 0;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!queue.isEmpty() && fresh > 0) {
            minutes++;
            int levelSize = queue.size();

            for (int i = 0; i < levelSize; i++) {
                int[] curr = queue.poll();
                int r = curr[0], c = curr[1];

                for (int[] d : dirs) {
                    int nr = r + d[0], nc = c + d[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 1) {
                        grid[nr][nc] = 2;
                        fresh--;
                        queue.offer(new int[]{nr, nc});
                    }
                }
            }
        }

        return fresh == 0 ? minutes : -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \times N)$ — In the initial pass, every cell is scanned once. In the multi-source BFS, every cell is enqueued and dequeued at most once. Each move checks 4 directions in $\mathcal{O}(1)$ time. Overall time is strictly linear $\mathcal{O}(M \times N)$.
- **Space Complexity:** $\mathcal{O}(M \times N)$ — In the worst-case scenario (e.g. all oranges are rotten), the BFS queue holds up to $M \times N$ cells.

---

### Takeaway Pattern & Interview Traps

1. **The Overcounting Minute Trap:** If you increment minutes simply while `queue` is not empty (`while queue:`), the final pop of the newly rotted oranges will increment `minutes` by 1 even when they have no fresh neighbors to rot! Using `while queue and fresh > 0:` cleanly prevents this off-by-one error.
2. **Immediate State Mutation:** Changing `grid[nr][nc] = 2` upon enqueuing (not popping) ensures no orange is enqueued more than once by different neighboring rotten oranges.