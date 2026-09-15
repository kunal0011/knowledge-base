---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 317: Shortest Distance from All Buildings"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - matrix
  - amazon
  - google
---

# LeetCode 317: Shortest Distance from All Buildings

**Target Companies:** Google (All-Time Top #1 Signature Hard Graph Problem), Meta, Amazon, Microsoft  
**Difficulty:** Hard  
**Topic:** Multi-Source BFS / Distance Accumulation / Reachability Pruning / Matrix Graph Traversal

---

### Problem Statement

You are given an `m x n` grid `grid` of values `0`, `1`, or `2`, where:
- `0` represents an **empty land** that you can pass through freely.
- `1` represents a **building** that you cannot pass through.
- `2` represents an **obstacle** that you cannot pass through.

You want to build a single house on an empty land that reaches all buildings in the **shortest total travel distance**. You can only move up, down, left, and right.

Return the **minimum total travel distance** to all buildings. If it is not possible to build such a house according to the above rules, return `-1`.

The total travel distance is the sum of the distances from the houses of all the buildings to this empty land.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[int]]`
- **Output:** `int` — Minimum sum of distances to all buildings, or `-1`.
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 50$
  - `grid[i][j]` is either `0`, `1`, or `2`.
  - There will be **at least one building** in the grid.

---

### Key Idea & Intuition

- **Direction of Search (Building-to-Land vs Land-to-Building):**
  - Starting BFS from every empty land cell causes redundant traversals across the grid, resulting in Time Limit Exceeded ($\mathcal{O}((MN)^2)$).
  - Instead, initiate BFS from each **building (`1`)**:
    - The number of buildings $B$ is typically much smaller than empty cells.
    - Each building's BFS computes the shortest distance to all accessible empty land cells.
- **Two Tracking Metrics:**
  - `total_dist[r][c]`: Cumulative sum of shortest distances from all buildings to empty cell `(r, c)`.
  - `reach_count[r][c]`: Count of distinct buildings that can reach cell `(r, c)`.
- **Crucial Reachability Pruning:**
  - When processing the $k$-th building ($0$-indexed):
    - An empty cell `(nr, nc)` is only worth expanding if it was reached by **all previous $k$ buildings** (i.e. `reach_count[nr][nc] == k`).
    - If a cell was disconnected from building 0, it can never be a valid solution candidate for all buildings! Pruning this immediately avoids traversing dead zones.
- **Final Result Extraction:**
  - Scan all empty lands: find $\min(\text{total\_dist}[r][c])$ among cells where $\text{reach\_count}[r][c] == \text{total\_buildings}$.
  - If no cell meets this condition, return `-1`.

---

### Solution Approach (Step-by-Step)

1. Let $m = \text{len}(grid), n = \text{len}(grid[0])$.
2. Count total buildings $B$ and record their coordinates.
3. Initialize `total_dist = [[0] * n for _ in range(m)]` and `reach_count = [[0] * n for _ in range(m)]`.
4. For each building index $k$ at `(br, bc)`:
   - Run BFS from `(br, bc)`:
     - `queue = deque([(br, bc, 0)])`
     - Maintain a `visited` set for the current building.
     - While `queue` is not empty:
       - Pop `r, c, d`.
       - For each 4-directional neighbor `(nr, nc)`:
         - If $0 \le nr < m$ and $0 \le nc < n$ and `grid[nr][nc] == 0`:
           - If `(nr, nc)` not in `visited` and `reach_count[nr][nc] == k`:
             - `visited.add((nr, nc))`
             - `reach_count[nr][nc] += 1`
             - `total_dist[nr][nc] += d + 1`
             - `queue.append((nr, nc, d + 1))`
5. Find `min_dist = min(total_dist[r][c])` for all $(r, c)$ with `reach_count[r][c] == B`.
6. Return `min_dist` if valid candidates exist, else `-1`.

---

### Visual Algorithm Walkthrough

```
Grid:
1  0  2  0  1
0  0  0  0  0
0  0  1  0  0

Buildings at: (0, 0), (0, 4), (2, 2) -> Total = 3

Building 1 at (0, 0):
BFS expands distances:
  - (0, 1): dist 1, reach 1
  - (1, 0): dist 1, reach 1
  - (1, 1): dist 2, reach 1
  - (1, 2): dist 3, reach 1 ...

Building 2 at (0, 4):
BFS expands only to cells with reach == 1:
  - (0, 3): dist 1, reach 2
  - (1, 4): dist 1, reach 2
  - (1, 3): dist 2, reach 2
  - (1, 2): dist 3, total_dist += 3 -> reach 2 ...

Building 3 at (2, 2):
BFS expands only to cells with reach == 2:
  - (1, 2): dist 1, reach 3 (All 3 buildings reached!)
    total_dist[1][2] = 3 (from b1) + 3 (from b2) + 1 (from b3) = 7!

Best empty land is at (1, 2) with total travel distance = 7.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Reachable Grid
- **Input:**
  ```python
  grid = [
    [1, 0, 2, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0]
  ]
  ```
- **Analysis:** Cell `(1, 2)` can reach `(0, 0)` in 3 steps, `(0, 4)` in 3 steps, and `(2, 2)` in 1 step.
- **Total Distance:** $3 + 3 + 1 = 7$.
- **Output:** `7`

#### Example 2: Blocked Building (Impossible)
- **Input:**
  ```python
  grid = [
    [1, 2, 0],
    [2, 2, 0],
    [0, 0, 1]
  ]
  ```
- **Analysis:** Building `(0, 0)` is completely boxed in by obstacles (`2`). No empty cell can reach all buildings.
- **Output:** `-1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return -1
            
        m, n = len(grid), len(grid[0])
        total_dist = [[0] * n for _ in range(m)]
        reach_count = [[0] * n for _ in range(m)]
        buildings = []
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1:
                    buildings.append((r, c))
                    
        total_buildings = len(buildings)
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for k, (br, bc) in enumerate(buildings):
            queue = deque([(br, bc, 0)])
            visited = set()
            
            while queue:
                r, c, d = queue.popleft()
                
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 0:
                        # Only explore if reached by all previous k buildings
                        if (nr, nc) not in visited and reach_count[nr][nc] == k:
                            visited.add((nr, nc))
                            reach_count[nr][nc] += 1
                            total_dist[nr][nc] += d + 1
                            queue.append((nr, nc, d + 1))
                            
        min_dist = float('inf')
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 0 and reach_count[r][c] == total_buildings:
                    min_dist = min(min_dist, total_dist[r][c])
                    
        return min_dist if min_dist != float('inf') else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

class Solution {
public:
    int shortestDistance(std::vector<std::vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        std::vector<std::vector<int>> totalDist(m, std::vector<int>(n, 0));
        std::vector<std::vector<int>> reachCount(m, std::vector<int>(n, 0));

        std::vector<std::pair<int, int>> buildings;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (grid[r][c] == 1) {
                    buildings.push_back({r, c});
                }
            }
        }

        int totalBuildings = buildings.size();
        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        for (int k = 0; k < totalBuildings; ++k) {
            auto [br, bc] = buildings[k];
            std::queue<std::tuple<int, int, int>> q;
            std::vector<std::vector<bool>> visited(m, std::vector<bool>(n, false));

            q.push({br, bc, 0});

            while (!q.empty()) {
                auto [r, c, d] = q.front();
                q.pop();

                for (auto& dir : dirs) {
                    int nr = r + dir[0], nc = c + dir[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 0) {
                        if (!visited[nr][nc] && reachCount[nr][nc] == k) {
                            visited[nr][nc] = true;
                            reachCount[nr][nc]++;
                            totalDist[nr][nc] += d + 1;
                            q.push({nr, nc, d + 1});
                        }
                    }
                }
            }
        }

        int minDist = INT_MAX;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                if (grid[r][c] == 0 && reachCount[r][c] == totalBuildings) {
                    minDist = std::min(minDist, totalDist[r][c]);
                }
            }
        }

        return minDist == INT_MAX ? -1 : minDist;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int shortestDistance(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] totalDist = new int[m][n];
        int[][] reachCount = new int[m][n];

        List<int[]> buildings = new ArrayList<>();
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 1) {
                    buildings.add(new int[]{r, c});
                }
            }
        }

        int totalBuildings = buildings.size();
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        for (int k = 0; k < totalBuildings; k++) {
            int[] b = buildings.get(k);
            Queue<int[]> queue = new ArrayDeque<>();
            boolean[][] visited = new boolean[m][n];

            queue.offer(new int[]{b[0], b[1], 0});

            while (!queue.isEmpty()) {
                int[] curr = queue.poll();
                int r = curr[0], c = curr[1], d = curr[2];

                for (int[] dir : dirs) {
                    int nr = r + dir[0], nc = c + dir[1];
                    if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 0) {
                        if (!visited[nr][nc] && reachCount[nr][nc] == k) {
                            visited[nr][nc] = true;
                            reachCount[nr][nc]++;
                            totalDist[nr][nc] += d + 1;
                            queue.offer(new int[]{nr, nc, d + 1});
                        }
                    }
                }
            }
        }

        int minDist = Integer.MAX_VALUE;
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (grid[r][c] == 0 && reachCount[r][c] == totalBuildings) {
                    minDist = Math.min(minDist, totalDist[r][c]);
                }
            }
        }

        return minDist == Integer.MAX_VALUE ? -1 : minDist;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(B \times M \times N)$ where $B$ is the number of buildings and $M \times N$ is the grid size. In the worst case $B \le M \times N \implies \mathcal{O}((MN)^2)$. With $M, N \le 50$, $MN \le 2500$; the pruning condition `reachCount[nr][nc] == k` rapidly cuts down the search space in subsequent iterations.
- **Space Complexity:** $\mathcal{O}(M \times N)$ for the `totalDist` and `reachCount` matrices and the BFS queue.

---

### Takeaway Pattern & Interview Traps

1. **Why `reach_count[nr][nc] == k` is crucial:** Without this pruning check, every building searches all reachable cells independently. With this check, cells that cannot reach earlier buildings are discarded immediately, saving massive amounts of compute time and avoiding TLE.
2. **Passing Through Buildings:** Remember that buildings cannot be walked through. A path cannot pass through cell values `1` or `2`.
3. **Disconnected Return Value:** If no empty land can reach all buildings (e.g. islands of buildings separated by water/obstacles), returning `-1` is mandatory.