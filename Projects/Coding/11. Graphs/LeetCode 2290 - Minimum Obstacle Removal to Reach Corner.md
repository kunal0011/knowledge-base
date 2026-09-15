---
date: "2026-09-15"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 2290: Minimum Obstacle Removal to Reach Corner"
tags:
  - leetcode
  - coding
  - graphs
  - bfs
  - 0-1-bfs
  - deque
  - amazon
  - google
---

# LeetCode 2290: Minimum Obstacle Removal to Reach Corner

**Target Companies:** Google (Signature 0-1 BFS Classic), Amazon  
**Difficulty:** Hard  
**Topic:** 0-1 Breadth-First Search (Deque) / Binary Weighted Shortest Path / Grid Traversal

---

### Problem Statement

You are given a **0-indexed** 2D integer array `grid` of size `m x n`. Each cell has one of two values:
- `0` represents an **empty** cell,
- `1` represents an **obstacle** that may be removed.

You can move up, down, left, or right to and from an empty cell.

Return the **minimum number of obstacles** to remove to move from the upper left corner `(0, 0)` to the lower right corner `(m - 1, n - 1)`.

---

### Input & Output Formats & Constraints

- **Input:** `grid: List[List[int]]`
- **Output:** `int` — Minimum number of obstacle removals.
- **Constraints:**
  - $m == \text{grid.length}$
  - $n == \text{grid}[i].\text{length}$
  - $1 \le m, n \le 10^5$
  - $2 \le m \times n \le 10^5$
  - `grid[i][j]` is either `0` or `1`.
  - `grid[0][0] == grid[m - 1][n - 1] == 0`

---

### Key Idea & Intuition

- **Shortest Path with Binary Edge Weights ($0$ and $1$):**
  - Moving to an adjacent cell `(nr, nc)` incurs an edge cost equal to `grid[nr][nc]`:
    - Cost is $0$ if the next cell is empty (`grid[nr][nc] == 0`).
    - Cost is $1$ if the next cell has an obstacle (`grid[nr][nc] == 1`).
  - Standard Dijkstra’s algorithm solves this in $\mathcal{O}(MN \log(MN))$ using a Min-Heap.
- **The 0-1 BFS Optimization ($\mathcal{O}(MN)$ Linear Time):**
  - Because edge weights are strictly bounded to $\{0, 1\}$, a **Double-Ended Queue (Deque)** can maintain monotonic distance ordering without heap operations:
    - **Weight 0 edge:** Push to the **front** of the deque (`appendleft`). The destination cell has the *same* cumulative distance as the current cell and should be explored immediately!
    - **Weight 1 edge:** Push to the **back** of the deque (`append`). The destination cell has cumulative distance $+1$ and will be explored after all current-tier cells.
  - This ensures nodes are popped from the deque in strictly non-decreasing order of cost, guaranteeing that the first time `(m - 1, n - 1)` is popped, its distance is minimal.

---

### Solution Approach (Step-by-Step)

1. Let $m = \text{len}(grid), n = \text{len}(grid[0])$.
2. Initialize 2D array `dist` of dimensions $m \times n$ filled with $\infty$. Set `dist[0][0] = 0`.
3. Initialize double-ended queue `q = deque([(0, 0, 0)])` storing `(cost, r, c)`.
4. While `q` is not empty:
   - Pop front: `cost, r, c = q.popleft()`.
   - If `r == m - 1 and c == n - 1`: return `cost`.
   - If `cost > dist[r][c]`: continue (stale entry).
   - For `(dr, dc)` in `[(0, 1), (0, -1), (1, 0), (-1, 0)]`:
     - Let `nr = r + dr, nc = c + dc`.
     - If $0 \le nr < m$ and $0 \le nc < n$:
       - `new_cost = cost + grid[nr][nc]`.
       - If `new_cost < dist[nr][nc]`:
         - `dist[nr][nc] = new_cost`.
         - If `grid[nr][nc] == 0`: `q.appendleft((new_cost, nr, nc))`.
         - Else: `q.append((new_cost, nr, nc))`.
5. Return `dist[m - 1][n - 1]`.

---

### Visual Algorithm Walkthrough

```
Grid (3x3):
0 1 1
1 1 0
1 1 0

Start at (0, 0), dist = 0.

Deque Evolution:
1. Pop (cost=0, 0, 0):
   - To (0, 1): obstacle (cost=1) -> push back: Deque: [(1, 0, 1)]
   - To (1, 0): obstacle (cost=1) -> push back: Deque: [(1, 0, 1), (1, 1, 0)]

2. Pop (cost=1, 0, 1):
   - To (0, 2): obstacle (cost=2) -> push back: Deque: [(1, 1, 0), (2, 0, 2)]
   - To (1, 1): obstacle (cost=2) -> push back: Deque: [(1, 1, 0), (2, 0, 2), (2, 1, 1)]

3. Pop (cost=1, 1, 0):
   - To (2, 0): obstacle (cost=2) -> push back

...

Eventually reaches (1, 2) which is 0:
- From (1, 2) to target (2, 2) is empty (cost 0)!
- Pushed to FRONT of deque!
- Explored before any cost=3 nodes.
Minimum obstacles removed = 2.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Grid
- **Input:**
  ```python
  grid = [
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 0]
  ]
  ```
- **Optimal Path:** $(0,0) \to (0,1) \text{ [obs 1]} \to (1,1) \text{ [obs 2]} \to (1,2) \text{ [free]} \to (2,2) \text{ [free]}$.
- **Total Obstacles:** `2`
- **Output:** `2`

#### Example 2: No Obstacles
- **Input:**
  ```python
  grid = [
    [0, 0],
    [0, 0]
  ]
  ```
- **Output:** `0` (Only cost 0 edges pushed to front, reaches corner immediately).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = 0
        
        q = deque([(0, 0, 0)])  # (cost, r, c)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while q:
            cost, r, c = q.popleft()
            
            if r == m - 1 and c == n - 1:
                return cost
                
            if cost > dist[r][c]:
                continue
                
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    new_cost = cost + grid[nr][nc]
                    if new_cost < dist[nr][nc]:
                        dist[nr][nc] = new_cost
                        if grid[nr][nc] == 0:
                            q.appendleft((new_cost, nr, nc))
                        else:
                            q.append((new_cost, nr, nc))
                            
        return dist[m - 1][n - 1]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>
#include <climits>

class Solution {
public:
    int minimumObstacles(std::vector<std::vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        std::vector<std::vector<int>> dist(m, std::vector<int>(n, INT_MAX));
        dist[0][0] = 0;

        std::deque<std::pair<int, int>> dq;
        dq.push_front({0, 0});

        int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!dq.empty()) {
            auto [r, c] = dq.front();
            dq.pop_front();

            if (r == m - 1 && c == n - 1) {
                return dist[r][c];
            }

            for (auto& d : dirs) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                    int newCost = dist[r][c] + grid[nr][nc];
                    if (newCost < dist[nr][nc]) {
                        dist[nr][nc] = newCost;
                        if (grid[nr][nc] == 0) {
                            dq.push_front({nr, nc});
                        } else {
                            dq.push_back({nr, nc});
                        }
                    }
                }
            }
        }

        return dist[m - 1][n - 1];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

class Solution {
    public int minimumObstacles(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dist = new int[m][n];
        for (int[] row : dist) {
            Arrays.fill(row, Integer.MAX_VALUE);
        }
        dist[0][0] = 0;

        Deque<int[]> dq = new ArrayDeque<>();
        dq.offerFirst(new int[]{0, 0});

        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

        while (!dq.isEmpty()) {
            int[] curr = dq.pollFirst();
            int r = curr[0], c = curr[1];

            if (r == m - 1 && c == n - 1) {
                return dist[r][c];
            }

            for (int[] d : dirs) {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < m && nc >= 0 && nc < n) {
                    int newCost = dist[r][c] + grid[nr][nc];
                    if (newCost < dist[nr][nc]) {
                        dist[nr][nc] = newCost;
                        if (grid[nr][nc] == 0) {
                            dq.offerFirst(new int[]{nr, nc});
                        } else {
                            dq.offerLast(new int[]{nr, nc});
                        }
                    }
                }
            }
        }

        return dist[m - 1][n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \times N)$ — In 0-1 BFS, each grid cell is added and removed from the deque at most twice. Since operations on both ends of a deque take $\mathcal{O}(1)$, the total time is strictly linear in the number of cells $\mathcal{O}(M \times N)$, avoiding the $\log(M \times N)$ factor of priority queues.
- **Space Complexity:** $\mathcal{O}(M \times N)$ — The 2D `dist` table and the double-ended queue each hold up to $M \times N$ elements.

---

### Takeaway Pattern & Interview Traps

1. **When to use 0-1 BFS over Dijkstra:** Whenever edge weights are restricted to binary values $\{0, 1\}$ (or $\{0, W\}$), 0-1 BFS using a Deque achieves $\mathcal{O}(V + E)$ linear time, strictly outperforming Dijkstra's $\mathcal{O}((V + E) \log V)$.
2. **Push Order Rule:**
   - Weight 0 edge $\implies$ `push_front` (or `appendleft`).
   - Weight 1 edge $\implies$ `push_back` (or `append`).
3. **Distance Relaxation Guard:** Always verify `new_cost < dist[nr][nc]` before pushing into the deque to prevent redundant states from cluttering the queue.