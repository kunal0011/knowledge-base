---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 994: Rotting Oranges"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 994: Rotting Oranges

---

### Problem Statement

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten. Return the minimum number of minutes that must elapse until no cell has a fresh orange (-1 if impossible).

---

### Key Observation

* All initially rotten oranges rot their neighbors simultaneously at time `t = 0`.
* This requires **Multi-Source BFS** where all sources start in the queue at level 0.

---

### Core Technique: Multi-Source Level-Order BFS

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        queue = deque()
        fresh = 0
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1
                    
        if fresh == 0:
            return 0
            
        minutes = 0
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        while queue and fresh > 0:
            minutes += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
                        
        return minutes if fresh == 0 else -1
```

---

### Worked-Out Example

```python
grid = [[2,1,1],[1,1,0],[0,1,1]]
queue = [(0,0)], fresh = 6
minute 1: (0,1) and (1,0) rot -> fresh = 4
minute 2: (0,2) and (1,1) rot -> fresh = 2
minute 3: (2,1) rots -> fresh = 1
minute 4: (2,2) rots -> fresh = 0
Result = 4 minutes
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n)`
* **Space Complexity:** `O(m * n)`

---

### Takeaway Pattern

Add all starting points to queue before initiating BFS to achieve uniform multi-source expansion.