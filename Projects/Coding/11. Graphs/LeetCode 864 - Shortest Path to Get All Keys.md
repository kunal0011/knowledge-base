---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 864: Shortest Path to Get All Keys"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 864: Shortest Path to Get All Keys

**Target Companies:** Google (Top Signature Hard), Amazon | **Algorithm:** State-Space Graph BFS with Bitmask Key Ring

---

### Problem Statement

Given an `m x n` grid with '@' (start), '.' (empty), '#' (wall), 'a'-'f' (keys), and 'A'-'F' (locks). Return lowest steps to collect all keys.

---

### Key Observation

* Revisiting the same coordinate `(r, c)` is allowed if and only if we have collected a **different set of keys**.
* Represent collected keys as an integer Bitmask (e.g. key 'a' is bit 0, 'b' is bit 1).
* State in BFS: `(r, c, key_mask)`.
* Lock 'A' can be opened if `key_mask & (1 << 0) != 0`.
* Picking key 'a' updates `next_mask = key_mask | (1 << 0)`.

---

### Core Algorithm & Technique: Multi-Dimensional State-Space BFS `(row, col, key\_mask)`

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        m, n = len(grid), len(grid[0])
        start_r = start_c = total_keys = 0
        
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '@':
                    start_r, start_c = r, c
                elif grid[r][c].islower():
                    total_keys += 1
                    
        target_mask = (1 << total_keys) - 1
        queue = deque([(start_r, start_c, 0, 0)])  # (r, c, key_mask, steps)
        visited = {(start_r, start_c, 0)}
        
        while queue:
            r, c, mask, steps = queue.popleft()
            
            if mask == target_mask:
                return steps
                
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    cell = grid[nr][nc]
                    if cell == '#':
                        continue
                    
                    next_mask = mask
                    if cell.islower():
                        next_mask |= (1 << (ord(cell) - ord('a')))
                    elif cell.isupper():
                        # Lock: check if corresponding key bit is set
                        if not (mask & (1 << (ord(cell) - ord('A')))):
                            continue
                            
                    if (nr, nc, next_mask) not in visited:
                        visited.add((nr, nc, next_mask))
                        queue.append((nr, nc, next_mask, steps + 1))
                        
        return -1
```

---

### Worked-Out Example

```python
grid = ["@.a..","###.#","b.A.B"]
Keys = 2 ('a', 'b'), target_mask = 11_2 (3)
Step 1: Move from '@' -> collect key 'a' (mask = 01_2)
Step 2: Walk past Lock 'A' (opened because key 'a' bit is set) -> collect key 'b' (mask = 11_2)
Steps = 8
```

---

### Complexity Analysis

* **Time Complexity:** `O(m * n * 2^K) where K is number of keys (at most 6)`
* **Space Complexity:** `O(m * n * 2^K)`

---

### Takeaway Pattern

When cells have conditional re-entry permissions, expand the 2D search space into 3D state `(r, c, bitmask)`.