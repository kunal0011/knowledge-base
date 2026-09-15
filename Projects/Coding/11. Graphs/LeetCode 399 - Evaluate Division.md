---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 399: Evaluate Division"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 399: Evaluate Division

---

### Problem Statement

Given equations like `A / B = 2.0`, `B / C = 3.0`, evaluate queries like `A / C = ?`.

---

### Key Observation

* Model as a Directed Weighted Graph where edge `A -> B` has weight `val`, and reciprocal edge `B -> A` has weight `1 / val`.
* Query `src / dst` is equivalent to finding path from `src` to `dst` and multiplying edge weights along the path.
* Use DFS / BFS or Weighted Union-Find.

---

### Core Technique: Weighted Graph Path Product Search via DFS

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        adj = defaultdict(dict)
        for (u, v), val in zip(equations, values):
            adj[u][v] = val
            adj[v][u] = 1.0 / val
            
        def dfs(curr, target, visited):
            if curr not in adj or target not in adj:
                return -1.0
            if curr == target:
                return 1.0
            visited.add(curr)
            for neighbor, weight in adj[curr].items():
                if neighbor not in visited:
                    prod = dfs(neighbor, target, visited)
                    if prod != -1.0:
                        return weight * prod
            return -1.0
            
        res = []
        for src, dst in queries:
            res.append(dfs(src, dst, set()))
        return res
```

---

### Worked-Out Example

```python
equations = [["a","b"],["b","c"]], values = [2.0, 3.0]
query: ["a", "c"]
Path a -> b (weight 2.0), b -> c (weight 3.0) -> total = 2.0 * 3.0 = 6.0
```

---

### Complexity Analysis

* **Time Complexity:** `O(Q * (V + E)) where Q is number of queries`
* **Space Complexity:** `O(V + E)`

---

### Takeaway Pattern

Divisions form reciprocal weighted graphs where path evaluation equals edge multiplication.