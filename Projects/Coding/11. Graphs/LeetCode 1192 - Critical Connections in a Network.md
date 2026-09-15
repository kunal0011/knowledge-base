---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 1192: Critical Connections in a Network"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 1192: Critical Connections in a Network

**Target Companies:** Amazon (Signature Hard), Google | **Algorithm:** Tarjan's Bridge-Finding Algorithm (DFS Low-Link Values)

---

### Problem Statement

There are `n` servers numbered `0` to `n - 1` and a list of `connections`. A critical connection (bridge) is a connection that, if removed, makes some servers unable to reach other servers. Return all critical connections.

---

### Key Observation

* An edge `(u, v)` is a bridge if and only if there is **NO back-edge** connecting `v` or any of its descendants back to `u` or an ancestor of `u`.
* Track two values for each node during DFS:
* 1. `disc[u]`: discovery time (timestamp when node `u` was first visited).
* 2. `low[u]`: lowest discovery time reachable from `u` using at most one back-edge.
* Condition for bridge: if `low[v] > disc[u]`, then `(u, v)` is a critical connection!

---

### Core Algorithm & Technique: Tarjan's Bridge Finding via DFS Discovery and Low-Link Values

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        adj = defaultdict(list)
        for u, v in connections:
            adj[u].append(v)
            adj[v].append(u)
            
        disc = [-1] * n
        low = [-1] * n
        bridges = []
        time = 0
        
        def dfs(u: int, parent: int):
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            
            for v in adj[u]:
                if v == parent:
                    continue  # do not traverse back to immediate parent
                    
                if disc[v] != -1:
                    # v already visited -> back-edge found!
                    low[u] = min(low[u], disc[v])
                else:
                    # Forward edge -> recurse
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    
                    # Bridge condition check
                    if low[v] > disc[u]:
                        bridges.append([u, v])
                        
        dfs(0, -1)
        return bridges
```

---

### Worked-Out Example

```python
n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
DFS cycle: 0 -> 1 -> 2 -> (back to 0):
  disc[0]=0, low[0]=0
  disc[1]=1, low[1]=0
  disc[2]=2, low[2]=0
For node 3 (leaf off node 1):
  disc[3]=3, low[3]=3
  low[3] (3) > disc[1] (1) -> [1, 3] is a Bridge!
Result = [[1, 3]]
```

---

### Complexity Analysis

* **Time Complexity:** `O(V + E) (single linear DFS pass)`
* **Space Complexity:** `O(V + E) for adjacency list and recursion stack`

---

### Takeaway Pattern

When identifying single points of failure / critical edges in networks, use Tarjan's discovery and low-link tracking `low[v] > disc[u]`.