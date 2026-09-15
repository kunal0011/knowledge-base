---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 133: Clone Graph"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 133: Clone Graph

---

### Problem Statement

Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

---

### Key Observation

* Use a Hash Map mapping `original_node -> cloned_node` to avoid infinite recursion on graph cycles.
* Perform DFS/BFS traversing neighbors and linking cloned nodes.

---

### Core Technique: Graph Traversal with Visited Hash Map

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        clones = {}
        
        def dfs(curr: 'Node') -> 'Node':
            if curr in clones:
                return clones[curr]
            copy = Node(curr.val)
            clones[curr] = copy
            for neighbor in curr.neighbors:
                copy.neighbors.append(dfs(neighbor))
            return copy
            
        return dfs(node)
```

---

### Worked-Out Example

```python
Node 1 connects to 2, 4
dfs(1): creates copy 1 -> calls dfs(2) -> creates copy 2 -> calls dfs(1) (returns copy 1 from clones map)
Graph cycles are safely terminated via map lookup.
```

---

### Complexity Analysis

* **Time Complexity:** `O(V + E) (vertices + edges)`
* **Space Complexity:** `O(V)`

---

### Takeaway Pattern

Always maintain an `old\_node -> new\_node` map when deep-cloning pointer graph structures with cycles.