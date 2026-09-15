---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 332: Reconstruct Itinerary"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 332: Reconstruct Itinerary

**Target Companies:** Google (Top Asked Hard), Amazon | **Algorithm:** Hierholzer's Algorithm (Eulerian Path in Directed Graph)

---

### Problem Statement

Given a list of airline tickets `[from, to]`, reconstruct the itinerary in order that uses all tickets exactly once, starting at 'JFK'. If multiple valid itineraries exist, return the smallest lexicographical one.

---

### Key Observation

* This problem asks for an **Eulerian Path** (a path visiting every edge in a directed graph exactly once).
* **Hierholzer's Algorithm**: Greedily follow edges in lexicographical order. When reaching a dead end (node with no outgoing edges), prepend it to the itinerary (Postorder DFS).
* Reverse the postorder traversal at the end to obtain the valid Eulerian path.

---

### Core Algorithm & Technique: Hierholzer's Algorithm via Postorder DFS Traversal

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        adj = defaultdict(list)
        # Sort in reverse so we can pop from end in O(1)
        for u, v in sorted(tickets, reverse=True):
            adj[u].append(v)
            
        itinerary = []
        
        def dfs(airport: str):
            while adj[airport]:
                next_dest = adj[airport].pop()
                dfs(next_dest)
            itinerary.append(airport)
            
        dfs("JFK")
        return itinerary[::-1]  # reverse postorder
```

---

### Worked-Out Example

```python
tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
adj: JFK -> ["MUC"], MUC -> ["LHR"], LHR -> ["SFO"], SFO -> ["SJC"], SJC -> []
DFS path: JFK -> MUC -> LHR -> SFO -> SJC (dead end!)
Postorder appends: SJC, SFO, LHR, MUC, JFK
Reversed: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

---

### Complexity Analysis

* **Time Complexity:** `O(E log E) for sorting edges`
* **Space Complexity:** `O(V + E)`

---

### Takeaway Pattern

To traverse every directed edge exactly once in lexicographical order, use Hierholzer's postorder DFS.