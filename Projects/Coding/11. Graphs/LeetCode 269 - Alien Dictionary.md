---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 269: Alien Dictionary"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 269: Alien Dictionary

---

### Problem Statement

Given a sorted dictionary of an alien language, derive the lexicographical order of characters in the language. Return empty string if invalid/cycle exists.

---

### Key Observation

* Compare adjacent words `w1` and `w2`: the first differing character `w1[i] != w2[i]` defines a directed graph edge `w1[i] -> w2[i]`.
* If `w2` is a proper prefix of `w1` (e.g. `"abc"` before `"ab"`), sorting is invalid.
* Extract topological ordering via Kahn's BFS or Postorder DFS with cycle detection.

---

### Core Technique: Lexicographical Comparison + Topological Sort

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import defaultdict, deque

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        adj = {c: set() for w in words for c in w}
        indegree = {c: 0 for c in adj}
        
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:min_len] == w2[:min_len]:
                return ""  # invalid prefix order
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in adj[w1[j]]:
                        adj[w1[j]].add(w2[j])
                        indegree[w2[j]] += 1
                    break
                    
        queue = deque([c for c in indegree if indegree[c] == 0])
        res = []
        
        while queue:
            c = queue.popleft()
            res.append(c)
            for neighbor in adj[c]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return "".join(res) if len(res) == len(adj) else ""
```

---

### Worked-Out Example

```python
words = ["wrt","wrf","er","ett","rftt"]
wrt vs wrf -> t -> f
wrf vs er -> w -> e
er vs ett -> r -> t
ett vs rftt -> e -> r
Order: w -> e -> r -> t -> f -> "wertf"
```

---

### Complexity Analysis

* **Time Complexity:** `O(C) where C is total length of all characters across words`
* **Space Complexity:** `O(V + E) = O(1) (bounded by unique alphabet size 26)`

---

### Takeaway Pattern

Diff adjacent words to build precedence DAG, then Topological Sort to recover total alphabet ordering.