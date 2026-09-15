---
date: "2026-08-29"
type: leetcode-solution
category: "Graphs"
folder: "11. Graphs"
title: "LeetCode 207: Course Schedule"
tags:
  - leetcode
  - coding
  - graphs
---

# LeetCode 207: Course Schedule

---

### Problem Statement

There are `numCourses` courses labeled from `0` to `numCourses - 1`. You are given an array `prerequisites` where `[a, b]` means you must take `b` before `a`. Return `true` if you can finish all courses.

---

### Key Observation

* Can be modeled as a Directed Graph: an edge `b -> a` represents a dependency.
* Finishing all courses is possible if and only if the directed graph contains **NO directed cycles**.
* Use Kahn's Algorithm (BFS with In-degree count) or DFS 3-color state traversal.

---

### Core Technique: Topological Sort / Kahn's Algorithm (BFS In-Degree)

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque, defaultdict

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = defaultdict(list)
        indegree = [0] * numCourses
        
        for course, prereq in prerequisites:
            adj[prereq].append(course)
            indegree[course] += 1
            
        # Queue all nodes with indegree 0 (no prerequisites)
        queue = deque([i for i in range(numCourses) if indegree[i] == 0])
        completed = 0
        
        while queue:
            node = queue.popleft()
            completed += 1
            for neighbor in adj[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return completed == numCourses
```

---

### Worked-Out Example

```python
numCourses = 2, prerequisites = [[1, 0]]
indegree = [0: 0, 1: 1], adj = {0: [1]}
queue = [0]
pop 0 -> completed = 1, indegree[1] becomes 0 -> queue = [1]
pop 1 -> completed = 2
completed == 2 -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(V + E)`
* **Space Complexity:** `O(V + E)`

---

### Takeaway Pattern

Topological sort via in-degree tracking resolves dependency resolution and directed cycle detection.