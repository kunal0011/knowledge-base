---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 863: All Nodes Distance K in Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 863: All Nodes Distance K in Binary Tree

---

### Problem Statement

Given root of a binary tree, target node, and integer `k`, return an array of values of all nodes that have a distance `k` from target node.

---

### Key Observation

* Trees only allow downward traversal from parents to children.
* Traverse tree with DFS/BFS to build parent pointers (effectively converting tree into an undirected Graph).
* Perform BFS starting from `target` node expanding in 3 directions (left, right, parent) up to level `k`.

---

### Core Technique: Tree to Graph Conversion + Breadth-First Search

---

### Python 3 Solution (with typing)

```python
from typing import List
from collections import deque

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        parents = {}
        
        # 1. Map parent pointers
        def map_parents(node, parent):
            if not node:
                return
            parents[node] = parent
            map_parents(node.left, node)
            map_parents(node.right, node)
            
        map_parents(root, None)
        
        # 2. BFS from target
        queue = deque([target])
        visited = {target}
        curr_dist = 0
        
        while queue:
            if curr_dist == k:
                return [n.val for n in queue]
            for _ in range(len(queue)):
                curr = queue.popleft()
                for neighbor in (curr.left, curr.right, parents[curr]):
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            curr_dist += 1
            
        return []
```

---

### Worked-Out Example

```
Target = 5, k = 2
BFS Level 1: [parent=3, left=6, right=2]
BFS Level 2: [node 1, node 7, node 4]
Result = [1, 7, 4]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Convert tree parent pointers into graph edges to search radially outwards using BFS.