---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 987: Vertical Order Traversal of a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 987: Vertical Order Traversal of a Binary Tree

**Target Companies:** Amazon, Google, Meta

---

### Problem Statement

Given root of a binary tree, calculate the vertical order traversal: nodes with same column and row are sorted by value.

---

### Key Observation

* Assign each node coordinates `(row, col)`: root is `(0, 0)`, left child is `(row + 1, col - 1)`, right child is `(row + 1, col + 1)`.
* Traverse tree with BFS/DFS collecting nodes in dictionary mapping `col -> [(row, val)]`.
* Sort columns from left to right, and sort nodes within each column by `(row, val)`.

---

### Core Technique: Coordinate Coordinate Mapping + Multi-Level Sorting

---

### Python 3 Solution (with typing)

```python
from typing import List, Optional
from collections import defaultdict, deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
            
        nodes = defaultdict(list)  # col -> list of (row, val)
        queue = deque([(root, 0, 0)])  # (node, row, col)
        
        while queue:
            node, r, c = queue.popleft()
            nodes[c].append((r, node.val))
            if node.left:
                queue.append((node.left, r + 1, c - 1))
            if node.right:
                queue.append((node.right, r + 1, c + 1))
                
        res = []
        for c in sorted(nodes.keys()):
            # Sort by row first, then by node value
            col_sorted = [val for r, val in sorted(nodes[c], key=lambda x: (x[0], x[1]))]
            res.append(col_sorted)
            
        return res
```

---

### Worked-Out Example

```python
Tree: [1,2,3,4,5,6,7]
col -2: [4]
col -1: [2]
col  0: [(0, 1), (2, 5), (2, 6)] -> [1, 5, 6]
col +1: [3]
col +2: [7]
Result: [[4],[2],[1,5,6],[3],[7]]
```

---

### Complexity Analysis

* **Time Complexity:** `O(N log N)`
* **Space Complexity:** `O(N)`

---

### Takeaway Pattern

Track `(row, col)` coordinate frames during BFS and use composite sort `(row, val)` for tie-breaking.