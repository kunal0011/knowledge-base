---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 297: Serialize and Deserialize Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 297: Serialize and Deserialize Binary Tree

---

### Problem Statement

Design an algorithm to serialize a binary tree to a string and deserialize that string back to the original tree structure.

---

### Key Observation

* Preorder traversal (Root -> Left -> Right) uniquely encodes a binary tree if null nodes are explicitly stored as sentinel tokens (e.g. `'N'`).
* Deserialization uses an iterator/queue popping tokens recursively in preorder.

---

### Core Technique: Preorder DFS String Tokenization & Recursive Construction

---

### Python 3 Solution (with typing)

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Codec:
    def serialize(self, root: TreeNode) -> str:
        vals = []
        def dfs(node):
            if not node:
                vals.append("N")
                return
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(vals)

    def deserialize(self, data: str) -> TreeNode:
        vals = iter(data.split(","))
        def dfs():
            val = next(vals)
            if val == "N":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()
```

---

### Worked-Out Example

```python
Tree: 1 -> left: 2, right: 3 (left: 4, right: 5)
Serialized string: "1,2,N,N,3,4,N,N,5,N,N"
Deserializer reconstructs root=1, left=2, right=3... exactly.
```

---

### Complexity Analysis

* **Time Complexity:** `O(n) for both serialize and deserialize`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Include explicit null markers in preorder DFS to reconstruct binary trees without ambiguity.