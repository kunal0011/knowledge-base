---
date: "2026-08-29"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal

---

### Problem Statement

Given two integer arrays `preorder` and `inorder`, construct and return the binary tree.

---

### Key Observation

* `preorder[0]` is always the current subtree Root.
* Finding root in `inorder` splits nodes into Left subtree and Right subtree.
* Build a Hash Map of `inorder` indices for `O(1)` root position lookups.

---

### Core Technique: Divide & Conquer Traversal Reconstruction with Hash Map

---

### Python 3 Solution (with typing)

```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        in_map = {val: idx for idx, val in enumerate(inorder)}
        pre_idx = 0
        
        def helper(left, right):
            nonlocal pre_idx
            if left > right:
                return None
                
            root_val = preorder[pre_idx]
            pre_idx += 1
            root = TreeNode(root_val)
            
            mid = in_map[root_val]
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root
            
        return helper(0, len(inorder) - 1)
```

---

### Worked-Out Example

```
preorder = [3, 9, 20, 15, 7], inorder = [9, 3, 15, 20, 7]
root = 3 (index 1 in inorder)
left subtree inorder = [9]
right subtree inorder = [15, 20, 7]
Recursively builds left and right subtrees.
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Use a hash map on inorder array to avoid O(n^2) slicing and achieve strict O(n) construction.