---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 993: Cousins in Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 993: Cousins in Binary Tree

Below is a structured, interview-grade explanation of **LeetCode 993 – Cousins in Binary Tree**, aligned with your usual format.

---

## LeetCode 993: Cousins in Binary Tree

### Problem Statement

Given the `root` of a binary tree and two distinct values `x` and `y`, return `true` if the nodes corresponding to `x` and `y` are **cousins**.

Two nodes are cousins **if and only if**:

1. They are at the **same depth** (same level).
2. They have **different parents**.

Otherwise, return `false`.

---

## Key Observations

1. **Depth matters**  
   Cousins must lie on the same level of the tree.
2. **Parent must be different**  
   Even if two nodes are on the same level, they are **not cousins** if they share the same parent (siblings).
3. **Binary tree traversal gives both facts**  
   During traversal, for every node we can record:

   * Its **depth**
   * Its **parent**
4. **Once both nodes are found, no further traversal is needed**  
   Early termination is possible.

---

## Core Concepts Used

* **Tree traversal** (DFS or BFS)
* **Depth tracking**
* **Parent tracking**
* **State capture during traversal**

Both DFS and BFS work.  
Below, DFS is used for conceptual clarity and simplicity.

---

## Algorithm (DFS-based)

1. Traverse the tree.
2. When node value equals `x` or `y`, record:

   * Its depth
   * Its parent
3. After traversal:

   * Check if depths are equal
   * Check if parents are different

---

## Python 3 Solution (with Typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        x_info = None  # (parent, depth)
        y_info = None  # (parent, depth)

        def dfs(node: Optional[TreeNode],
                parent: Optional[TreeNode],
                depth: int) -> None:
            nonlocal x_info, y_info

            if not node:
                return

            if node.val == x:
                x_info = (parent, depth)
            elif node.val == y:
                y_info = (parent, depth)

            # Stop early if both found
            if x_info and y_info:
                return

            dfs(node.left, node, depth + 1)
            dfs(node.right, node, depth + 1)

        dfs(root, None, 0)

        if not x_info or not y_info:
            return False

        return x_info[1] == y_info[1] and x_info[0] != y_info[0]
```

---

## Worked Example

### Input Tree

```
        1
       / \
      2   3
       \   \
        4   5
```

* `x = 4`
* `y = 5`

![Image](https://www.codiwan.com/img/cousins-in-binary-tree-993/traversal.png)

![Image](https://assets.leetcode.com/uploads/2019/02/12/q1248-01.png)

---

### Step-by-Step Traversal

| Node | Parent | Depth |
| --- | --- | --- |
| 1 | None | 0 |
| 2 | 1 | 1 |
| 3 | 1 | 1 |
| 4 | 2 | 2 |
| 5 | 3 | 2 |

---

### Evaluation

* Depth of `4` = 2
* Depth of `5` = 2
* Parent of `4` = 2
* Parent of `5` = 3

Same depth + different parents ⇒ **Cousins**

**Output:** `True`

---

## Conceptual Illustration (Why It Works)

* DFS ensures **every node knows its parent and depth**
* Cousin condition becomes a **simple comparison**
* No tree restructuring or extra data structures required

---

## Time and Space Complexity

* **Time:** `O(N)` — each node visited once
* **Space:** `O(H)` — recursion stack, where `H` is tree height

---

## Summary

* Cousins = **same depth, different parents**
* Track parent and depth during traversal
* DFS or BFS both valid; DFS is concise
* Early termination improves efficiency

If you want, I can also:

* Rewrite this using **BFS (level-order)**
* Show **why sibling cases fail** with counterexamples
* Convert this into a **template for similar tree problems**