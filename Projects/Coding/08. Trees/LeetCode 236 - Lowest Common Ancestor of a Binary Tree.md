---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 236: Lowest Common Ancestor of a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 236: Lowest Common Ancestor of a Binary Tree

Below is a **structured, interview-grade explanation** of **LeetCode 236 – Lowest Common Ancestor of a Binary Tree**, aligned with your earlier DSA pattern requests.

---

## 🔹 Problem Statement (LeetCode 236)

Given a **binary tree** (not necessarily a BST) and two distinct nodes `p` and `q`, return their **lowest common ancestor (LCA)**.

> **Lowest Common Ancestor**:  
> The lowest node in the tree that has **both `p` and `q` as descendants** (a node can be a descendant of itself).

---

## 🔹 Key Observations

1. **Tree is NOT a BST**

   * No ordering property
   * Cannot use value comparisons
2. **Nodes are guaranteed to exist**

   * We do not need to handle missing nodes
3. **Definition allows self-ancestor**

   * If `p` is ancestor of `q`, then `p` is the LCA
4. **Post-order traversal is natural**

   * We need information from both subtrees **before deciding**

---

## 🔹 Core Concepts Used

| Concept | Role |
| --- | --- |
| Binary Tree Traversal | Explore entire tree |
| Post-order DFS | Decide LCA after children |
| Divide & Conquer | Each subtree returns partial result |
| Recursion | Elegant bottom-up aggregation |

---

## 🔹 Key Insight (Most Important)

At **each node**, ask:

> Do I see `p` or `q` in my left subtree?  
> Do I see `p` or `q` in my right subtree?

### Three decisive cases:

1. **Current node is `p` or `q`**  
   → Return current node
2. **`p` and `q` found in different subtrees**  
   → Current node is the **LCA**
3. **Both found in one subtree**  
   → Propagate that subtree’s result upward

---

## 🔹 Algorithm (High Level)

```
DFS(node):
    if node is None:
        return None

    if node == p or node == q:
        return node

    left = DFS(node.left)
    right = DFS(node.right)

    if left and right:
        return node   # LCA found

    return left or right
```

---

## 🔹 Python 3 Solution (With Typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, x: int):
        self.val = x
        self.left: Optional['TreeNode'] = None
        self.right: Optional['TreeNode'] = None

class Solution:
    def lowestCommonAncestor(
        self,
        root: 'TreeNode',
        p: 'TreeNode',
        q: 'TreeNode'
    ) -> 'TreeNode':

        if root is None:
            return None

        if root == p or root == q:
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:
            return root

        return left if left else right
```

---

## 🔹 Worked Example

### Tree Structure

```
            3
          /   \
         5     1
        / \   / \
       6   2 0   8
          / \
         7   4
```

### Input

```
p = 5
q = 1
```

---

### Step-by-Step DFS Reasoning

1. Start at `3`
2. Explore left subtree → returns `5`
3. Explore right subtree → returns `1`
4. Both left & right are non-null  
   ✅ **LCA = 3**

---

### Another Example

```
p = 5
q = 4
```

Traversal result:

* `4` found under subtree of `5`
* `5` matches `p`
* One side returns `None`, other returns `4`

✅ **LCA = 5**

---

## 🔹 Conceptual Illustration (Decision Flow)

```
DFS(node):
    ├── returns p?
    ├── returns q?
    └── returns LCA?
```

| Left | Right | Result |
| --- | --- | --- |
| p | q | current node (LCA) |
| p | None | p |
| None | q | q |
| None | None | None |

---

## 🔹 Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(N)** — visit each node once |
| Space | **O(H)** — recursion stack (`H` = tree height) |
| Worst Case | O(N) for skewed tree |

---

## 🔹 Why This Approach Is Optimal

* Single DFS pass
* No extra data structures
* Works for **any binary tree**
* Clean recursive reasoning (very interview-friendly)

---

If you want next:

* Iterative version
* Parent pointer approach
* Follow-up when nodes may not exist
* Visual backtracking tree like your earlier requests

Tell me how deep you want to go.