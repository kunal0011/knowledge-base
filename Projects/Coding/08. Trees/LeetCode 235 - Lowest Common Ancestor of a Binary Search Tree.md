---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 235: Lowest Common Ancestor of a Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 235: Lowest Common Ancestor of a Binary Search Tree

Below is a structured, interview-ready explanation of **LeetCode 235 – Lowest Common Ancestor of a Binary Search Tree**, aligned with how you have been studying other LeetCode patterns.

---

## LeetCode 235

**Lowest Common Ancestor of a Binary Search Tree**

---

## 1. Problem Statement

You are given a **Binary Search Tree (BST)** and two nodes `p` and `q`.

The task is to find their **Lowest Common Ancestor (LCA)**.

> **Definition**  
> The *lowest common ancestor* of two nodes `p` and `q` is the **lowest node in the tree that has both `p` and `q` as descendants** (a node can be a descendant of itself).

### Constraints

* All node values are **unique**
* `p` and `q` **exist in the BST**
* Tree is a **valid BST**

---

## 2. Key Observations

This problem is **BST-specific**, which dramatically simplifies the solution.

### BST Property

For any node `root`:

* All values in the **left subtree** are `< root.val`
* All values in the **right subtree** are `> root.val`

---

## 3. Core Insight (Most Important)

At any node `root`:

| Condition | Interpretation |
| --- | --- |
| `p.val < root.val` **and** `q.val < root.val` | Both nodes lie in **left subtree** |
| `p.val > root.val` **and** `q.val > root.val` | Both nodes lie in **right subtree** |
| Otherwise | **Split happens here → this node is the LCA** |

This “**split point**” is the essence of the problem.

---

## 4. Algorithm (Conceptual)

1. Start from the `root`
2. Compare `p.val` and `q.val` with `root.val`
3. Move:

   * **Left** if both are smaller
   * **Right** if both are larger
4. If one is on each side (or one equals root), **current node is LCA**

---

## 5. Python 3 Solution (with Typing)

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
    def lowestCommonAncestor(
        self,
        root: TreeNode,
        p: TreeNode,
        q: TreeNode
    ) -> TreeNode:

        curr = root

        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                return curr
```

---

## 6. Worked-Out Example

### Input Tree

```
        6
       / \
      2   8
     / \ / \
    0  4 7  9
      / \
     3   5
```

Let:

* `p = 2`
* `q = 8`

---

### Step-by-Step Traversal

| Current Node | p.val | q.val | Decision |
| --- | --- | --- | --- |
| `6` | 2 | 8 | Split happens → **LCA found** |

### Answer

```
LCA = 6
```

---

## 7. Conceptual Illustration (Split Point)

```
        6   ← LCA
       / \
    p=2   q=8
```

* `p` is in the **left subtree**
* `q` is in the **right subtree**
* First node where paths diverge → **LCA**

---

## 8. Another Example (Both on Same Side)

### Input

* `p = 2`
* `q = 4`

Traversal:

```
6 → left (both < 6)
2 → split happens (2 ≤ 2 ≤ 4)
```

### Answer

```
LCA = 2
```

---

## 9. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(h)** (tree height) |
| Space Complexity | **O(1)** (iterative) |

* Balanced BST → `O(log n)`
* Skewed BST → `O(n)`

---

## 10. Key Takeaways (Interview Gold)

* **BST property eliminates recursion/backtracking**
* LCA is the **first node where paths to p and q diverge**
* Iterative solution is cleaner and optimal
* This logic **does NOT work for a general binary tree** (that is LeetCode 236)

---

If you want, I can next:

* Contrast **LeetCode 235 vs 236**
* Show **recursive version**
* Convert this into a **generic LCA template**
* Provide **edge-case traps interviewers ask**

Just indicate your preference.