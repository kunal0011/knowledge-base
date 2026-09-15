---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 114: Flatten Binary Tree to Linked List"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 114: Flatten Binary Tree to Linked List

Below is a complete, structured explanation of **LeetCode 114 – Flatten Binary Tree to Linked List**, aligned with interview and problem-solving expectations.

---

## 📌 Problem Statement (LeetCode 114)

Given the `root` of a binary tree, **flatten the tree into a linked list in-place**.

* The linked list must follow the **same order as a pre-order traversal** of the binary tree.
* After flattening:

  * Each node’s `right` pointer points to the next node in the list.
  * Each node’s `left` pointer must be `None`.

### Example

Input:

```
    1
   / \
  2   5
 / \   \
3   4   6
```

Output (linked list using right pointers):

```python
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

---

## 🔑 Key Observations & Core Concepts

### 1. **Traversal Order is Fixed**

* The flattened list must follow **pre-order traversal**:

  ```
  root → left → right
  ```

### 2. **In-place Constraint**

* You are **not allowed** to:

  * Create new nodes
  * Use extra data structures like arrays to rebuild the tree
* Only pointer rewiring is permitted.

### 3. **Tree → Linked List Transformation**

At every node:

1. Flatten the left subtree
2. Flatten the right subtree
3. Attach:

   * Left subtree to the right
   * Original right subtree to the tail of the left subtree
4. Set `left = None`

### 4. **Post-order Thinking**

Although the required order is **pre-order**, the cleanest solution works using **post-order recursion**, because:

* You must fully flatten children **before** rearranging pointers at the current node.

---

## 🧠 Conceptual Illustration (Pointer Rewiring)

![Image](https://assets.leetcode.com/uploads/2021/01/14/flaten.jpg)

![Image](https://static.takeuforward.org/content/-19gKIV8N)

![Image](https://media2.dev.to/dynamic/image/width%3D1000%2Cheight%3D420%2Cfit%3Dcover%2Cgravity%3Dauto%2Cformat%3Dauto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fskhtqoa1oo1on3a9yw10.png)

At each node:

```
Before:
   1
  / \
 L   R

After:
   1
    \
     L
      \
       R
```

---

## ✅ Python 3 Solution (In-place, Recursive)

### Approach: **Reverse Pre-order (Right → Left → Root)**

We maintain a pointer `prev` that represents the previously processed node in the flattened list.

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
    def flatten(self, root: Optional[TreeNode]) -> None:
        self.prev: Optional[TreeNode] = None

        def dfs(node: Optional[TreeNode]) -> None:
            if not node:
                return

            # Reverse preorder: right -> left -> node
            dfs(node.right)
            dfs(node.left)

            node.right = self.prev
            node.left = None
            self.prev = node

        dfs(root)
```

---

## 🧪 Worked-Out Example (Step-by-Step)

Tree:

```
    1
   / \
  2   5
 / \   \
3   4   6
```

### Reverse Preorder Traversal Order

```
6 → 5 → 4 → 3 → 2 → 1
```

### Pointer Updates (Bottom-Up)

| Current Node | prev Before | right After | prev After |
| --- | --- | --- | --- |
| 6 | None | None | 6 |
| 5 | 6 | 6 | 5 |
| 4 | 5 | 5 | 4 |
| 3 | 4 | 4 | 3 |
| 2 | 3 | 3 | 2 |
| 1 | 2 | 2 | 1 |

### Final Structure

```python
1 -> 2 -> 3 -> 4 -> 5 -> 6
```

(All via `right` pointers, all `left = None`)

---

## ⏱ Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(h)** (recursion stack, `h` = tree height) |
| Extra Data Structures | ❌ None |

---

## 🎯 Key Takeaways (Interview-Ready)

* The output order is **pre-order**, but the implementation is easiest using **reverse pre-order**
* Always think in terms of **pointer rewiring**, not rebuilding
* Using a `prev` pointer eliminates the need to search for subtree tails
* This is a classic example of **tree → list transformation using DFS**

If you want, I can also explain:

* Iterative solution using a stack
* Morris traversal (O(1) extra space)
* Why naive preorder + tail search becomes O(n²)