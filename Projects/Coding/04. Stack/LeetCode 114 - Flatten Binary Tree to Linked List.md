---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 114: Flatten Binary Tree to Linked List"
tags:
  - leetcode
  - coding
  - stack
---

# LeetCode 114: Flatten Binary Tree to Linked List

Below is a structured, interview-ready explanation for **LeetCode 114 – Flatten Binary Tree to Linked List**, aligned with how this problem is typically expected to be reasoned and implemented.

---

## LeetCode 114 – Flatten Binary Tree to Linked List

### Problem Statement

Given the `root` of a binary tree, **flatten the tree into a linked list in-place**.

* The linked list should follow the **same order as a pre-order traversal** (root → left → right).
* After flattening:

  * Each node’s `right` pointer points to the next node in the list.
  * Each node’s `left` pointer must be set to `None`.
* You must do this **in-place**, without using extra nodes.

---

### Key Observation

1. **Required order is Preorder traversal**

   ```
   root → left subtree → right subtree
   ```
2. The final structure is:

   ```
   root
     \
      node1
        \
         node2
           \
            ...
   ```
3. This is essentially:

   > Convert a binary tree into a **right-skewed tree** following preorder.

---

### Stack-Based Key Insight (Iterative Approach)

The recursive preorder traversal is straightforward, but **iteration with a stack** gives more control and avoids recursion depth issues.

#### Core Insight

* Use a stack to simulate preorder traversal.
* Always process:

  1. Current node
  2. Left child
  3. Right child

But since a stack is **LIFO**:

* Push **right child first**
* Push **left child next**

This ensures the left subtree is processed before the right subtree.

---

### Algorithm (Stack-Based)

1. Initialize a stack with `root`.
2. Maintain a pointer `prev` to track the previously processed node.
3. While stack is not empty:

   * Pop the top node (`curr`)
   * If `prev` exists:

     * `prev.right = curr`
     * `prev.left = None`
   * Push `curr.right` (if exists)
   * Push `curr.left` (if exists)
   * Update `prev = curr`

---

### Python 3 Solution (with typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return

        stack = [root]
        prev: Optional[TreeNode] = None

        while stack:
            curr = stack.pop()

            if prev:
                prev.right = curr
                prev.left = None

            # Push right first so left is processed first
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)

            prev = curr
```

---

### Worked-Out Example

#### Input Tree

```
        1
       / \
      2   5
     / \   \
    3   4   6
```

---

#### Step-by-Step Stack Simulation

| Stack (Top → Bottom) | Current | Action |
| --- | --- | --- |
| [1] | 1 | push 5, push 2 |
| [5, 2] | 2 | link 1 → 2, push 4, push 3 |
| [5, 4, 3] | 3 | link 2 → 3 |
| [5, 4] | 4 | link 3 → 4 |
| [5] | 5 | link 4 → 5, push 6 |
| [6] | 6 | link 5 → 6 |

---

#### Final Flattened Structure

```
1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

* All `left` pointers → `None`
* All nodes linked via `right` pointer
* Order matches preorder traversal

---

### Complexity Analysis

* **Time Complexity:** `O(n)` — each node processed once
* **Space Complexity:** `O(n)` — stack usage in worst case (skewed tree)

---

### Why This Stack Approach Is Preferred in Interviews

* Clearly demonstrates preorder traversal control
* Avoids recursion stack overflow
* Clean, deterministic pointer rewiring
* Easily extendable to explain Morris traversal as a follow-up

---

If you want, I can also:

* Explain the **O(1) space Morris Traversal** version
* Compare **recursive vs iterative** flattening
* Draw a **pointer-rewiring diagram** for deeper intuition