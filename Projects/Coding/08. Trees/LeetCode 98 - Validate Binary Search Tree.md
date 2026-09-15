---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 98: Validate Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 98: Validate Binary Search Tree

Below is a structured, interview-grade explanation for **LeetCode 98 – Validate Binary Search Tree**, aligned with how this problem is expected to be reasoned about in system design and algorithmic interviews.

---

## 1. Problem Statement (LeetCode 98)

You are given the **root of a binary tree**.  
Determine whether it is a **valid Binary Search Tree (BST)**.

A **valid BST** satisfies **all** of the following:

1. The left subtree of a node contains **only nodes with keys strictly less** than the node’s key.
2. The right subtree of a node contains **only nodes with keys strictly greater** than the node’s key.
3. **Both** the left and right subtrees must also be valid BSTs.

---

## 2. Key Observations & Core Concepts

### ❌ Common Incorrect Assumption

Checking only:

* `node.left.val < node.val`
* `node.right.val > node.val`

is **not sufficient**.

**Why?**  
Because BST validity is a **global constraint**, not just a parent–child constraint.

---

### ✅ Correct Insight (Global Range Constraint)

Each node must satisfy:

```
min_allowed < node.val < max_allowed
```

Where:

* `min_allowed` comes from ancestors on the **left**
* `max_allowed` comes from ancestors on the **right**

This range **shrinks as we go down the tree**.

---

## 3. Two Correct Conceptual Approaches

### Approach 1: DFS with Min / Max Bounds (Preferred)

* Start with range `(-∞, +∞)`
* For each node:

  * Validate `min < node.val < max`
  * Recurse:

    * Left child → `(min, node.val)`
    * Right child → `(node.val, max)`

**Time Complexity:** `O(n)`  
**Space Complexity:** `O(h)` (recursion stack)

---

### Approach 2: Inorder Traversal (Property-Based)

* Inorder traversal of a valid BST produces a **strictly increasing sequence**
* Track previous value and ensure current > previous

This works, but **Approach 1 is clearer and safer** in interviews.

---

## 4. Python 3 Solution (with Typing)

### DFS with Range Validation

```python
from typing import Optional

class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right: Optional["TreeNode"] = None
    ):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        def dfs(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True

            if not (low < node.val < high):
                return False

            return (
                dfs(node.left, low, node.val) and
                dfs(node.right, node.val, high)
            )

        return dfs(root, float("-inf"), float("inf"))
```

---

## 5. Worked-Out Example (Step-by-Step)

### Example Tree (Invalid BST)

```
        5
       / \
      1   4
         / \
        3   6
```

---

### Step-wise Validation

| Node | Allowed Range | Node Value | Valid? |
| --- | --- | --- | --- |
| 5 | (-∞, +∞) | 5 | ✅ |
| 1 | (-∞, 5) | 1 | ✅ |
| 4 | (5, +∞) | 4 | ❌ |

🚫 **Violation:**  
`4` appears in the **right subtree of 5**, but `4 < 5`.

---

## 6. Conceptual Illustration (Range Propagation)

```
Start:
Node 5 → range (-∞, +∞)

Left Subtree:
Node 1 → range (-∞, 5)

Right Subtree:
Node 4 → range (5, +∞) ❌
```

This illustrates **why local checks fail** and **global range tracking is required**.

---

## 7. Why This Problem Is Important

* Tests understanding of **tree recursion**
* Reinforces **global vs local constraints**
* Commonly used as a **filter problem** in interviews
* Pattern repeats in:

  * Tree validation
  * Interval problems
  * Distributed consistency checks

---

## 8. Key Takeaways

* BST validation is about **ranges, not neighbors**
* Always think **top-down constraints**
* Use **DFS with bounds** for clarity and correctness
* Strict inequalities (`<` and `>`) matter

If you want, I can also:

* Walk through the **inorder approach**
* Provide a **visual recursion trace**
* Show **iterative stack-based solution**
* Map this problem to a **general tree validation pattern**

Just let me know.