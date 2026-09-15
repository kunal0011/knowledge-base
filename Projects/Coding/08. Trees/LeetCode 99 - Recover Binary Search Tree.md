---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 99: Recover Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 99: Recover Binary Search Tree

Below is a **complete, structured explanation** of **LeetCode 99 – Recover Binary Search Tree**, aligned with how you have been studying other problems (problem statement → key observation → core concept → Python 3 solution with typing → worked example with a conceptual illustration).

---

## 1. Problem Statement (LeetCode 99)

You are given the **root of a Binary Search Tree (BST)** where **exactly two nodes have been swapped by mistake**.

Your task is to **recover the tree without changing its structure**, i.e., restore the BST by swapping the values of the incorrect nodes.

### Constraints

* The number of nodes is in the range `[2, 1000]`
* Node values are integers
* You must modify the tree **in-place**

---

## 2. Key Observations

### Observation 1: Inorder traversal of a BST

For a valid BST:

* **Inorder traversal produces a strictly increasing sequence**

Example:

```
BST inorder → [1, 2, 3, 4, 5]
```

---

### Observation 2: What happens when two nodes are swapped?

Swapping two nodes **breaks the sorted order** in the inorder traversal.

There are **two possible cases**:

#### Case 1: Swapped nodes are adjacent in inorder

```
Expected: [1, 2, 3, 4]
Actual:   [1, 3, 2, 4]
              ↑  ↑
```

→ One inversion

#### Case 2: Swapped nodes are non-adjacent

```
Expected: [1, 2, 3, 4, 5]
Actual:   [1, 4, 3, 2, 5]
              ↑     ↑
```

→ Two inversions

---

## 3. Core Concept & Algorithm

### Key Insight

During **inorder traversal**, track:

* `prev` → previously visited node
* `first` → first incorrect node
* `second` → second incorrect node

### Detection Rule

Whenever:

```
prev.val > current.val
```

we have found an inversion.

* First time inversion:

  * `first = prev`
  * `second = current`
* Second time inversion:

  * Update `second = current`

Finally:

* **Swap `first.val` and `second.val`**

---

## 4. Python 3 Solution (with Typing)

### Recursive Inorder Traversal (O(n) time, O(h) stack)

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
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        self.first = None
        self.second = None
        self.prev = None

        def inorder(node: Optional[TreeNode]) -> None:
            if not node:
                return

            inorder(node.left)

            if self.prev and self.prev.val > node.val:
                if not self.first:
                    self.first = self.prev
                self.second = node

            self.prev = node
            inorder(node.right)

        inorder(root)

        # Swap the incorrect nodes
        self.first.val, self.second.val = self.second.val, self.first.val
```

---

## 5. Worked Example with Conceptual Illustration

### Input Tree (Incorrect BST)

```
        3
       / \
      1   4
         /
        2
```

### Inorder Traversal

```
[1, 3, 2, 4]
```

### Step-by-Step Inorder Detection

| Prev | Current | Condition | Action |
| --- | --- | --- | --- |
| 1 | 3 | OK | continue |
| 3 | 2 | 3 > 2 | first = 3, second = 2 |
| 2 | 4 | OK | continue |

---

### Conceptual Illustration of Inversion

```
Inorder: 1 → 3 → 2 → 4
                ↑
         violation detected
```

* `first = 3`
* `second = 2`

---

### After Swapping

```
Swap values: 3 ↔ 2
```

### Correct BST

```
        2
       / \
      1   4
         /
        3
```

### Valid Inorder Traversal

```
[1, 2, 3, 4]
```

---

## 6. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(h)** (recursion stack) |
| In-place | ✅ Yes |

---

## 7. Key Takeaways (Exam & Interview Focus)

* **BST + Inorder = Sorted**
* Identify **inversions** in inorder traversal
* Track only **two nodes**
* No restructuring of the tree
* Common follow-up: **Morris Traversal (O(1) space)**

---

If you want, I can next:

* Explain the **Morris Traversal (constant space)** version
* Provide a **visual trace with pointer movement**
* Compare this with **LeetCode 98 (Validate BST)** to avoid confusion