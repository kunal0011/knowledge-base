---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 222: Count Complete Tree Nodes"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 222: Count Complete Tree Nodes

Below is a structured, interview-ready explanation of **LeetCode 222 – Count Complete Tree Nodes**, aligned with how you have been studying other problems.

---

## LeetCode 222: Count Complete Tree Nodes

---

### Problem Statement

You are given the **root of a complete binary tree**.  
Return the **number of nodes** in the tree.

**Definition – Complete Binary Tree**

* Every level, except possibly the last, is **completely filled**
* All nodes in the last level are as **far left as possible**

**Constraints**

* The tree can have up to ( 5 \times 10^4 ) nodes
* A naive traversal solution (O(n)) works, but the problem **expects better than O(n)**

---

## Key Observation & Core Concepts

### 1. Structure Property of Complete Binary Trees

For any node:

* If **left subtree height == right subtree height**  
  → the **left subtree is a perfect binary tree**
* Else  
  → the **right subtree is a perfect binary tree**

A **perfect binary tree** with height `h` has:  
[  
\text{nodes} = 2^h - 1  
]

---

### 2. Height Definition (Important)

Height is measured by:

* Following only **left child pointers**
* Height of an empty tree = 0

This works because a complete tree is always left-aligned.

---

### 3. Divide & Conquer Strategy

At each node:

1. Compute left height
2. Compute right height
3. Decide which subtree is perfect
4. Count nodes using formula + recurse on the other subtree

This reduces complexity to:

[  
O((\log n)^2)  
]

---

## Conceptual Illustration

![Image](https://deen3evddmddt.cloudfront.net/uploads/content-images/what-is-complete-binary-tree.webp)

![Image](https://media.geeksforgeeks.org/wp-content/uploads/20220630154756/img2.jpg)

![Image](https://miro.medium.com/1%2ACMGFtehu01ZEBgzHG71sMg.png)

Example tree:

```
        1
       / \
      2   3
     / \  /
    4  5 6
```

* Left height = 3
* Right height = 2
* Right subtree is perfect → count directly
* Recurse on left subtree

---

## Python 3 Solution (with typing)

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
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        def left_height(node: Optional[TreeNode]) -> int:
            height = 0
            while node:
                height += 1
                node = node.left
            return height

        lh = left_height(root.left)
        rh = left_height(root.right)

        # Left subtree is perfect
        if lh == rh:
            return (1 << lh) + self.countNodes(root.right)
        # Right subtree is perfect
        else:
            return (1 << rh) + self.countNodes(root.left)
```

---

## Worked-Out Example (Step-by-Step)

Tree:

```
        1
       / \
      2   3
     / \  /
    4  5 6
```

### Step 1: At root (1)

* Left height = 2 (2 → 4)
* Right height = 1 (3 → 6)
* Heights differ → right subtree is perfect

Right subtree nodes:  
[  
2^1 = 2  
]

Recurse on left subtree

---

### Step 2: At node (2)

* Left height = 1 (4)
* Right height = 1 (5)
* Equal → left subtree is perfect

Left subtree nodes:  
[  
2^1 = 2  
]

Recurse on right subtree

---

### Step 3: At node (5)

* Leaf node → count = 1

---

### Final Count

| Component | Nodes |
| --- | --- |
| Root | 1 |
| Perfect right subtree | 2 |
| Perfect left subtree | 2 |
| Leaf | 1 |
| **Total** | **6** |

---

## Why This Works Better Than DFS

| Approach | Time Complexity |
| --- | --- |
| DFS / BFS | (O(n)) |
| This method | (O((\log n)^2)) |

Because:

* Height calculation = (O(\log n))
* Recursion depth = (O(\log n))

---

## Key Interview Takeaways

* Use **tree properties**, not brute force
* Recognize **perfect subtree patterns**
* Height is computed via **left pointers only**
* Bit-shift `(1 << h)` is faster than `pow(2, h)`

---

If you want, I can also provide:

* Iterative version
* Binary search on last level approach
* Comparison with DFS solution
* Visualization of recursion stack

Just tell me what you want next.