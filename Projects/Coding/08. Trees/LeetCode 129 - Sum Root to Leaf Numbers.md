---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 129: Sum Root to Leaf Numbers"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 129: Sum Root to Leaf Numbers

Below is a structured, interview-ready explanation of **LeetCode 129 – Sum Root to Leaf Numbers**, aligned with how you have been practicing other tree problems.

---

## Problem Statement (LeetCode 129)

You are given the **root of a binary tree**, where each node contains a digit from **0 to 9**.

Each **root-to-leaf path** represents a number formed by concatenating the digits along the path.

Return the **sum of all numbers** formed from root to leaf paths.

### Definition

* A **leaf** is a node with **no left and no right child**
* Numbers are formed in **base-10**, not by addition

---

## Key Observations

1. **Path matters, not individual nodes**

   * You must consider the entire path from root to leaf as a number.
2. **Digit concatenation, not summation**

   * If path digits are `1 → 2 → 3`, the number is **123**, not `1 + 2 + 3`.
3. **Binary Tree traversal**

   * Every root-to-leaf path must be visited exactly once.
   * This strongly suggests **DFS**.
4. **Running value technique**

   * While traversing, keep the number formed so far.
   * At each node:

     ```
     current = previous * 10 + node.val
     ```
5. **Leaf node = terminal contribution**

   * Only when a leaf is reached do we add the number to the final sum.

---

## Core Concepts Used

* Depth-First Search (DFS)
* Tree traversal
* Carrying state in recursion
* Preorder traversal (process node before children)

---

## Conceptual Illustration

Consider this tree:

```
        1
       / \
      2   3
```

### Root-to-Leaf Paths

| Path | Number Formed |
| --- | --- |
| 1 → 2 | 12 |
| 1 → 3 | 13 |

### Final Answer

```
12 + 13 = 25
```

---

## DFS Conceptual Flow (Mental Model)

```
Start at root with current = 0

Node 1:
  current = 0 * 10 + 1 = 1

  Go left (Node 2):
    current = 1 * 10 + 2 = 12
    Leaf → add 12

  Go right (Node 3):
    current = 1 * 10 + 3 = 13
    Leaf → add 13

Return total = 25
```

---

## Python 3 Solution (With Typing)

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
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], current: int) -> int:
            if not node:
                return 0

            # Update the number formed so far
            current = current * 10 + node.val

            # If leaf node, return the formed number
            if not node.left and not node.right:
                return current

            # Otherwise, sum from left and right subtrees
            return dfs(node.left, current) + dfs(node.right, current)

        return dfs(root, 0)
```

---

## Time and Space Complexity

### Time Complexity

* **O(n)**  
  Each node is visited once.

### Space Complexity

* **O(h)**  
  Recursive call stack, where `h` is the height of the tree.

  * Worst case (skewed tree): `O(n)`
  * Best case (balanced tree): `O(log n)`

---

## Common Mistakes to Avoid

1. **Adding node values instead of forming numbers**

   ```
   WRONG: sum += node.val
   RIGHT: current = current * 10 + node.val
   ```
2. **Adding values at non-leaf nodes**

   * Only leaf paths contribute to the final sum.
3. **Using global state unnecessarily**

   * Cleaner to return values directly from DFS.

---

## Pattern Identification

This problem fits the pattern:

> **Binary Tree → Root-to-Leaf Path Aggregation with State Carrying**

You will see the same pattern in:

* Path Sum problems
* Tree path construction problems
* Expression trees

If you want, I can next:

* Convert this to an **iterative stack-based DFS**
* Show a **backtracking tree diagram**
* Compare this with **Path Sum II / III patterns**

Just tell me.