---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 124: Binary Tree Maximum Path Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 124: Binary Tree Maximum Path Sum

**LeetCode 124 – Binary Tree Maximum Path Sum**, with **clear state definition, transition equations, and DP table construction using an example**.

---

## LeetCode 124 — Binary Tree Maximum Path Sum

### Problem Statement

Given the `root` of a binary tree, return the **maximum path sum**.

**Path definition**

* A path can start and end at **any node**
* Must move **parent ↔ child**
* Must contain **at least one node**
* Does **not** need to pass through the root

---

## Key Insight (Why DP on Trees)

This is a **tree DP (postorder)** problem where:

* Each node must decide **what value it can contribute upward** to its parent
* Simultaneously, each node may form a **complete path** passing through itself (left + node + right)

---

## DP State Definition

For every node `u`, define:

### **State 1 — `down[u]`**

> Maximum sum of a path that **starts at node `u` and goes downward** to **at most one child**

This value is what `u` can contribute to its parent.

---

### **Global State — `ans`**

> Maximum path sum **anywhere in the tree**

This may include:

* Only the node
* Node + left subtree
* Node + right subtree
* Left + node + right

---

## State Transition

Let:

* `L = down[left child]`
* `R = down[right child]`

### Transition for `down[u]`

```
down[u] = u.val + max(0, L, R)
```

Why?

* You may **ignore a subtree** if it decreases the sum
* You can only choose **one direction upward**

---

### Update global answer at node `u`

```
ans = max(ans, u.val + max(0, L) + max(0, R))
```

This considers a **path passing through `u`**, possibly using **both children**

---

## Base Case

```
down[null] = 0
```

---

## DP Table Creation with Example

### Example Tree

```
        -10
        /  \
       9    20
           /  \
          15   7
```

---

### Postorder Traversal (Bottom-Up DP)

| Node | Left down | Right down | down[node] | Path through node | ans |
| --- | --- | --- | --- | --- | --- |
| 9 | 0 | 0 | `9` | `9` | `9` |
| 15 | 0 | 0 | `15` | `15` | `15` |
| 7 | 0 | 0 | `7` | `7` | `15` |
| 20 | 15 | 7 | `20 + max(15,7) = 35` | `20 + 15 + 7 = 42` | `42` |
| -10 | 9 | 35 | `-10 + 35 = 25` | `-10 + 9 + 35 = 34` | `42` |

---

### Final Answer

```
Maximum Path Sum = 42
```

Path:

```
15 → 20 → 7
```

---

## Why Two Different Values?

| Quantity | Meaning | Why needed |
| --- | --- | --- |
| `down[u]` | Best sum **extendable to parent** | Parent can only choose one child |
| `ans` | Best sum **anywhere in tree** | Path may end at any node |

---

## Python 3 DP Solution (with Typing)

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
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.ans = float('-inf')

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left = dfs(node.left)
            right = dfs(node.right)

            # DP state: downward path
            down = node.val + max(0, left, right)

            # Update global maximum path
            self.ans = max(self.ans, node.val + max(0, left) + max(0, right))

            return down

        dfs(root)
        return self.ans
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(h)** recursion stack (`h` = tree height) |

---

## Conceptual Summary

* This is **tree DP with two perspectives**
* One value flows **upward** (`down`)
* One value is **global** (`ans`)
* Postorder traversal ensures children are computed first

---

If you want, I can also provide:

* A **pure DP table visualization**
* A **recursive call stack trace**
* A **comparison with Kadane’s algorithm**
* A **LeetCode-style dry run walkthrough**

Specify your preference.