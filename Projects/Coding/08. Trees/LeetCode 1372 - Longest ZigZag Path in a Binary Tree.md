---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1372: Longest ZigZag Path in a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 1372: Longest ZigZag Path in a Binary Tree

Below is a structured, interview-ready explanation for **LeetCode 1372 – Longest ZigZag Path in a Binary Tree**, aligned with your preferred format.

---

## LeetCode 1372: Longest ZigZag Path in a Binary Tree

---

## Problem Statement

You are given the `root` of a binary tree.

A **ZigZag path** is defined as a path where:

* You start from **any node**.
* You move **downwards only** (parent → child).
* The direction of movement **alternates** between left and right at each step.

Return the **maximum number of edges** in the longest ZigZag path.

---

## Key Observations

1. **Direction matters**

   * A ZigZag path depends on the **previous direction** taken.
   * At each node, the next valid move is constrained.
2. **Path can start anywhere**

   * The root is **not mandatory** as a starting point.
   * Any node can be treated as a potential start.
3. **Count edges, not nodes**

   * Single node → length `0`.
   * Each valid move adds `+1`.
4. **Local optimal ≠ Global optimal**

   * We must compute ZigZag lengths at **every node**.
   * A global maximum must be tracked.

---

## Core Concept

At each node, maintain:

* `left_len`: Longest ZigZag path **ending at this node** where the **last move was left**.
* `right_len`: Longest ZigZag path **ending at this node** where the **last move was right**.

### Transition Rule

* If we move **left**, the previous move must have been **right**.
* If we move **right**, the previous move must have been **left**.

---

## Algorithm (DFS / Post-Order Traversal)

1. Perform DFS on the tree.
2. For each node:

   * Recursively compute `(left_len, right_len)` from children.
   * Update:

     * `curr_left = 1 + right_len_of_left_child`
     * `curr_right = 1 + left_len_of_right_child`
3. Update a global maximum.
4. Return `(curr_left, curr_right)`.

---

## Python 3 Solution (with typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.ans = 0

        def dfs(node: Optional[TreeNode]) -> tuple[int, int]:
            if not node:
                return -1, -1  # base case to offset +1 correctly

            left = dfs(node.left)
            right = dfs(node.right)

            left_len = 1 + left[1]
            right_len = 1 + right[0]

            self.ans = max(self.ans, left_len, right_len)

            return left_len, right_len

        dfs(root)
        return self.ans
```

---

## Worked Example

### Input Tree

```
        1
         \
          1
         /
        1
         \
          1
         /
        1
```

### ZigZag Path

```
Right → Left → Right → Left
```

### Step-by-Step Evaluation

| Node | left\_len | right\_len |
| --- | --- | --- |
| leaf | 0 | 0 |
| parent | 1 | 0 |
| parent | 0 | 2 |
| parent | 3 | 0 |

**Final Answer = 4 edges**

---

## Conceptual Illustration

![Image](https://assets.leetcode.com/uploads/2020/01/22/sample_1_1702.png)

![Image](https://ninjasfiles.s3.amazonaws.com/asset_0000000000000417_1618599937_path_in_zigzag_tree_example.png)

![Image](https://i.sstatic.net/U7mzd.png)

**Interpretation**

* Each edge alternates direction.
* The path may start at any depth.
* DFS ensures every node contributes to the global maximum.

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(N)** |
| Space | **O(H)** (recursion stack) |

Where:

* `N` = number of nodes
* `H` = height of tree

---

## Final Takeaway

* This is a **tree DP problem with direction-aware state**.
* Track two values per node instead of recomputing paths.
* Post-order DFS is essential because each node depends on its children.

If you want, I can also:

* Show an **iterative DFS version**
* Convert this to a **state-diagram explanation**
* Compare this with similar problems (e.g., alternating paths in graphs)