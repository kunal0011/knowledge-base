---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1022: Sum of Root To Leaf Binary Numbers"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 1022: Sum of Root To Leaf Binary Numbers

Below is a **complete, structured explanation** of **LeetCode 1022 – Sum of Root To Leaf Binary Numbers**, aligned with your usual learning format.

---

## Problem Statement (LeetCode 1022)

You are given the `root` of a **binary tree** where each node has a value `0` or `1`.

Each **root-to-leaf path** represents a **binary number** formed by concatenating the node values along the path.

Return the **sum of all these binary numbers**, converted to decimal.

---

### Example

```
        1
       / \
      0   1
     / \   \
    0   1   1
```

Root-to-leaf paths:

* `100` → 4
* `101` → 5
* `111` → 7

Answer = `4 + 5 + 7 = 16`

---

## Key Observations & Core Concepts

### 1. Root-to-Leaf Traversal

* We must process **all root-to-leaf paths**
* This is a classic **tree DFS problem**

### 2. Binary Number Construction

At each node:

```
new_value = (previous_value << 1) | current_node.val
```

Why this works:

* Left shift (`<< 1`) multiplies by 2
* OR (`|`) appends the current bit (`0` or `1`)

Example:

```
binary: 10
next bit: 1
→ (10 << 1) | 1 = 101
```

### 3. Leaf Node Condition

* A path contributes to the sum **only when we reach a leaf**
* Leaf = no left child AND no right child

---

## Algorithm (DFS)

1. Start DFS from the root with `current_value = 0`
2. At each node:

   * Update `current_value`
3. If leaf:

   * Add `current_value` to result
4. Recurse left and right

---

## Python 3 Solution (with Typing)

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
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        
        def dfs(node: Optional[TreeNode], curr: int) -> int:
            if not node:
                return 0
            
            # Append current bit
            curr = (curr << 1) | node.val
            
            # If leaf node, return the formed number
            if not node.left and not node.right:
                return curr
            
            # Continue DFS
            return dfs(node.left, curr) + dfs(node.right, curr)
        
        return dfs(root, 0)
```

---

## Worked-Out Example (Step-by-Step)

### Tree

```
        1
       / \
      0   1
     / \   \
    0   1   1
```

---

### DFS Trace

#### Path 1: `1 → 0 → 0`

```
curr = 0
1 → (0 << 1) | 1 = 1
0 → (1 << 1) | 0 = 2
0 → (2 << 1) | 0 = 4   ← leaf
```

#### Path 2: `1 → 0 → 1`

```
curr = 0
1 → 1
0 → 2
1 → (2 << 1) | 1 = 5   ← leaf
```

#### Path 3: `1 → 1 → 1`

```
curr = 0
1 → 1
1 → 3
1 → (3 << 1) | 1 = 7   ← leaf
```

---

### Final Sum

```
4 + 5 + 7 = 16
```

---

## Conceptual Illustration (Binary Formation)

```
Path: 1 → 0 → 1

Start: 0
Step 1: (0 << 1) | 1 = 1      → "1"
Step 2: (1 << 1) | 0 = 2      → "10"
Step 3: (2 << 1) | 1 = 5      → "101"
```

---

## Complexity Analysis

* **Time Complexity:** `O(N)`

  * Each node visited once
* **Space Complexity:** `O(H)`

  * Recursion stack (H = height of tree)

---

## Pattern Recognition

This problem is a combination of:

* **DFS on Trees**
* **Bit Manipulation**
* **Root-to-Leaf Path Aggregation**

If you want, I can also:

* Show an **iterative stack-based DFS**
* Convert this to a **BFS approach**
* Map this problem to a **general tree path-sum pattern**

Just tell me.