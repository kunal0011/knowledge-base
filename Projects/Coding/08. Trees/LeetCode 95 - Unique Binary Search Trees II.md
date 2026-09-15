---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 95: Unique Binary Search Trees II

Below is a structured, interview-grade explanation of **LeetCode 95 – Unique Binary Search Trees II**, aligned with how you have been studying recent problems (problem statement → key observation → core concept → typed Python solution → worked example with conceptual illustration).

---

## LeetCode 95: Unique Binary Search Trees II

---

## 1. Problem Statement

You are given an integer `n`.  
Generate **all structurally unique Binary Search Trees (BSTs)** that store values from `1` to `n`.

Return all possible BST roots.

### Constraints

* `1 ≤ n ≤ 8`

---

## 2. Key Observations

1. **BST Property**

   * For any node with value `k`:

     * All values in the left subtree must be `< k`
     * All values in the right subtree must be `> k`
2. **Root Selection Drives Structure**

   * If we choose `k` as the root:

     * Left subtree must be built from `[1 ... k-1]`
     * Right subtree must be built from `[k+1 ... n]`
3. **Independent Subproblems**

   * Left and right subtree constructions are **independent**
   * For a fixed root `k`, total trees =

     ```
     (# left subtrees) × (# right subtrees)
     ```
4. **Overlapping Subproblems**

   * Same `(start, end)` range is recomputed many times
   * This is a classic **divide-and-conquer with memoization** scenario

---

## 3. Core Concepts Used

### 1. Recursive Tree Construction

We recursively generate all BSTs for a given range `[start, end]`.

### 2. Cartesian Product of Subtrees

For each root `k`, we:

* Generate all possible left subtrees
* Generate all possible right subtrees
* Combine **every left** with **every right**

### 3. Base Case (Very Important)

* If `start > end`, return `[None]`

  * This allows valid attachment of empty children

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List, Optional, Dict, Tuple

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        if n == 0:
            return []

        memo: Dict[Tuple[int, int], List[Optional[TreeNode]]] = {}

        def build(start: int, end: int) -> List[Optional[TreeNode]]:
            if start > end:
                return [None]

            if (start, end) in memo:
                return memo[(start, end)]

            result: List[Optional[TreeNode]] = []

            for root_val in range(start, end + 1):
                left_trees = build(start, root_val - 1)
                right_trees = build(root_val + 1, end)

                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(root_val)
                        root.left = left
                        root.right = right
                        result.append(root)

            memo[(start, end)] = result
            return result

        return build(1, n)
```

---

## 5. Worked Example (n = 3)

### Step 1: Choose Root = 1

* Left range: `[]` → `[None]`
* Right range: `[2, 3]`

Right subtree possibilities:

```
   2          3
    \        /
     3      2
```

Generated trees:

```
1           1
 \           \
  2           3
   \         /
    3       2
```

---

### Step 2: Choose Root = 2

* Left range: `[1]`
* Right range: `[3]`

Only one combination:

```
   2
  / \
 1   3
```

---

### Step 3: Choose Root = 3

* Left range: `[1, 2]`
* Right range: `[]`

Left subtree possibilities:

```
   1          2
    \        /
     2      1
```

Generated trees:

```
     3          3
    /          /
   1          2
    \        /
     2      1
```

---

## 6. Conceptual Illustration (Recursive Decomposition)

```
build(1, 3)
├── root = 1
│   ├── left = build(1, 0) → [None]
│   └── right = build(2, 3)
│
├── root = 2
│   ├── left = build(1, 1)
│   └── right = build(3, 3)
│
└── root = 3
    ├── left = build(1, 2)
    └── right = build(4, 3) → [None]
```

Each node combines **all left × all right** trees.

---

## 7. Complexity Analysis

* **Time Complexity:**  
  Catalan number growth  
  [  
  O(C\_n) \approx O\left(\frac{4^n}{n^{3/2}}\right)  
  ]
* **Space Complexity:**

  * Recursion stack: `O(n)`
  * Memoized trees: `O(C_n)`

---

## 8. Key Takeaways (Interview Ready)

* This is **not DP over integers**, but **DP over tree structures**
* Core trick: return `[None]` for empty ranges
* This problem is the **constructive version** of LeetCode 96
* Pattern: **Divide & Combine with Memoization**

---

If you want, next we can:

* Contrast **LeetCode 95 vs 96**
* Visualize memo table usage
* Rewrite this using bottom-up DP (conceptual)
* Convert output to level-order arrays for debugging

Tell me how you want to proceed.