---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 95: Unique Binary Search Trees II

## LeetCode 95 — Unique Binary Search Trees II

---

### Problem Statement

Given an integer `n`, return **all structurally unique BSTs (Binary Search Trees)** that store values `1` to `n`.

Each BST must satisfy:

* Left subtree values `< root`
* Right subtree values `> root`

Return the answer in **any order**.

**Constraints**

* `1 ≤ n ≤ 8`

**Example**

```text
Input: n = 3
Output:
[
  [1,null,2,null,3],
  [1,null,3,2],
  [2,1,3],
  [3,1,null,null,2],
  [3,2,null,1]
]
```

---

## Key Observations (Critical)

1. This is **not** about counting trees (that’s LeetCode 96), but **constructing all trees**.
2. For any range `[start, end]`:

   * Choose each value `i` in `[start, end]` as root
   * Left subtree comes from `[start, i-1]`
   * Right subtree comes from `[i+1, end]`
3. **Cartesian product**:

   * Every left subtree can pair with every right subtree
4. This is **recursive tree construction + backtracking**
5. Same subproblems repeat → **memoization (DP)** is highly effective

---

## Core Recursive Definition

Let:

```
build(start, end) → list of all BSTs using values [start..end]
```

Then:

```
For each root i in [start..end]:
    leftTrees  = build(start, i - 1)
    rightTrees = build(i + 1, end)

    For each L in leftTrees:
        For each R in rightTrees:
            root = TreeNode(i)
            root.left = L
            root.right = R
            add root to result
```

---

## Python 3 Solution (with Typing + Memoization)

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

            all_trees: List[Optional[TreeNode]] = []

            for root_val in range(start, end + 1):
                left_trees = build(start, root_val - 1)
                right_trees = build(root_val + 1, end)

                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(root_val)
                        root.left = left
                        root.right = right
                        all_trees.append(root)

            memo[(start, end)] = all_trees
            return all_trees

        return build(1, n)
```

---

## Example Explanation (`n = 3`)

Values: `{1, 2, 3}`

---

### Case 1: Root = 1

* Left: `[]`
* Right: trees from `[2,3]`

Produces:

```
1           1
 \           \
  2           3
   \         /
    3       2
```

---

### Case 2: Root = 2

* Left: `[1]`
* Right: `[3]`

Produces:

```
2
  / \
 1   3
```

---

### Case 3: Root = 3

* Left: trees from `[1,2]`
* Right: `[]`

Produces:

```
3          3
   /          /
  1          2
   \        /
    2      1
```

---

## Backtracking Tree Structure (Construction Flow)

![https://i.ytimg.com/vi/m907FlQa2Yc/sddefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/m907FlQa2Yc/sddefault.jpg?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/20200206162306/Untitled-Diagram311.jpg?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20200206162306/Untitled-Diagram311.jpg?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/20250904151404252799/bst2.webp?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20250904151404252799/bst2.webp?utm_source=chatgpt.com)

### Conceptual Recursion Tree (`build(1,3)`)

```
build(1,3)
 ├─ root=1
 │   ├─ left: build(1,0) → [None]
 │   └─ right: build(2,3)
 │       ├─ root=2
 │       │   └─ right=3
 │       └─ root=3
 │           └─ left=2
 │
 ├─ root=2
 │   ├─ left: build(1,1)
 │   └─ right: build(3,3)
 │
 └─ root=3
     ├─ left: build(1,2)
     │   ├─ root=1
     │   └─ root=2
     └─ right: build(4,3) → [None]
```

---

## Why `[None]` Is Returned for Empty Range

This is **extremely important**.

If `start > end`:

```
return [None]
```

Reason:

* Allows pairing:

  * `(root.left = None)`
  * `(root.right = None)`
* Without this, valid BSTs would never be formed

---

## Complexity Analysis

* **Number of trees:** Catalan number `Cₙ`
* **Time Complexity:** `O(Cₙ)`
* **Space Complexity:** `O(Cₙ)` (memo + result storage)

---

## Pattern Recognition

This problem is a **canonical example** of:

* Divide & Conquer on intervals
* Tree DP
* Cartesian product of sub-results

Related problems:

* LC 96 — Count Unique BSTs
* LC 241 — Different Ways to Add Parentheses
* LC 894 — All Possible Full Binary Trees

---

## One-Line Interview Explanation

> “For every value as root, we recursively generate all left and right subtrees from valid ranges and combine them using Cartesian product to construct all unique BSTs.”

---

If you want next, I can:

* Show **LC 95 vs LC 96** side-by-side
* Serialize these trees for easy visualization
* Explain **why Catalan numbers appear here**
* Convert this into a **generic interval-DP template**

Cartesian product: Every left subtree can pair with every right subtree. Expalin this with example

Below is a **clear, concrete explanation** of *Cartesian product* in the context of **LeetCode 95 (Unique BSTs)**, with a **step-by-step example** and a **tree-pairing view**.

---

## What “Cartesian Product” Means Here

In general math:

```
A × B = all possible pairs (a, b) where a ∈ A and b ∈ B
```

In **LeetCode 95**:

* `A` = all possible **left subtrees**
* `B` = all possible **right subtrees**

**Every left subtree must be paired with every right subtree** to form all valid BSTs for a chosen root.

---

## Concrete Example (n = 3)

### Choose root = 2

Remaining values:

```
Left range  = [1]
Right range = [3]
```

---

### Step 1: Generate subtrees

**Left subtrees from [1]:**

```
L = [  Tree(1)  ]
```

**Right subtrees from [3]:**

```
R = [  Tree(3)  ]
```

---

### Step 2: Cartesian Product L × R

```
L × R = [
  (Tree(1), Tree(3))
]
```

### Construct BST

```
2
   / \
  1   3
```

Only **one pairing**, so only **one tree**.

---

## More Interesting Example (n = 4, root = 2)

### Values

```
Left range  = [1]
Right range = [3,4]
```

---

### Left Subtrees

```
L = [
  Tree(1)
]
```

### Right Subtrees

From `[3,4]`, two possible BSTs:

```
R1:   3        R2:   4
        \          /
         4        3
```

So:

```
R = [R1, R2]
```

---

### Cartesian Product L × R

```
(Tree(1), R1)
(Tree(1), R2)
```

### Resulting Trees

```
Tree A:              Tree B:

    2                    2
   / \                  / \
  1   3                1   4
       \                  /
        4                3
```

**1 left × 2 right = 2 BSTs**

---

## Larger Example (Key Insight)

If:

```
Left subtrees  = 3
Right subtrees = 4
```

Then:

```
Total BSTs = 3 × 4 = 12
```

This multiplication is **not optional** — skipping any pairing would miss valid BSTs.

---

## Visual Pairing Table (Cartesian Product)

| Left | Right | Resulting Tree |
| --- | --- | --- |
| L1 | R1 | Tree |
| L1 | R2 | Tree |
| L2 | R1 | Tree |
| L2 | R2 | Tree |

Each row is **one unique BST**.

---

## Why `[None]` Is Part of the Cartesian Product

When a subtree is empty:

```
Left = [None]
```

Example (root = 1, n = 3):

```
Left  = [None]
Right = [Tree(2→3), Tree(3←2)]
```

Cartesian product:

```
(None, R1)
(None, R2)
```

Produces:

```
1        1
 \        \
  2        3
   \      /
    3    2
```

Without `[None]`, **no trees would be built**.

---

## Code Mapping (Exact Line)

```
for left in left_trees:
    for right in right_trees:
        root = TreeNode(root_val)
        root.left = left
        root.right = right
        all_trees.append(root)
```

This **double loop IS the Cartesian product**.

---

## Mental Model (Important)

> “Fix the root.  
> Try **every possible left shape** with **every possible right shape**.”

That is exactly what “Cartesian product” means in this problem.

---

## One-Line Interview Explanation

> “For a chosen root, we generate all possible left and right subtrees and combine them using Cartesian product so that every left structure is paired with every right structure.”

---

If you want next:

* Why this leads to **Catalan numbers**
* A **diagram of all pairings for n = 3**
* How this pattern appears in **LC 241 / LC 894**