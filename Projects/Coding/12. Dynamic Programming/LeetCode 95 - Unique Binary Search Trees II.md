---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 95: Unique Binary Search Trees II

**LeetCode 95 – Unique Binary Search Trees II**, with **clear state definition, transitions, DP table construction, and a worked example**.  
I will intentionally avoid jumping directly to “just recursion” and instead frame it as **DP over intervals**, which is the correct mental model for this problem.

---

## Problem Statement (LeetCode 95)

Given an integer `n`, return **all structurally unique BSTs** that store values `1 … n`.

Each BST must satisfy:

* Left subtree values `< root`
* Right subtree values `> root`

---

## Key Insight

This is **not counting** BSTs (that is LeetCode 96).  
Here, we must **construct all trees**.

The structure of a BST depends only on:

* Which value is chosen as root
* All possible left subtrees from smaller values
* All possible right subtrees from larger values

This naturally leads to **DP on ranges**.

---

## DP State Definition

### State

Let:

```
dp[l][r] = list of all unique BSTs that can be formed using values from l to r
```

Each entry stores **TreeNode roots**, not counts.

---

### Base Case

1. **Empty range**

```
l > r → dp[l][r] = [None]
```

Why?

* Needed to attach `None` as left/right child when subtree is empty.

2. **Single node**

```
l == r → dp[l][r] = [TreeNode(l)]
```

---

## State Transition

For a given range `[l, r]`:

1. Choose each value `root ∈ [l, r]` as root
2. Left subtrees come from `dp[l][root - 1]`
3. Right subtrees come from `dp[root + 1][r]`
4. Combine **every left subtree with every right subtree**

### Transition Formula

```
dp[l][r] = []
for root in range(l, r + 1):
    for left in dp[l][root - 1]:
        for right in dp[root + 1][r]:
            new_tree = TreeNode(root)
            new_tree.left = left
            new_tree.right = right
            dp[l][r].append(new_tree)
```

This is a **Cartesian product** of left and right subtree possibilities.

---

## DP Table Construction Order

Because `dp[l][r]` depends on **smaller ranges**, we fill by **increasing interval length**.

```
length = 1 → 2 → ... → n
```

For each length:

```
l = 1 to n
r = l + length - 1
```

---

## Example: n = 3

We want `dp[1][3]`.

---

### Step 1: Length = 1

```
dp[1][1] = [1]
dp[2][2] = [2]
dp[3][3] = [3]
```

(each is a single-node tree)

---

### Step 2: Length = 2

#### dp[1][2]

Roots:

* root = 1

  * left = dp[1][0] = [None]
  * right = dp[2][2] = [2]

  Tree:

  ```
    1
     \
      2
  ```
* root = 2

  * left = dp[1][1] = [1]
  * right = dp[3][2] = [None]

  Tree:

  ```
      2
     /
    1
  ```

So:

```
dp[1][2] = [ (1,null,2), (2,1,null) ]
```

---

#### dp[2][3]

Similarly:

```
dp[2][3] = [ (2,null,3), (3,2,null) ]
```

---

### Step 3: Length = 3 → dp[1][3]

Roots = 1, 2, 3

---

#### root = 1

* left = dp[1][0] = [None]
* right = dp[2][3] (2 trees)

Trees:

```
1          1
 \          \
  2          3
   \        /
    3      2
```

---

#### root = 2

* left = dp[1][1] = [1]
* right = dp[3][3] = [3]

Tree:

```
   2
  / \
 1   3
```

---

#### root = 3

* left = dp[1][2] (2 trees)
* right = dp[4][3] = [None]

Trees:

```
    3        3
   /        /
  1        2
   \      /
    2    1
```

---

### Final Result

Total trees = **5**

This matches the expected output.

---

## Python 3 DP Implementation (Typed)

```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val: int, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        if n == 0:
            return []

        # dp[l][r] = list of BSTs formed using values l..r
        dp = [[[] for _ in range(n + 2)] for _ in range(n + 2)]

        # empty ranges
        for i in range(1, n + 2):
            dp[i][i - 1] = [None]

        # build by length
        for length in range(1, n + 1):
            for l in range(1, n - length + 2):
                r = l + length - 1
                dp[l][r] = []

                for root in range(l, r + 1):
                    for left in dp[l][root - 1]:
                        for right in dp[root + 1][r]:
                            node = TreeNode(root)
                            node.left = left
                            node.right = right
                            dp[l][r].append(node)

        return dp[1][n]
```

---

## Time and Space Complexity

### Time

* Number of trees = **Catalan(n)**
* Each tree creation costs O(n) in total node creation

```
O(Catalan(n) × n)
```

### Space

* DP table stores all trees:

```
O(Catalan(n))
```

---

## Mental Model Summary

* **DP over intervals**
* State stores **objects, not numbers**
* Transition = choose root + Cartesian product of subtrees
* Empty subtree represented by `[None]`

If you want next:

* Convert this DP into **pure recursion + memo**
* Visualize this as a **backtracking tree**
* Or compare with **LeetCode 96 (counting vs constructing)**

Tell me which direction you want.