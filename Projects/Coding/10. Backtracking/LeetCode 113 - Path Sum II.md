---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 113: Path Sum II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 113: Path Sum II

## LeetCode 113 — Path Sum II

---

### Problem Statement

Given the `root` of a binary tree and an integer `targetSum`, return **all root-to-leaf paths** where the sum of the node values along the path equals `targetSum`.

A **leaf** is a node with no children.

**Constraints**

* Number of nodes: `0 ≤ n ≤ 5000`
* `-1000 ≤ Node.val ≤ 1000`
* `-1000 ≤ targetSum ≤ 1000`

---

### Example

**Input**

```
root = [5,4,8,11,null,13,4,7,2,null,null,5,1]
targetSum = 22
```

**Output**

```
[
  [5,4,11,2],
  [5,8,4,5]
]
```

---

## Key Observations

1. The path must:

   * Start at the **root**
   * End at a **leaf**
2. This is **not** a prefix or partial path problem; intermediate sums do not count.
3. Tree traversal is naturally **Depth-First Search (DFS)**.
4. Backtracking is required because:

   * We reuse the same path list while exploring different branches.
5. We must check the sum **only at leaf nodes**.

---

## Core Idea (Backtracking on Tree)

At each node:

* Add the node value to the current path.
* Subtract the node value from the remaining sum.
* If the node is a leaf:

  * Check if remaining sum == node value.
* Recurse left and right.
* Backtrack (remove the node from path).

This follows the classic:

> **choose → explore → un-choose**

---

## Python 3 Solution (with Typing)

```python
from typing import List, Optional

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
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def dfs(node: Optional[TreeNode], remaining: int) -> None:
            if not node:
                return

            # choose
            path.append(node.val)
            remaining -= node.val

            # check leaf
            if not node.left and not node.right:
                if remaining == 0:
                    result.append(path.copy())
            else:
                # explore
                dfs(node.left, remaining)
                dfs(node.right, remaining)

            # un-choose (backtrack)
            path.pop()

        dfs(root, targetSum)
        return result
```

---

## Example Walkthrough

Target = `22`

### Path 1

```
5 → 4 → 11 → 2
Sum = 22 ✓
```

### Path 2

```
5 → 8 → 4 → 5
Sum = 22 ✓
```

### Invalid Path (discarded)

```
5 → 8 → 4 → 1
Sum = 18 ✗
```

Only **root-to-leaf** paths are considered.

---

## Backtracking Tree Structure (Navigation)

![https://miro.medium.com/v2/resize%3Afit%3A592/0%2AsYSK2E4brBcV-Y2k.jpg?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A592/0%2AsYSK2E4brBcV-Y2k.jpg?utm_source=chatgpt.com)

![https://assets.algo.monster/liteProblems/path_sum_ii.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/XgGnHnicu5mZUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw6JD7L0TXKpNHGNMnUKNIrPDS_zT7PMcXdMCfbOMIiKCMo2TXFJTC_zykn29jIty_dVKwYAR3UlsQ?utm_source=chatgpt.com)

![https://substackcdn.com/image/fetch/%24s_%21OuQ1%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F440177b1-836d-4940-b159-671405dc2299_800x450.png?utm_source=chatgpt.com](https://substackcdn.com/image/fetch/%24s_%21OuQ1%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F440177b1-836d-4940-b159-671405dc2299_800x450.png?utm_source=chatgpt.com)

### Conceptual Backtracking Tree

```
5
                    -------------------------
                   4                         8
             -------------             --------------
           11           X             13              4
        --------                      X            --------
       7        2                                   5        1
       X        ✓                                   ✓        X
```

Legend:

* ✓ valid path (sum = 22)
* X invalid or pruned

---

## Why Backtracking Is Required

`path` is a **mutable list** shared across recursive calls.

Without:

```
path.pop()
```

values from one branch would incorrectly appear in another branch.

---

## Complexity Analysis

* **Time Complexity:** `O(n)`  
  (Each node is visited once)
* **Space Complexity:** `O(h)`  
  where `h` = height of the tree (recursion stack + path)

---

## Pattern Recognition

This problem is a classic example of:

**Tree DFS + Path Backtracking**

Related problems:

* LeetCode 112 — Path Sum (boolean check)
* LeetCode 257 — Binary Tree Paths
* LeetCode 437 — Path Sum III (prefix sum variant)

---

### One-Line Interview Explanation

> “We perform DFS from the root, track the current path and remaining sum, and whenever we reach a leaf with remaining sum zero, we record the path.”

If you want, I can next:

* Compare **LC 112 vs LC 113**
* Show how this changes for **Path Sum III**
* Explain how to convert this to an **iterative stack-based DFS**