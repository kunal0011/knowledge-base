---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 124: Binary Tree Maximum Path Sum"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 124: Binary Tree Maximum Path Sum

## LeetCode 124 — Binary Tree Maximum Path Sum

![Image](https://media.geeksforgeeks.org/wp-content/uploads/20220920163023/btreedrawio-660x461.png)

![Image](https://assets.leetcode.com/uploads/2021/01/18/pathsum1.jpg)

![Image](https://afteracademy.com/images/path-sum-in-binary-tree-fb1857ace44dccc1.png)

---

### Problem Statement

You are given the `root` of a **binary tree**.  
A **path** is defined as a sequence of nodes where each pair of adjacent nodes in the sequence has a parent–child relationship. A path **does not need to pass through the root**, but it **must contain at least one node**.

The **path sum** is the sum of the node values along the path.

**Return the maximum path sum of any path in the tree.**

---

### Key Observations

1. **A path can “turn” at a node**

   * At any node, the maximum path may include:

     * the node alone
     * node + left subtree
     * node + right subtree
     * node + left subtree + right subtree (this is where the path turns)
2. **Negative contributions should be ignored**

   * If a subtree contributes a negative sum, it is better to **cut it off**.
   * Hence, we use `max(0, left_gain)` and `max(0, right_gain)`.
3. **Two different values per node**

   * **Return value (upward path)**:  
     The maximum sum of a path that **starts at the current node and extends upward** to its parent.
   * **Global answer update (split path)**:  
     The maximum sum of a path that **uses both children and the current node**.
4. **Why global tracking is required**

   * The best path may be completely inside a subtree and never be returned upward.
   * Hence, we maintain a global maximum.

---

### Core Concept (Very Important)

At every node:

```
left_gain  = max(0, gain from left child)
right_gain = max(0, gain from right child)

path_through_node = node.val + left_gain + right_gain
```

* Update global maximum using `path_through_node`
* Return upward only:

```
node.val + max(left_gain, right_gain)
```

Because a parent can only extend **one direction**, not both.

---

### Python 3 Solution (with Typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float("-inf")

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            # Compute maximum contribution from left and right subtrees
            left_gain = max(0, dfs(node.left))
            right_gain = max(0, dfs(node.right))

            # Path that passes through the current node (can split here)
            current_path_sum = node.val + left_gain + right_gain

            # Update global maximum
            self.max_sum = max(self.max_sum, current_path_sum)

            # Return maximum path sum extending upwards
            return node.val + max(left_gain, right_gain)

        dfs(root)
        return self.max_sum
```

---

### Worked-Out Example

#### Input Tree

```
        -10
        /  \
       9   20
          /  \
         15   7
```

---

### Step-by-Step Evaluation

#### Node 9

* left = 0, right = 0
* path = `9`
* global max = `9`
* return `9`

#### Node 15

* path = `15`
* global max = `15`
* return `15`

#### Node 7

* path = `7`
* global max remains `15`
* return `7`

#### Node 20

* left\_gain = 15
* right\_gain = 7
* path through node = `20 + 15 + 7 = 42`
* global max = `42`
* return upward = `20 + max(15, 7) = 35`

#### Node -10

* left\_gain = 9
* right\_gain = 35
* path = `-10 + 9 + 35 = 34`
* global max remains `42`
* return value is irrelevant (root)

---

### Final Answer

```
Maximum Path Sum = 42
```

---

### Conceptual Illustration (Path Selection)

```
        -10
           \
            20
           /  \
         15    7

Chosen path: 15 → 20 → 7
```

* The root `-10` is excluded because it reduces the total sum.
* This highlights why paths **do not need to include the root**.

---

### Time & Space Complexity

* **Time Complexity**: `O(N)`  
  (Each node is visited exactly once)
* **Space Complexity**: `O(H)`  
  (Recursive stack, where `H` is the height of the tree)

---

### Key Takeaway

This problem is a classic example of **post-order DFS with global state**, where:

* local decisions determine return values
* global tracking captures split paths that cannot propagate upward

If you want, I can also provide:

* iterative intuition
* dry-run with negative-only trees
* or comparison with similar problems (e.g., path sum with constraints)