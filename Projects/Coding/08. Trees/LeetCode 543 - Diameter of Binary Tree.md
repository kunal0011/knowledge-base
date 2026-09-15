---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 543: Diameter of Binary Tree"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 543: Diameter of Binary Tree

Below is a **complete, structured explanation** of **LeetCode 543 – Diameter of Binary Tree**, aligned with your preferred format.

---

## LeetCode 543: Diameter of Binary Tree

---

## 1. Problem Statement

You are given the `root` of a binary tree.

The **diameter** of a binary tree is defined as the **length of the longest path between any two nodes** in the tree.  
This path **may or may not pass through the root**.

* The length of a path is measured by the **number of edges** between nodes.
* A path does not need to go downward only; it can go from left subtree → parent → right subtree.

### Return

The **diameter** of the binary tree.

---

## 2. Key Observations

1. **Diameter at a node**

   * If a path passes through a node, its length is:

     ```
     height(left subtree) + height(right subtree)
     ```
2. **Global nature of diameter**

   * The longest diameter may:

     * pass through the root
     * lie entirely in the left subtree
     * lie entirely in the right subtree
3. **Height vs Diameter**

   * Height of a node = max depth from that node to a leaf
   * Diameter is **not necessarily related to the height of the root**
4. **Brute force is inefficient**

   * Computing height repeatedly for every node leads to **O(n²)** time

---

## 3. Core Concept Used

### Postorder DFS (Bottom-Up)

* Traverse left subtree
* Traverse right subtree
* Use returned heights to:

  * update diameter
  * compute current node height

This ensures **each node is processed exactly once**.

---

## 4. Algorithm (High-Level)

1. Initialize a variable `diameter = 0`
2. Define a DFS function that:

   * Returns height of the current node
   * Updates diameter as:

     ```
     diameter = max(diameter, left_height + right_height)
     ```
3. Call DFS on root
4. Return diameter

---

## 5. Python 3 Solution (with typing)

```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: Optional["TreeNode"] = None, right: Optional["TreeNode"] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left_height = dfs(node.left)
            right_height = dfs(node.right)

            # Update diameter at this node
            self.diameter = max(self.diameter, left_height + right_height)

            # Return height of this node
            return 1 + max(left_height, right_height)

        dfs(root)
        return self.diameter
```

---

## 6. Worked Example

### Input Tree

```
        1
       / \
      2   3
     / \
    4   5
```

---

### Step-by-Step DFS Computation

| Node | Left Height | Right Height | Diameter Update | Return Height |
| --- | --- | --- | --- | --- |
| 4 | 0 | 0 | max(0, 0) = 0 | 1 |
| 5 | 0 | 0 | max(0, 0) = 0 | 1 |
| 2 | 1 | 1 | max(0, 2) = 2 | 2 |
| 3 | 0 | 0 | max(2, 0) = 2 | 1 |
| 1 | 2 | 1 | max(2, 3) = 3 | 3 |

---

### Final Answer

```
Diameter = 3
```

---

## 7. Conceptual Illustration (Path View)

Longest path:

```
4 → 2 → 1 → 3
```

* Number of edges = **3**
* Path passes through the root, but that is **not required by definition**

---

## 8. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(h)** (recursion stack, h = height of tree) |

---

## 9. Common Mistakes

1. **Returning diameter instead of height from DFS**
2. **Counting nodes instead of edges**
3. **Assuming diameter must pass through root**
4. **Using preorder instead of postorder**

---

## 10. Pattern Recognition (Interview Insight)

This problem is a classic example of:

> **“Compute something global while returning something local”**

* Local return → height
* Global update → diameter

This same pattern appears in:

* Maximum Path Sum
* Balanced Binary Tree
* Binary Tree Tilt

---

If you want, I can also:

* Convert this into an **iterative solution**
* Show a **dry-run with recursion stack**
* Compare with **Maximum Path Sum (LC 124)** to highlight differences