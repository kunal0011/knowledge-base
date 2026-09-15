---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 222: Count Complete Tree Nodes"
tags:
  - leetcode
  - coding
  - trees
  - binary-search
  - divide-and-conquer
  - amazon
  - google
---

# LeetCode 222: Count Complete Tree Nodes

**Target Companies:** Google (Classic Sub-Linear Tree Algorithm), Amazon, Meta  
**Difficulty:** Easy / Medium  
**Topic:** Complete Binary Tree / Divide & Conquer / Sub-linear $\mathcal{O}(\log^2 N)$ Traversal

---

### Problem Statement

Given the `root` of a **complete** binary tree, return the number of the nodes in the tree.

According to **Wikipedia**, every level, except possibly the last, is completely filled in a complete binary tree, and all nodes in the last level are as far left as possible. It can have between $1$ and $2^h$ nodes inclusive at the last level $h$.

Design an algorithm that runs in less than $\mathcal{O}(n)$ time complexity.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — Total number of nodes in the complete binary tree.
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 5 \times 10^4]$.
  - $0 \le \text{Node.val} \le 5 \times 10^4$
  - The tree is guaranteed to be **complete**.

---

### Key Idea & Intuition

- **Properties of a Complete Binary Tree:**
  - A perfect binary tree of height $h$ contains exactly $2^h - 1$ nodes.
  - In a complete binary tree, at least one of the two subtrees (left or right) is guaranteed to be a **perfect binary tree**!
- **Height Comparison Subtree Pruning:**
  - Define `get_depth(node)`: Traverse strictly along `node.left` to the deepest leaf in $\mathcal{O}(h)$ time.
  - Let $h_L = \text{get\_depth}(node.left)$ and $h_R = \text{get\_depth}(node.right)$:
    1. **If $h_L == h_R$:**
       - The left subtree is a **perfect binary tree** of height $h_L$.
       - The last level has reached into the right subtree.
       - Nodes in left subtree $+$ current root $= (2^{h_L} - 1) + 1 = 2^{h_L} = (1 \ll h_L)$.
       - Recurse only on the right subtree:
         $$\text{count}(node) = 2^{h_L} + \text{count}(node.right)$$
    2. **If $h_L > h_R$ ($h_L = h_R + 1$):**
       - The right subtree is a **perfect binary tree** of height $h_R$.
       - The last level has NOT reached the right subtree (the right subtree is one level shorter).
       - Nodes in right subtree $+$ current root $= (2^{h_R} - 1) + 1 = 2^{h_R} = (1 \ll h_R)$.
       - Recurse only on the left subtree:
         $$\text{count}(node) = 2^{h_R} + \text{count}(node.left)$$
- **Sub-linear Complexity:**
  - In each step, we calculate heights in $\mathcal{O}(\log N)$ and discard one half of the tree.
  - Recurrence: $T(N) = T(N/2) + \mathcal{O}(\log N) \implies \mathcal{O}(\log^2 N)$ total time!

---

### Solution Approach (Step-by-Step)

1. Base case: If `root is None`, return `0`.
2. Compute `left_height = get_depth(root.left)` and `right_height = get_depth(root.right)`:
   - `get_depth(node)`: Follow `node = node.left` until `None`, counting steps.
3. If `left_height == right_height`:
   - Left subtree is perfect: return `(1 << left_height) + countNodes(root.right)`.
4. If `left_height > right_height`:
   - Right subtree is perfect: return `(1 << right_height) + countNodes(root.left)`.

---

### Visual Algorithm Walkthrough

```
Case 1: left_height == right_height (h_L = 2, h_R = 2)

             1               <- Root
           /   \
          2     3            <- Both subtrees reach height 2
         / \   /
        4   5 6

- Left subtree (2, 4, 5) is PERFECT (height 2).
- Count left subtree + root = 2^2 = 4 nodes (1, 2, 4, 5).
- Recurse on Right child (3): countNodes(3).

-----------------------------------------------------------

Case 2: left_height > right_height (h_L = 2, h_R = 1)

             1               <- Root
           /   \
          2     3            <- Right subtree does not reach height 2
         /
        4

- Right subtree (3) is PERFECT (height 1).
- Count right subtree + root = 2^1 = 2 nodes (1, 3).
- Recurse on Left child (2): countNodes(2).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Full Complete Tree
- **Input:** `root = [1,2,3,4,5,6]`
- **Step Trace:**
  | Step | Node | $h_L$ | $h_R$ | Condition | Known Nodes | Next Recurse Node |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | 1 | 1 | 2 | 2 | $h_L == h_R$ | $2^2 = 4$ | Node 3 |
  | 2 | 3 | 1 | 0 | $h_L > h_R$ | $2^0 = 1$ | Node 6 |
  | 3 | 6 | 0 | 0 | $h_L == h_R$ | $2^0 = 1$ | `None` |
  | 4 | `None` | - | - | Base case | 0 | Terminate |
- **Total Nodes:** $4 + 1 + 1 + 0 = 6$

#### Example 2: Empty Tree
- **Input:** `root = []`
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
            
        def get_depth(node: Optional[TreeNode]) -> int:
            d = 0
            while node:
                d += 1
                node = node.left
            return d
            
        left_h = get_depth(root.left)
        right_h = get_depth(root.right)
        
        if left_h == right_h:
            # Left subtree is perfect of height left_h
            return (1 << left_h) + self.countNodes(root.right)
        else:
            # Right subtree is perfect of height right_h
            return (1 << right_h) + self.countNodes(root.left)
```

#### 2. C++ (C++17 / STL)
```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;

        int leftH = getDepth(root->left);
        int rightH = getDepth(root->right);

        if (leftH == rightH) {
            // Left subtree is a perfect binary tree of height leftH
            return (1 << leftH) + countNodes(root->right);
        } else {
            // Right subtree is a perfect binary tree of height rightH
            return (1 << rightH) + countNodes(root->left);
        }
    }

private:
    int getDepth(TreeNode* node) {
        int depth = 0;
        while (node) {
            depth++;
            node = node->left;
        }
        return depth;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode() {}
    TreeNode(int val) { this.val = val; }
    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

class Solution {
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }

        int leftH = getDepth(root.left);
        int rightH = getDepth(root.right);

        if (leftH == rightH) {
            // Left subtree is perfect binary tree
            return (1 << leftH) + countNodes(root.right);
        } else {
            // Right subtree is perfect binary tree
            return (1 << rightH) + countNodes(root.left);
        }
    }

    private int getDepth(TreeNode node) {
        int depth = 0;
        while (node != null) {
            depth++;
            node = node.left;
        }
        return depth;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\log^2 N)$ or $\mathcal{O}(H^2)$ — Tree height is $H = \log N$. At each level of recursion, we compute depths down the left spine in $\mathcal{O}(H)$ time, and we recurse down only one child. Total calls $= H$, total time $= \mathcal{O}(H \times H) = \mathcal{O}(\log^2 N)$. For $N = 50000$, $\log_2(50000) \approx 16 \implies \approx 256$ operations, dramatically faster than naive $\mathcal{O}(N)$ traversal.
- **Space Complexity:** $\mathcal{O}(H) = \mathcal{O}(\log N)$ — Recursion call stack depth bounded by tree height $H$.

---

### Takeaway Pattern & Interview Traps

1. **Bitwise Power of 2:** Use `1 << h` to compute $2^h$ in $\mathcal{O}(1)$ time with zero floating point errors.
2. **Left Spine Invariant:** In a complete binary tree, the maximum depth of any subtree is always reached by greedily traversing `node = node.left`. There is no need to check right children when measuring maximum subtree height.
3. **Interview Sub-linear Clarification:** If an interviewer asks: *"Can you do better than $\mathcal{O}(N)$?"*, immediately mention the complete tree property allows binary search on the leaves or divide-and-conquer height comparison in $\mathcal{O}(\log^2 N)$ time.