---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 235: Lowest Common Ancestor of a Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
  - bst
  - amazon
  - google
---

# LeetCode 235: Lowest Common Ancestor of a Binary Search Tree

**Target Companies:** Amazon (Top Tier Tree Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Binary Search Tree Property / LCA / Iterative Split Search ($\mathcal{O}(1)$ Space)

---

### Problem Statement

Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the **definition of LCA on Wikipedia**: “The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow **a node to be a descendant of itself**).”

---

### Input & Output Formats & Constraints

- **Input:** `root: TreeNode`, `p: TreeNode`, `q: TreeNode`
- **Output:** `TreeNode` (The Lowest Common Ancestor)
- **Constraints:**
  - The number of nodes in the tree is in the range $[2, 10^5]$.
  - $-10^9 \le \text{Node.val} \le 10^9$
  - All `Node.val` are **unique**.
  - `p != q`
  - `p` and `q` will exist in the BST.

---

### Key Idea & Intuition

- **Binary Search Tree Invariant:**
  - For any node `curr` in a BST:
    - Every value in `curr.left` is strictly smaller than `curr.val`.
    - Every value in `curr.right` is strictly greater than `curr.val`.
- **The LCA "Split Point":**
  - Starting from `curr = root`:
    1. **Both $p$ and $q$ are strictly smaller than `curr.val`:**
       - Both targets reside exclusively in the left subtree.
       - Move left: `curr = curr.left`.
    2. **Both $p$ and $q$ are strictly greater than `curr.val`:**
       - Both targets reside exclusively in the right subtree.
       - Move right: `curr = curr.right`.
    3. **Split Condition ($p.\text{val} \le curr.\text{val} \le q.\text{val}$ or vice versa):**
       - The paths to $p$ and $q$ diverge here!
       - One node is in the left subtree (or is `curr` itself), and the other is in the right subtree (or is `curr` itself).
       - Therefore, `curr` is the **Lowest Common Ancestor**!
- **$\mathcal{O}(1)$ Auxiliary Space:**
  - Because this is a guided single-path descent, we do not need recursion or backtracking; a simple while-loop achieves $\mathcal{O}(1)$ extra memory.

---

### Solution Approach (Step-by-Step)

1. Initialize pointer `curr = root`.
2. Loop while `curr` is not null:
   - If `p.val < curr.val` and `q.val < curr.val`:
     - `curr = curr.left`
   - Else if `p.val > curr.val` and `q.val > curr.val`:
     - `curr = curr.right`
   - Else:
     - Found split point (or one of the target nodes matches `curr`): return `curr`.
3. Return `None` (should not be reached under problem constraints).

---

### Visual Algorithm Walkthrough

```
BST Example:
         6
       /   \
      2     8
     / \   / \
    0   4 7   9
       / \
      3   5

Query: p = 2, q = 8
1. Start at curr = 6:
   - p.val = 2 (< 6)
   - q.val = 8 (> 6)
   - Split occurs at 6! Node 6 is the LCA.

Query: p = 2, q = 4
1. Start at curr = 6:
   - p.val = 2 (< 6) and q.val = 4 (< 6) -> Move left to 2.
2. curr = 2:
   - p.val = 2 (== curr.val) and q.val = 4 (> 2) -> Split!
   - Node 2 is the LCA (a node can be a descendant of itself).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Ancestors on Opposite Subtrees
- **Input:** `root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8`
- **Step Trace:**
  | Step | `curr.val` | `p.val` (2) vs `curr` | `q.val` (8) vs `curr` | Decision |
  | :--- | :--- | :--- | :--- | :--- |
  | 1 | 6 | $2 < 6$ (Left) | $8 > 6$ (Right) | Split point reached! Return 6 |
- **Output:** `6`

#### Example 2: One Node is Ancestor of the Other
- **Input:** `root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4`
- **Step Trace:**
  | Step | `curr.val` | `p.val` (2) vs `curr` | `q.val` (4) vs `curr` | Decision |
  | :--- | :--- | :--- | :--- | :--- |
  | 1 | 6 | $2 < 6$ | $4 < 6$ | Both smaller $\to$ Move left |
  | 2 | 2 | $2 == 2$ (Root match) | $4 > 2$ (Right) | Split / Match reached! Return 2 |
- **Output:** `2`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed - $\mathcal{O}(1)$ Space)
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        curr = root
        
        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                # Split point: p and q diverge, or one matches curr
                return curr
                
        return root
```

#### 2. C++ (C++17 / STL - $\mathcal{O}(1)$ Space)
```cpp
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* curr = root;

        while (curr != nullptr) {
            if (p->val < curr->val && q->val < curr->val) {
                curr = curr->left;
            } else if (p->val > curr->val && q->val > curr->val) {
                curr = curr->right;
            } else {
                return curr;
            }
        }

        return nullptr;
    }
};
```

#### 3. Java (Modern, Typed - $\mathcal{O}(1)$ Space)
```java
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode curr = root;

        while (curr != null) {
            if (p.val < curr.val && q.val < curr.val) {
                curr = curr.left;
            } else if (p.val > curr.val && q.val > curr.val) {
                curr = curr.right;
            } else {
                return curr;
            }
        }

        return null;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(H)$ — We traverse down a single path from root to the LCA node. In a balanced BST, $H = \mathcal{O}(\log N)$; in the worst-case degenerate BST, $H = \mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(1)$ — The iterative approach requires strictly constant extra memory with no recursion stack frames.

---

### Takeaway Pattern & Interview Traps

1. **BST vs General Binary Tree:**
   - For a general binary tree (LeetCode 236), you must search both subtrees in post-order traversal taking $\mathcal{O}(N)$ time and $\mathcal{O}(H)$ stack space.
   - For a **BST** (LeetCode 235), use the ordering property to prune an entire half of the tree at each step, operating in $\mathcal{O}(H)$ time and $\mathcal{O}(1)$ space!
2. **Descendant of Itself:** Remember that LCA allows a node to be a descendant of itself. If `curr.val == p.val`, `curr` is the LCA since `q` is guaranteed to be in its subtree.