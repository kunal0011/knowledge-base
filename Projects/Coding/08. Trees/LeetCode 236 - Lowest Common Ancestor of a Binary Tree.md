---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 236: Lowest Common Ancestor of a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - amazon
  - google
---

# LeetCode 236: Lowest Common Ancestor of a Binary Tree

**Target Companies:** Amazon (Top #1 Tree), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Post-Order DFS Bottom-Up Propagation

---

### Problem Statement

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes $p$ and $q$ as the lowest node in $T$ that has both $p$ and $q$ as descendants (where we allow **a node to be a descendant of itself**).”

---

### Input & Output Formats & Constraints

- **Input:** `root: TreeNode`, `p: TreeNode`, `q: TreeNode`
- **Output:** `TreeNode` (the lowest common ancestor node reference)
- **Constraints:**
  - The number of nodes in the tree is in the range $[2, 10^5]$.
  - $-10^9 \le \text{Node.val} \le 10^9$
  - All `Node.val` are **unique**.
  - $p \ne q$, and both $p$ and $q$ exist in the tree.

---

### Key Idea & Intuition

- **Bottom-Up Post-Order DFS:**
  - If the current node is `None`, return `None`.
  - If the current node is `p` or `q`, we have found one of the targets! Return `root`.
  - Recursively search the left and right subtrees:
    - `left = lowestCommonAncestor(root.left, p, q)`
    - `right = lowestCommonAncestor(root.right, p, q)`
- **Three Core Propagation Cases:**
  1. **Both `left` and `right` are non-null:** One target is in the left subtree, and the other is in the right subtree. Therefore, `root` **is the Lowest Common Ancestor**! Return `root`.
  2. **Only one side is non-null:** Both targets are in the same subtree, or one is an ancestor of the other. Return the non-null side.
  3. **Both are null:** Neither target exists in this subtree. Return `None`.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
            
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right:
            return root
        return left if left else right
```

#### 2. C++ (C++17 / STL)
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
        if (!root || root == p || root == q) return root;

        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);

        if (left && right) return root;
        return left ? left : right;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int x) { val = x; }
}

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left != null && right != null) return root;
        return left != null ? left : right;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — In the worst case, visits all $N$ nodes of the tree.
- **Space Complexity:** $O(H)$ where $H$ is tree height (recursion stack depth; $O(N)$ worst case, $O(\log N)$ for balanced tree).
