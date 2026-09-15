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
  - dfs
  - amazon
  - google
---

# LeetCode 236: Lowest Common Ancestor of a Binary Tree

**Target Companies:** Meta (All-Time #1 Tree Classic), Google, Amazon, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Tree DFS / Post-order Traversal / Subtree LCA Propagation

---

### Problem Statement

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes `p` and `q`.

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
  - `p` and `q` will exist in the tree.

---

### Key Idea & Intuition

- **Post-Order Bottom-Up Propagation:**
  - Unlike a Binary Search Tree (where node values guide navigation), a general binary tree provides no ordering guarantees. We must search both subtrees.
  - At any given node `root`:
    1. **Base Case:** If `root` is `None`, return `None`. If `root == p` or `root == q`, return `root` (we found one of the targets).
    2. **Recursive Search:** Search both subtrees:
       - `left = lowestCommonAncestor(root.left, p, q)`
       - `right = lowestCommonAncestor(root.right, p, q)`
    3. **Combine Results:**
       - **Both non-null (`left and right`):** Node `p` was found in one subtree and `q` was found in the other subtree. Therefore, `root` is their Lowest Common Ancestor!
       - **One non-null:** Both `p` and `q` reside in that same non-null subtree, or one node is an ancestor of the other and already returned itself. Forward that non-null node upward.
       - **Both null:** Neither `p` nor `q` exists in this subtree. Return `None`.

---

### Solution Approach (Step-by-Step)

1. Base check:
   - If `root is None or root == p or root == q`: return `root`.
2. Recursively search left: `left = lowestCommonAncestor(root.left, p, q)`.
3. Recursively search right: `right = lowestCommonAncestor(root.right, p, q)`.
4. If both `left` and `right` are not `None`, return `root`.
5. Return `left if left else right`.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
         3
       /   \
      5     1
     / \   / \
    6   2 0   8
       / \
      7   4

Query: p = 5, q = 1
1. At Node 5: root == p -> returns Node 5.
2. At Node 1: root == q -> returns Node 1.
3. At Root 3:
   - left returns Node 5 (non-null)
   - right returns Node 1 (non-null)
   Both are non-null -> LCA is Node 3!

Query: p = 5, q = 4
1. At Node 5:
   - left search: Node 6 returns None.
   - right search (Node 2):
     - Node 7 returns None.
     - Node 4 matches q -> returns Node 4.
     Node 2 returns Node 4.
   Node 5 matches p, but also its descendant contains q!
   Root 5 returns itself immediately when evaluated as `root == p`, correctly designating Node 5 as LCA.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Nodes in Different Subtrees
- **Input:** `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1`
- **Tracing Table:**
  | Node | Left Result | Right Result | Combined Action | Return Value |
  | :--- | :--- | :--- | :--- | :--- |
  | 6 | `None` | `None` | Both null | `None` |
  | 7 | `None` | `None` | Both null | `None` |
  | 4 | - | - | Match `q` (4) | Node 4 |
  | 2 | `None` | Node 4 | One non-null | Node 4 |
  | 5 | - | - | Match `p` (5) | Node 5 |
  | 0 | `None` | `None` | Both null | `None` |
  | 8 | `None` | `None` | Both null | `None` |
  | 1 | - | - | Match `q` (1) | Node 1 |
  | 3 | Node 5 | Node 1 | Both non-null! | **Node 3** |
- **Output:** `Node 3`

#### Example 2: One Node is Parent of Other
- **Input:** `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4`
- **Output:** `Node 5`

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
        # Base cases: empty node, or found p, or found q
        if not root or root == p or root == q:
            return root
            
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # If both left and right return non-null, root is the LCA
        if left and right:
            return root
            
        # Otherwise, propagate the non-null result upward
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
        if (!root || root == p || root == q) {
            return root;
        }

        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);

        if (left && right) {
            return root;
        }

        return left ? left : right;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            return root;
        }

        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);

        if (left != null && right != null) {
            return root;
        }

        return left != null ? left : right;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — In the worst case, every node in the binary tree is visited once.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion call stack space corresponds to tree height $H$. For balanced trees $H = \mathcal{O}(\log N)$; for skewed trees $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Short-Circuiting Optimization:** When `root == p`, returning `root` immediately without exploring its subtrees is 100% correct because:
   - If `q` is in `p`'s subtree, `p` is indeed the LCA.
   - If `q` is outside `p`'s subtree, `p` will be forwarded upward to meet `q`'s branch at their true common ancestor.
2. **Missing Node Follow-up (LeetCode 1644):** If `p` or `q` is **not guaranteed** to exist in the tree, this standard algorithm fails (it may return `p` even if `q` does not exist). In LeetCode 1644, you must traverse all nodes and maintain counters `found_p` and `found_q` before confirming the LCA.
