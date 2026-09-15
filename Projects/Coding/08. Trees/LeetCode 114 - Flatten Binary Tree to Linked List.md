---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 114: Flatten Binary Tree to Linked List"
tags:
  - leetcode
  - coding
  - trees
  - linked-list
  - dfs
  - morris-traversal
  - amazon
  - google
---

# LeetCode 114: Flatten Binary Tree to Linked List

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Tree Pointer Manipulation / Morris-style In-place Traversal / Reverse Pre-order DFS

---

### Problem Statement

Given the `root` of a binary tree, flatten the tree into a "linked list":
- The "linked list" should use the same `TreeNode` class where the `right` child pointer points to the next node in the list and the `left` child pointer is always `null`.
- The "linked list" should be in the same order as a **pre-order traversal** of the binary tree.

You must modify the tree **in-place**.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `None` (modify `root` in-place)
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 2000]$.
  - $-100 \le \text{Node.val} \le 100$
- **Follow-up:** Can you flatten the tree in-place with $\mathcal{O}(1)$ extra space?

---

### Key Idea & Intuition

- **Pre-order Traversal Order:**
  - Pre-order is `Root -> Left Subtree -> Right Subtree`.
  - In a flattened chain, the entire left subtree must be inserted between `Root` and `Root.right`.
  - Specifically, the **rightmost leaf** of the left subtree is the node that immediately precedes `Root.right` in pre-order traversal!
- **Optimal $\mathcal{O}(1)$ Auxiliary Space (Morris-style Relinking):**
  1. Iterate `curr` starting from `root`.
  2. If `curr.left` exists:
     - Find the rightmost node of `curr.left` (let's call it `predecessor`).
     - Point `predecessor.right = curr.right` (the original right subtree will now execute after the entire left subtree finishes).
     - Move `curr.left` to `curr.right`: `curr.right = curr.left`.
     - Clear left pointer: `curr.left = None`.
  3. Move forward along the right chain: `curr = curr.right`.
  4. Repeat until `curr` is `None`. Every node and edge is visited at most twice $\implies \mathcal{O}(N)$ time and $\mathcal{O}(1)$ auxiliary space!

---

### Solution Approach (Step-by-Step - $\mathcal{O}(1)$ Space)

1. Set `curr = root`.
2. While `curr is not None`:
   - If `curr.left` is present:
     - Find the predecessor: `pred = curr.left`.
     - While `pred.right is not None`: `pred = pred.right`.
     - Rewire: `pred.right = curr.right`.
     - Move left subtree to right: `curr.right = curr.left`.
     - Set `curr.left = None`.
   - Advance: `curr = curr.right`.

---

### Visual Algorithm Walkthrough

```
Original Tree:
        1
       / \
      2   5
     / \   \
    3   4   6

Step 1: curr = 1. Left child exists (2).
        Find predecessor in left subtree:
        From 2, traverse right -> pred = 4.
        Rewire pred.right = 1.right (5):
        4.right = 5
        Move left subtree to right: 1.right = 2, 1.left = None.

        1
         \
          2
         / \
        3   4
             \
              5
               \
                6

Step 2: curr = 2. Left child exists (3).
        pred = 3 (pred.right is None).
        Rewire pred.right = 2.right (4):
        3.right = 4
        Move left subtree to right: 2.right = 3, 2.left = None.

        1
         \
          2
           \
            3
             \
              4
               \
                5
                 \
                  6

Step 3..6: curr visits 3, 4, 5, 6. None have left children. Finished!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Binary Tree
- **Input:** `root = [1,2,5,3,4,null,6]`
- **Step Trace:**
  | Iteration | `curr` | Has Left? | `pred` | Rewiring Action | Next `curr` |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 1 | 1 | Yes (2) | 4 | `4.right = 5`, `1.right = 2`, `1.left = None` | 2 |
  | 2 | 2 | Yes (3) | 3 | `3.right = 4`, `2.right = 3`, `2.left = None` | 3 |
  | 3 | 3 | No | - | None | 4 |
  | 4 | 4 | No | - | None | 5 |
  | 5 | 5 | No | - | None | 6 |
  | 6 | 6 | No | - | None | Done |
- **Output:** `[1,null,2,null,3,null,4,null,5,null,6]`

#### Example 2: Empty Tree or Single Node
- **Input:** `root = []` $\implies$ Loop terminates immediately, outputs `[]`.
- **Input:** `root = [0]` $\implies$ `curr.left` is null, loop finishes, outputs `[0]`.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed - $\mathcal{O}(1)$ Space)
```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        curr = root
        
        while curr:
            if curr.left:
                # Find the rightmost node of the left subtree
                pred = curr.left
                while pred.right:
                    pred = pred.right
                
                # Rewire predecessor's right to current's right
                pred.right = curr.right
                # Shift left subtree to right and nullify left
                curr.right = curr.left
                curr.left = None
            
            curr = curr.right
```

#### 2. C++ (C++17 / STL - $\mathcal{O}(1)$ Space)
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
    void flatten(TreeNode* root) {
        TreeNode* curr = root;

        while (curr != nullptr) {
            if (curr->left != nullptr) {
                // Find rightmost node in the left subtree
                TreeNode* pred = curr->left;
                while (pred->right != nullptr) {
                    pred = pred->right;
                }

                // Connect predecessor's right to current's right
                pred->right = curr->right;
                // Swap left to right and set left to null
                curr->right = curr->left;
                curr->left = nullptr;
            }
            curr = curr->right;
        }
    }
};
```

#### 3. Java (Modern, Typed - $\mathcal{O}(1)$ Space)
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
    public void flatten(TreeNode root) {
        TreeNode curr = root;

        while (curr != null) {
            if (curr.left != null) {
                // Find the rightmost node in the left subtree
                TreeNode pred = curr.left;
                while (pred.right != null) {
                    pred = pred.right;
                }

                // Splice original right subtree onto predecessor
                pred.right = curr.right;
                curr.right = curr.left;
                curr.left = null;
            }
            curr = curr.right;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Although there is a nested while-loop to find the predecessor, each edge in the tree is traversed at most twice (once to find the predecessor, once when `curr` advances through it).
- **Space Complexity:** $\mathcal{O}(1)$ — No recursion stack and no auxiliary data structures are allocated. Strictly constant space.

---

### Takeaway Pattern & Interview Traps

1. **Forgetting to Nullify Left Pointer:** Interviewers strictly check `curr.left == null`. Failing to set `curr.left = None` results in invalid tree structure errors.
2. **Reverse Pre-order DFS Alternative:** An elegant recursive solution traverses `right -> left -> root` (reverse pre-order) while maintaining a global `prev` pointer:
   ```python
   def flatten(self, root):
       prev = None
       def dfs(node):
           nonlocal prev
           if not node: return
           dfs(node.right)
           dfs(node.left)
           node.right = prev
           node.left = None
           prev = node
       dfs(root)
   ```
   This takes $\mathcal{O}(H)$ stack space. Be prepared to explain both the $\mathcal{O}(H)$ recursive and the $\mathcal{O}(1)$ Morris approaches.