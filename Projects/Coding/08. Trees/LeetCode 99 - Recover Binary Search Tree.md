---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 99: Recover Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
  - bst
  - in-order
  - morris-traversal
  - amazon
  - google
---

# LeetCode 99: Recover Binary Search Tree

**Target Companies:** Google, Amazon, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** BST In-Order Traversal / Monotonic Inversion Detection / In-Place Value Swap

---

### Problem Statement

You are given the `root` of a binary search tree (BST), where the values of **exactly two nodes** of the tree were swapped by mistake. *Recover the tree without changing its structure*.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `None` (Modify tree in-place)
- **Constraints:**
  - The number of nodes in the tree is in the range $[2, 1000]$.
  - $-2^{31} \le \text{Node.val} \le 2^{31} - 1$
- **Follow-up:** A solution using $\mathcal{O}(n)$ space is pretty straight-forward. Could you devise a constant $\mathcal{O}(1)$ space solution?

---

### Key Idea & Intuition

- **In-Order Traversal Invariant:**
  - An in-order traversal (`Left -> Root -> Right`) of a valid BST always yields a **strictly monotonically increasing** sequence:
    $$A_1 < A_2 < \dots < A_{k-1} < A_k < \dots < A_n$$
  - When exactly two node values are swapped, the sorted order is violated by one or two inversions (points where $prev.\text{val} > curr.\text{val}$):
    1. **Case 1: Swapped nodes are NOT adjacent:**
       - Example: sorted $[1, 2, 3, 4, 5]$ with $2$ and $5$ swapped becomes $[1, \mathbf{5}, 3, 4, \mathbf{2}]$.
       - Two inversions occur:
         - First drop: $5 > 3 \implies \text{first} = 5, \text{second} = 3$.
         - Second drop: $4 > 2 \implies \text{second} = 2$.
       - The swapped nodes are `first` (from the first drop) and `second` (from the second drop).
    2. **Case 2: Swapped nodes ARE adjacent:**
       - Example: sorted $[1, 2, 3, 4]$ with $2$ and $3$ swapped becomes $[1, \mathbf{3}, \mathbf{2}, 4]$.
       - Only **one** inversion occurs:
         - Drop: $3 > 2 \implies \text{first} = 3, \text{second} = 2$.
- **Algorithm:**
  - Perform in-order traversal keeping a pointer to the previously visited node `prev`.
  - Whenever $prev.\text{val} > curr.\text{val}$:
    - If `first` is not yet set: `first = prev`, `second = curr`.
    - If `first` is already set: `second = curr`.
  - After traversal, swap `first.val` and `second.val`.

---

### Solution Approach (Step-by-Step)

1. Maintain pointers: `first = None`, `second = None`, `prev = None`.
2. Define recursive in-order traversal `inorder(node)`:
   - If `not node`: return.
   - `inorder(node.left)`
   - Detect inversion:
     - If `prev` and `prev.val > node.val`:
       - If `not first`:
         - `first = prev`
       - `second = node`  (updated in both single and double inversion cases)
     - Update `prev = node`
   - `inorder(node.right)`
3. Run `inorder(root)`.
4. Swap values: `first.val, second.val = second.val, first.val`.

---

### Visual Algorithm Walkthrough

```
Example 1: Non-adjacent Swap
Original valid in-order: [1, 2, 3, 4, 5]
Swapped 2 and 5:         [1, 5, 3, 4, 2]

Traverse in-order:
- prev = 1, curr = 5 -> (1 < 5) OK. prev = 5
- prev = 5, curr = 3 -> (5 > 3) INVERSION #1!
  - first = prev = 5
  - second = curr = 3
  - prev = 3
- prev = 3, curr = 4 -> (3 < 4) OK. prev = 4
- prev = 4, curr = 2 -> (4 > 2) INVERSION #2!
  - second = curr = 2
  - prev = 2

Final nodes to swap: first (5) and second (2).
Swap values: [1, 2, 3, 4, 5] -> Restored!

----------------------------------------------------

Example 2: Adjacent Swap
Original valid in-order: [1, 2, 3, 4]
Swapped 2 and 3:         [1, 3, 2, 4]

Traverse in-order:
- prev = 1, curr = 3 -> OK. prev = 3
- prev = 3, curr = 2 -> (3 > 2) INVERSION #1!
  - first = prev = 3
  - second = curr = 2
  - prev = 2
- prev = 2, curr = 4 -> OK. prev = 4

Only 1 inversion occurred: first (3) and second (2).
Swap values: [1, 2, 3, 4] -> Restored!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Tree
- **Input:** `root = [1,3,null,null,2]` (In-order: `[3, 2, 1]`)
- **Step Trace:**
  | Node Visited | `prev` | Comparison | `first` | `second` |
  | :--- | :--- | :--- | :--- | :--- |
  | 3 | `None` | - | `None` | `None` |
  | 2 | 3 | $3 > 2$ (Inversion 1) | Node 3 | Node 2 |
  | 1 | 2 | $2 > 1$ (Inversion 2) | Node 3 | Node 1 |
- **Swap:** Node 3 with Node 1 $\implies$ In-order restored to `[1, 2, 3]`.

#### Example 2: Adjacent Nodes Swapped
- **Input:** `root = [3,1,4,null,null,2]` (In-order: `[1, 3, 2, 4]`)
- **Step Trace:**
  - Drop at $3 > 2$: `first = 3`, `second = 2`.
  - Swap 3 and 2 $\implies$ In-order restored to `[1, 2, 3, 4]`.

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
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        first = None
        second = None
        prev = None
        
        def inorder(node: Optional[TreeNode]) -> None:
            nonlocal first, second, prev
            if not node:
                return
            
            inorder(node.left)
            
            # Detect inversion
            if prev and prev.val > node.val:
                if not first:
                    first = prev
                second = node
            prev = node
            
            inorder(node.right)
            
        inorder(root)
        
        # Swap values of the two erroneous nodes
        if first and second:
            first.val, second.val = second.val, first.val
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
    void recoverTree(TreeNode* root) {
        TreeNode* first = nullptr;
        TreeNode* second = nullptr;
        TreeNode* prev = nullptr;

        inorder(root, first, second, prev);

        if (first && second) {
            int temp = first->val;
            first->val = second->val;
            second->val = temp;
        }
    }

private:
    void inorder(TreeNode* node, TreeNode*& first, TreeNode*& second, TreeNode*& prev) {
        if (!node) return;

        inorder(node->left, first, second, prev);

        if (prev && prev->val > node->val) {
            if (!first) {
                first = prev;
            }
            second = node;
        }
        prev = node;

        inorder(node->right, first, second, prev);
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
    private TreeNode first = null;
    private TreeNode second = null;
    private TreeNode prev = null;

    public void recoverTree(TreeNode root) {
        inorder(root);

        if (first != null && second != null) {
            int temp = first.val;
            first.val = second.val;
            second.val = temp;
        }
    }

    private void inorder(TreeNode node) {
        if (node == null) {
            return;
        }

        inorder(node.left);

        if (prev != null && prev.val > node.val) {
            if (first == null) {
                first = prev;
            }
            second = node;
        }
        prev = node;

        inorder(node.right);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — The in-order traversal visits every node in the binary tree exactly once.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion call stack requires space proportional to tree height $H$ ($\mathcal{O}(\log N)$ for balanced trees, $\mathcal{O}(N)$ for skewed trees).
  *(Follow-up note: Morris Traversal achieves strictly $\mathcal{O}(1)$ auxiliary space without recursion or stack by temporarily threading leaf right pointers to predecessors).*

---

### Takeaway Pattern & Interview Traps

1. **Why `second = node` must be updated on both drops:**
   - On the first drop, `first = prev` and `second = node` correctly sets both targets in case the swapped nodes are adjacent (only 1 drop occurs).
   - If a second drop occurs later, updating `second = node` replaces `second` with the actual distant swapped element.
2. **Swap Values, Don't Rewire Pointers:** The problem statement explicitly states: *"Recover the tree without changing its structure"*. Swapping tree node pointers is error-prone and unnecessary; simply swapping `first.val` and `second.val` is the expected canonical solution.