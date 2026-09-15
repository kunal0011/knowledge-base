---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1022: Sum of Root To Leaf Binary Numbers"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 1022: Sum of Root To Leaf Binary Numbers

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Easy  
**Topic:** Tree DFS / Bit Manipulation / Pre-order Traversal

---

### Problem Statement

You are given the `root` of a binary tree where each node has a value `0` or `1`. Each root-to-leaf path represents a binary number starting with the most significant bit at the root.

- For example, if the path is `0 -> 1 -> 1 -> 0 -> 1`, then this represents `01101` in binary, which is $13$ in decimal.

For all leaves in the tree, consider the numbers represented by the path from the root to that leaf. Return the **sum of these numbers**.

A **leaf** is a node with no children.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — The sum of all root-to-leaf binary numbers in decimal.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 1000]$.
  - `Node.val` is either `0` or `1`.
  - The test cases are generated so that the answer fits in a **32-bit** integer.

---

### Key Idea & Intuition

- **Binary Shift Invariant:**
  - As we descend down a path from parent to child, appending a bit $b \in \{0, 1\}$ corresponds to shifting the current accumulated binary value left by 1 bit and performing a bitwise OR (or addition):
    $$\text{curr} = (\text{curr} \ll 1) \mid \text{node.val} = 2 \times \text{curr} + \text{node.val}$$
- **Leaf Aggregation:**
  - When reaching a leaf node (`not node.left and not node.right`), the binary representation is complete. We return `curr`.
  - For non-leaf nodes, the total sum contributed by paths passing through this node is simply the sum of the valid numbers formed in its left and right subtrees:
    $$\text{dfs}(\text{node}, \text{curr}) = \text{dfs}(\text{node.left}, \text{curr}) + \text{dfs}(\text{node.right}, \text{curr})$$
  - If a child pointer is `null`, it returns $0$.

---

### Solution Approach (Step-by-Step)

1. Define a helper `dfs(node, curr)`.
2. Base case: If `node is None`, return `0`.
3. Shift and append: Update `curr = (curr << 1) | node.val`.
4. If `node` is a leaf (i.e. `node.left is None and node.right is None`), return `curr`.
5. Recurse on left child: `dfs(node.left, curr)`.
6. Recurse on right child: `dfs(node.right, curr)`.
7. Return `left_sum + right_sum`.
8. Start recursion with `dfs(root, 0)`.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
         1
       /   \
      0     1
     / \   / \
    0   1 0   1

Paths & Decimals:
Root (curr = 1)
├── Left (curr = (1<<1)|0 = 2)
│   ├── Left (curr = (2<<1)|0 = 4)   -> Leaf! return 4  ("100"_2 = 4)
│   └── Right (curr = (2<<1)|1 = 5)  -> Leaf! return 5  ("101"_2 = 5)
│   Sum of Left Subtree = 4 + 5 = 9
│
└── Right (curr = (1<<1)|1 = 3)
    ├── Left (curr = (3<<1)|0 = 6)   -> Leaf! return 6  ("110"_2 = 6)
    └── Right (curr = (3<<1)|1 = 7)  -> Leaf! return 7  ("111"_2 = 7)
    Sum of Right Subtree = 6 + 7 = 13

Total Tree Sum = 9 + 13 = 22
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Balanced Tree
- **Input:** `root = [1,0,1,0,1,0,1]`
- **Step Trace:**
  | Node Val | Path From Root | Bit Representation | Decimal Value | Leaf? | Return Value |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 1 | `[1]` | `1` | 1 | No | Recurse left & right |
  | 0 | `[1, 0]` | `10` | 2 | No | Recurse left & right |
  | 0 | `[1, 0, 0]` | `100` | 4 | **Yes** | 4 |
  | 1 | `[1, 0, 1]` | `101` | 5 | **Yes** | 5 |
  | 1 | `[1, 1]` | `11` | 3 | No | Recurse left & right |
  | 0 | `[1, 1, 0]` | `110` | 6 | **Yes** | 6 |
  | 1 | `[1, 1, 1]` | `111` | 7 | **Yes** | 7 |
- **Output:** $4 + 5 + 6 + 7 = 22$

#### Example 2: Single Node Tree
- **Input:** `root = [0]`
- **Step Trace:**
  | Node Val | Path From Root | Bit Representation | Decimal Value | Leaf? | Return Value |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 0 | `[0]` | `0` | 0 | **Yes** | 0 |
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
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], curr: int) -> int:
            if not node:
                return 0
            
            # Left shift accumulated value and append current node's bit
            curr = (curr << 1) | node.val
            
            # If leaf node, return the computed decimal value
            if not node.left and not node.right:
                return curr
            
            return dfs(node.left, curr) + dfs(node.right, curr)
        
        return dfs(root, 0)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <iostream>

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
    int sumRootToLeaf(TreeNode* root) {
        return dfs(root, 0);
    }

private:
    int dfs(TreeNode* node, int curr) {
        if (!node) return 0;
        
        curr = (curr << 1) | node->val;
        
        if (!node->left && !node->right) {
            return curr;
        }
        
        return dfs(node->left, curr) + dfs(node->right, curr);
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
    public int sumRootToLeaf(TreeNode root) {
        return dfs(root, 0);
    }

    private int dfs(TreeNode node, int curr) {
        if (node == null) {
            return 0;
        }

        curr = (curr << 1) | node.val;

        // Leaf node reached
        if (node.left == null && node.right == null) {
            return curr;
        }

        return dfs(node.left, curr) + dfs(node.right, curr);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Every node in the binary tree is visited exactly once during DFS traversal.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion call stack requires space proportional to the height of the tree $H$. In the worst case of a skewed tree $H = \mathcal{O}(N)$; for a balanced tree $H = \mathcal{O}(\log N)$. Morris Traversal can reduce space to $\mathcal{O}(1)$.

---

### Takeaway Pattern & Interview Traps

1. **Leaf Checking Trap:** Do not add `curr` when `node is None`. A node with only one child will branch to a `None` child; if evaluated as a leaf, it would incorrectly add the partial path number twice! Always check `not node.left and not node.right`.
2. **Bit Shift Equivalences:** `(curr << 1) | node.val` is identical to `curr * 2 + node.val`. Using bitwise operations is both idiomatic and faster in hardware.
3. **Generalization:** This pattern directly solves LeetCode 129 (*Sum Root to Leaf Numbers*), where digits are base 10 (`curr * 10 + node.val`) instead of base 2.