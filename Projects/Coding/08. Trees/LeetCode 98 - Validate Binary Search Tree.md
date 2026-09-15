---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 98: Validate Binary Search Tree"
tags:
  - leetcode
  - coding
  - trees
  - bst
  - dfs
  - amazon
  - google
---

# LeetCode 98: Validate Binary Search Tree

**Target Companies:** Amazon (Top Tier Tree Question), Meta, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Binary Search Tree / Interval Range Validation / In-order Monotonicity

---

### Problem Statement

Given the `root` of a binary tree, determine if it is a valid binary search tree (BST).

A **valid BST** is defined as follows:
- The left subtree of a node contains only nodes with keys **strictly less than** the node's key.
- The right subtree of a node contains only nodes with keys **strictly greater than** the node's key.
- Both the left and right subtrees must also be binary search trees.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `bool` — `true` if the tree is a valid BST, `false` otherwise.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 10^4]$.
  - $-2^{31} \le \text{Node.val} \le 2^{31} - 1$

---

### Key Idea & Intuition

- **The Local Parent-Child Trap:**
  - A classic beginner bug is checking only immediate child relationships:
    `node.left.val < node.val` and `node.right.val > node.val`.
  - This fails for trees like:
    ```
        5
       / \
      1   6
         / \
        3   7
    ```
    Here $3 < 6$ and $7 > 6$ locally, but $3$ is in $5$'s right subtree and $3 < 5$, violating global BST ordering!
- **The Open Interval Invariant:**
  - BST validity is a **global ancestor constraint**. Every node $u$ must satisfy:
    $$\text{low} < u.\text{val} < \text{high}$$
  - When branching left: all descendants must be strictly smaller than $u.\text{val}$:
    $$\text{new\_range} = (\text{low}, u.\text{val})$$
  - When branching right: all descendants must be strictly greater than $u.\text{val}$:
    $$\text{new\_range} = (u.\text{val}, \text{high})$$
  - Root begins with bounds $(-\infty, +\infty)$.
- **Integer Boundary Guard:**
  - Constraints state node values can reach $-2^{31}$ or $2^{31} - 1$.
  - Using 32-bit `INT_MIN` / `INT_MAX` for initial bounds will cause false negatives if root equals `INT_MAX` or `INT_MIN`.
  - In C++ and Java, initial bounds must use 64-bit `long` or nullable wrapper types (`Long` / `TreeNode*`).

---

### Solution Approach (Step-by-Step)

1. Define recursive helper `validate(node, low, high)`:
   - Base case: If `node is None`, return `True`.
   - If `node.val <= low` or `node.val >= high`: return `False` (violates strict inequality).
   - Recurse left: `validate(node.left, low, node.val)`.
   - Recurse right: `validate(node.right, node.val, high)`.
   - Return `left_valid and right_valid`.
2. Initial call: `validate(root, -infinity, +infinity)`.

---

### Visual Algorithm Walkthrough

```
Example: Invalid BST
        5 (low = -inf, high = +inf)
       / \
      1   4 (low = 5, high = +inf)  <- Invalid! 4 is not > 5
         / \
        3   6

DFS Trace:
1. Root 5: Valid (-inf < 5 < +inf)
   - Branch Left: low = -inf, high = 5
   - Branch Right: low = 5, high = +inf

2. Node 1: Valid (-inf < 1 < 5)
   - Left and Right are null -> True

3. Node 4:
   - Check: is 5 < 4 < +inf?
   - 4 <= 5 (low bound violated!)
   - Returns False immediately.

Tree is NOT a valid BST.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Valid BST
- **Input:** `root = [2, 1, 3]`
- **Step Trace:**
  | Node | `low` | `high` | Condition $\text{low} < val < \text{high}$ | Result |
  | :--- | :--- | :--- | :--- | :--- |
  | 2 | $-\infty$ | $+\infty$ | $-\infty < 2 < +\infty$ | Valid |
  | 1 | $-\infty$ | 2 | $-\infty < 1 < 2$ | Valid (Leaf) |
  | 3 | 2 | $+\infty$ | $2 < 3 < +\infty$ | Valid (Leaf) |
- **Output:** `true`

#### Example 2: Duplicate Node Value
- **Input:** `root = [2, 2, 2]`
- **Trace:** Node 2 left child is 2. Range for left is $(-\infty, 2)$. $2 \nless 2 \implies$ Invalid!
- **Output:** `false` (Strict inequalities required)

#### Example 3: Extreme Bounds (32-bit Integer Limit)
- **Input:** `root = [-2147483648]`
- **Trace:** With 64-bit bounds $(-\infty, +\infty)$, $-2^{63} < -2147483648 < 2^{63}-1$.
- **Output:** `true`

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
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
                
            # Node value must be strictly within (low, high)
            if not (low < node.val < high):
                return False
                
            # Left subtree upper-bounded by node.val
            # Right subtree lower-bounded by node.val
            return (validate(node.left, low, node.val) and 
                    validate(node.right, node.val, high))
                    
        return validate(root, float('-inf'), float('inf'))
```

#### 2. C++ (C++17 / STL)
```cpp
#include <climits>

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
    bool isValidBST(TreeNode* root) {
        // Use 64-bit integers to prevent overflow when node values equal INT_MIN or INT_MAX
        return validate(root, LONG_MIN, LONG_MAX);
    }

private:
    bool validate(TreeNode* node, long long low, long long high) {
        if (!node) return true;

        if (node->val <= low || node->val >= high) {
            return false;
        }

        return validate(node->left, low, node->val) && 
               validate(node->right, node->val, high);
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
    public boolean isValidBST(TreeNode root) {
        // Pass Long.MIN_VALUE and Long.MAX_VALUE to avoid integer overflow
        return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean validate(TreeNode node, long low, long high) {
        if (node == null) {
            return true;
        }

        if (node.val <= low || node.val >= high) {
            return false;
        }

        return validate(node.left, low, node.val) && 
               validate(node.right, node.val, high);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Each node in the tree is visited at most once during traversal. If an invalid node is encountered, DFS terminates early.
- **Space Complexity:** $\mathcal{O}(H)$ — Where $H$ is the tree height corresponding to recursion call stack memory. In balanced trees $H = \mathcal{O}(\log N)$, in worst-case degenerate trees $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **No Duplicates Allowed:** In standard LeetCode BST definitions, all values in the left subtree must be *strictly* less ($<$), and all values in the right subtree *strictly* greater ($>$). If `node.val == low` or `node.val == high`, it is invalid.
2. **64-bit Integer Bounds:** If you initialize `low = Integer.MIN_VALUE` and `high = Integer.MAX_VALUE` in Java/C++, a root whose value is `Integer.MIN_VALUE` will immediately fail the check `node.val <= low`. Always use 64-bit `long` or `null` checks.
3. **In-order Monotonicity Alternative:** An in-order traversal of a valid BST must produce a strictly monotonically increasing sequence. Checking that `prev.val < curr.val` during an in-order traversal is an equally valid alternative solution.