---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 129: Sum Root to Leaf Numbers"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - pre-order-traversal
  - amazon
  - google
---

# LeetCode 129: Sum Root to Leaf Numbers

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Tree DFS / Pre-order Traversal / Base-10 Digit Concatenation

---

### Problem Statement

You are given the `root` of a binary tree containing digits from `0` to `9` only.

Each root-to-leaf path in the tree represents a number.
- For example, the root-to-leaf path `1 -> 2 -> 3` represents the number `123`.

Return the **total sum** of all root-to-leaf numbers. Test cases are generated so that the answer will fit in a **32-bit** integer.

A **leaf** node is a node with no children.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — The sum of all root-to-leaf numbers.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 1000]$.
  - $0 \le \text{Node.val} \le 9$
  - The depth of the tree will not exceed $10$.

---

### Key Idea & Intuition

- **Base-10 Digit Accumulation:**
  - When descending from a parent node with accumulated value `curr` to a child with digit `node.val`, the new value is:
    $$\text{curr} = \text{curr} \times 10 + \text{node.val}$$
  - For example, passing through digits $1 \to 2 \to 3$:
    - Root $1$: $0 \times 10 + 1 = 1$
    - Child $2$: $1 \times 10 + 2 = 12$
    - Child $3$: $12 \times 10 + 3 = 123$
- **Leaf Terminal Sum:**
  - When a node has neither a left child nor a right child (`not node.left and not node.right`), it is a **leaf**. Return `curr`.
  - Otherwise, return the sum of the valid numbers formed by its left and right subtrees:
    $$\text{dfs}(\text{node}, \text{curr}) = \text{dfs}(\text{node.left}, \text{curr}) + \text{dfs}(\text{node.right}, \text{curr})$$

---

### Solution Approach (Step-by-Step)

1. Define recursive helper `dfs(node, curr)`.
2. Base case: If `node is None`, return `0`.
3. Multiply current by 10 and add `node.val`: `curr = curr * 10 + node.val`.
4. Check if `node` is a leaf:
   - If `node.left is None and node.right is None`: return `curr`.
5. Recurse down both branches: return `dfs(node.left, curr) + dfs(node.right, curr)`.
6. Invoke `dfs(root, 0)` from the main function.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
         4
       /   \
      9     0
     / \
    5   1

Path Exploration:
Root (4) -> curr = 4
├── Left Child (9) -> curr = 4 * 10 + 9 = 49
│   ├── Left Child (5) -> curr = 49 * 10 + 5 = 495 (Leaf -> returns 495)
│   └── Right Child (1) -> curr = 49 * 10 + 1 = 491 (Leaf -> returns 491)
│   Sum from Node 9 = 495 + 491 = 986
│
└── Right Child (0) -> curr = 4 * 10 + 0 = 40 (Leaf -> returns 40)

Total Tree Sum = 986 + 40 = 1026
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Tree
- **Input:** `root = [4,9,0,5,1]`
- **Step Trace:**
  | Node Val | Path From Root | `curr` | Leaf? | Return Value |
  | :--- | :--- | :--- | :--- | :--- |
  | 4 | `[4]` | 4 | No | `986 + 40 = 1026` |
  | 9 | `[4, 9]` | 49 | No | `495 + 491 = 986` |
  | 5 | `[4, 9, 5]` | 495 | **Yes** | 495 |
  | 1 | `[4, 9, 1]` | 491 | **Yes** | 491 |
  | 0 | `[4, 0]` | 40 | **Yes** | 40 |
- **Output:** `1026`

#### Example 2: Linear Chain
- **Input:** `root = [1,2,null,3]`
- **Step Trace:**
  - Root `1`: `curr = 1`
  - Left child `2`: `curr = 12` (has right child, so not a leaf!)
  - Right child `3`: `curr = 123` (is a leaf $\implies 123$)
- **Output:** `123`

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
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], curr: int) -> int:
            if not node:
                return 0
            
            curr = curr * 10 + node.val
            
            # Leaf node reached: return completed path number
            if not node.left and not node.right:
                return curr
            
            return dfs(node.left, curr) + dfs(node.right, curr)
            
        return dfs(root, 0)
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
    int sumNumbers(TreeNode* root) {
        return dfs(root, 0);
    }

private:
    int dfs(TreeNode* node, int curr) {
        if (!node) return 0;

        curr = curr * 10 + node->val;

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
    public int sumNumbers(TreeNode root) {
        return dfs(root, 0);
    }

    private int dfs(TreeNode node, int curr) {
        if (node == null) {
            return 0;
        }

        curr = curr * 10 + node.val;

        // Leaf condition
        if (node.left == null && node.right == null) {
            return curr;
        }

        return dfs(node.left, curr) + dfs(node.right, curr);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Every node in the binary tree is visited exactly once.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion call stack space proportional to the tree height $H$. In balanced trees $H = \mathcal{O}(\log N)$, in worst-case skewed trees $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Premature Summing on Single-Child Nodes:** A node with only one child is NOT a leaf. If you return `curr` when `node == null`, the partial number of a single-child parent will be added, which is incorrect. Only sum when `node.left == null && node.right == null`.
2. **Integer Overflow:** The problem constraints state tree depth $\le 10$, so the maximum path number formed is $\le 9,999,999,999$. While $N \le 1000$ could theoretically produce large numbers if depth were 1000, here max depth $\le 10$ guarantees all intermediate and final values fit in a 32-bit signed integer.
3. **Equivalence with LeetCode 1022:** This is the identical paradigm to LeetCode 1022 (*Sum of Root To Leaf Binary Numbers*), replacing base 2 arithmetic (`(curr << 1) | bit`) with base 10 arithmetic (`curr * 10 + digit`).