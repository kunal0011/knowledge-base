---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 124: Binary Tree Maximum Path Sum"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 124: Binary Tree Maximum Path Sum

**Target Companies:** Meta (All-Time Top #1 Tree Hard), Google, Amazon, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** Tree Post-order DFS / Subtree Contribution / Global Inverted-V Path Optimization

---

### Problem Statement

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does **not** need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return the **maximum path sum** of any non-empty path.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — Maximum path sum across all possible paths.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 3 \times 10^4]$.
  - $-1000 \le \text{Node.val} \le 1000$

---

### Key Idea & Intuition

- **Split Paths vs. Upward Paths:**
  - At any node $u$, a path can:
    1. **Turn at $u$ (Inverted-V path):** Pass through $u$'s left child, through $u$, and down $u$'s right child:
       $$\text{local\_turn\_sum} = u.\text{val} + \text{left\_gain} + \text{right\_gain}$$
       This path **cannot** be extended upward to $u$'s parent because that would branch and violate the path definition (a path cannot visit a node twice).
    2. **Extend upward to $u$'s parent:** A valid path continuing upward can only take **one** of $u$'s branches (either left or right, whichever is larger):
       $$\text{return\_gain} = u.\text{val} + \max(0, \max(\text{left\_gain}, \text{right\_gain}))$$
- **Pruning Negative Branches:**
  - If a subtree's maximum gain is negative ($< 0$), including it will only decrease the sum. Thus, we clamp negative subtree contributions to $0$:
    $$\text{left\_gain} = \max(0, \text{dfs}(u.\text{left}))$$
    $$\text{right\_gain} = \max(0, \text{dfs}(u.\text{right}))$$
- **Global Tracker:**
  - Since the optimal path could be contained entirely within any subtree, we maintain a global variable `max_path_sum` initialized to $-\infty$. At every node $u$, we update:
    $$\text{max\_path\_sum} = \max(\text{max\_path\_sum}, u.\text{val} + \text{left\_gain} + \text{right\_gain})$$

---

### Solution Approach (Step-by-Step)

1. Initialize `max_path_sum = -infinity` (crucial for trees with all negative values).
2. Define recursive helper `max_gain(node)`:
   - Base case: If `node is None`, return `0`.
   - Compute left subtree gain: `left_gain = max(0, max_gain(node.left))`.
   - Compute right subtree gain: `right_gain = max(0, max_gain(node.right))`.
   - Calculate turn path sum at this node: `current_sum = node.val + left_gain + right_gain`.
   - Update `max_path_sum = max(max_path_sum, current_sum)`.
   - Return to parent the best single branch upward: `node.val + max(left_gain, right_gain)`.
3. Call `max_gain(root)` and return `max_path_sum`.

---

### Visual Algorithm Walkthrough

```
Tree:
         -10
         /  \
        9    20
            /  \
           15   7

1. Node 15 (Leaf):
   - left_gain = 0, right_gain = 0
   - local_sum = 15 + 0 + 0 = 15  -> max_path_sum = 15
   - returns: 15 + max(0, 0) = 15

2. Node 7 (Leaf):
   - left_gain = 0, right_gain = 0
   - local_sum = 7 + 0 + 0 = 7
   - returns: 7 + max(0, 0) = 7

3. Node 20:
   - left_gain = max(0, 15) = 15
   - right_gain = max(0, 7) = 7
   - local_sum = 20 + 15 + 7 = 42 -> max_path_sum = max(15, 42) = 42!  [Path: 15 -> 20 -> 7]
   - returns: 20 + max(15, 7) = 20 + 15 = 35

4. Node 9 (Leaf):
   - local_sum = 9 -> max_path_sum = 42
   - returns: 9

5. Node -10 (Root):
   - left_gain = max(0, 9) = 9
   - right_gain = max(0, 35) = 35
   - local_sum = -10 + 9 + 35 = 34
   - max_path_sum remains 42!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Tree with Inverted-V Optimal Path
- **Input:** `root = [-10, 9, 20, null, null, 15, 7]`
- **Step Trace:**
  | Node | `left_gain` | `right_gain` | `local_sum` ($val + L + R$) | `max_path_sum` | Upward Return ($val + \max(L, R)$) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 9 | 0 | 0 | $9 + 0 + 0 = 9$ | 9 | 9 |
  | 15 | 0 | 0 | $15 + 0 + 0 = 15$ | 15 | 15 |
  | 7 | 0 | 0 | $7 + 0 + 0 = 7$ | 15 | 7 |
  | 20 | 15 | 7 | $20 + 15 + 7 = 42$ | **42** | $20 + 15 = 35$ |
  | -10 | 9 | 35 | $-10 + 9 + 35 = 34$ | **42** | $-10 + 35 = 25$ |
- **Output:** `42` (Path: `15 -> 20 -> 7`)

#### Example 2: All Negative Nodes Tree
- **Input:** `root = [-3]`
- **Step Trace:**
  - `node = -3`, `left_gain = 0`, `right_gain = 0`.
  - `local_sum = -3 + 0 + 0 = -3`.
  - `max_path_sum = max(-inf, -3) = -3`.
- **Output:** `-3`

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
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        max_sum = float('-inf')
        
        def max_gain(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0
            
            # Clamp negative subtree returns to 0
            left_gain = max(0, max_gain(node.left))
            right_gain = max(0, max_gain(node.right))
            
            # Path turning at current node
            current_path = node.val + left_gain + right_gain
            max_sum = max(max_sum, current_path)
            
            # Return maximum gain if continuing path to parent
            return node.val + max(left_gain, right_gain)
        
        max_gain(root)
        return int(max_sum)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <algorithm>
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
    int maxPathSum(TreeNode* root) {
        int maxSum = INT_MIN;
        maxGain(root, maxSum);
        return maxSum;
    }

private:
    int maxGain(TreeNode* node, int& maxSum) {
        if (!node) return 0;

        int leftGain = std::max(0, maxGain(node->left, maxSum));
        int rightGain = std::max(0, maxGain(node->right, maxSum));

        // Evaluate path turning at current node
        int currentPath = node->val + leftGain + rightGain;
        maxSum = std::max(maxSum, currentPath);

        // Return single branch gain upward to parent
        return node->val + std::max(leftGain, rightGain);
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
    private int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxSum = Integer.MIN_VALUE;
        maxGain(root);
        return maxSum;
    }

    private int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }

        // Only include positive contributions from children
        int leftGain = Math.max(0, maxGain(node.left));
        int rightGain = Math.max(0, maxGain(node.right));

        // Path with current node as the highest ancestor
        int currentPath = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, currentPath);

        // Return max one-way branch to parent
        return node.val + Math.max(leftGain, rightGain);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Each node in the tree is visited exactly once in post-order traversal. All calculations at each node take $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(H)$ — Where $H$ is the tree height corresponding to the maximum depth of the recursion call stack. For a balanced tree $H = \mathcal{O}(\log N)$, and worst-case degenerate tree $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **The "All-Negative Values" Trap:** Never initialize `max_sum = 0`. If all nodes in the tree are negative (e.g. `root = [-3]`), initializing to `0` would incorrectly return `0`, when the answer must be `-3`. Always initialize to $-\infty$ (`Integer.MIN_VALUE` / `INT_MIN`).
2. **Upward Return vs. Local Path Distinction:** In tree DP problems (such as LeetCode 543 *Diameter of Binary Tree* and LeetCode 687 *Longest Univalue Path*), always distinguish between:
   - What is **returned** to the parent (must be a valid unbranched path segment).
   - What **updates** the global optimum (can combine both left and right branches).
3. **Clamping to Zero:** `max(0, gain)` is required because an empty subtree choice contributes $0$, avoiding negative penalties.