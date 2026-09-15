---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 124: Binary Tree Maximum Path Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - tree
  - dfs
  - meta
  - google
  - amazon
  - microsoft
---

# LeetCode 124: Binary Tree Maximum Path Sum

**Target Companies:** Meta (Signature Hard #1), Google, Amazon, Microsoft, ByteDance, Bloomberg  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Tree / Depth-First Search  

---

### Problem Statement

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return *the **maximum path sum** of any **non-empty** path*.

---

### Input & Output Formats & Constraints

- **Input:** The `root` of a binary tree (`1 <= number of nodes <= 3 * 10^4`).
- **Output:** An integer representing the maximum path sum.
- **Constraints:**
  - The number of nodes in the tree is in the range `[1, 3 * 10^4]`.
  - `-1000 <= Node.val <= 1000`

---

### Key Idea & Intuition

#### Tree DP (Postorder Traversal)
A path in a binary tree has a unique **highest node** (the lowest common ancestor of all nodes in the path, or the "turn point").
For every node $u$ acting as this turning point:
- The path enters from one of $u$'s subtrees, passes through $u$, and optionally continues down into $u$'s other subtree.
- Therefore, to compute the maximum path sum turning at $u$, we need to know the **maximum branch gain** each child can contribute upward.

#### The Dual-Role Invariant
At every node $u$, we must maintain two distinct concepts:
1. **Branch Gain (`max_gain(u)`):**
   The maximum sum of a path starting at node $u$ and extending downward into **at most one** of its children. This is the value node $u$ returns to its parent:
   $$\text{gain}(u) = u.val + \max(0, \text{gain}(u.left), \text{gain}(u.right))$$
   *(Note: If a child's branch gain is negative, we greedily prune it to 0)*.
2. **Arch Path Turning at $u$:**
   The complete path that uses $u$ as the highest apex, combining both left and right branches:
   $$\text{arch\_sum}(u) = u.val + \max(0, \text{gain}(u.left)) + \max(0, \text{gain}(u.right))$$
   We update our global answer with $\text{arch\_sum}(u)$.

---

### Solution Approach (Step-by-Step)

1. **Global Maximum Tracker:**
   - Initialize `max_sum = -infinity` (to handle trees where all node values are negative).
2. **Recursive DFS Helper (`max_gain(node)`):**
   - Base Case: If `node is None`, return `0`.
   - Postorder Recursion:
     - Compute left branch gain: `left_gain = max(0, max_gain(node.left))`.
     - Compute right branch gain: `right_gain = max(0, max_gain(node.right))`.
   - Update Global Result:
     - The path turning at `node` has sum `node.val + left_gain + right_gain`.
     - `max_sum = max(max_sum, node.val + left_gain + right_gain)`.
   - Return Value to Parent:
     - Return `node.val + max(left_gain, right_gain)`.
3. **Execute and Return:**
   - Call `max_gain(root)`.
   - Return `max_sum`.

---

### Visual Algorithm Walkthrough

#### Trace for Tree: `[-10, 9, 20, null, null, 15, 7]`
```
          -10
         /   \
        9     20
             /  \
            15   7

Postorder Traversal:
1. Visit Node 9:
   - Left child = None -> 0
   - Right child = None -> 0
   - Arch sum at 9: 9 + 0 + 0 = 9 -> max_sum = max(-inf, 9) = 9
   - Returns to parent (-10): 9 + max(0, 0) = 9

2. Visit Node 15:
   - Left = 0, Right = 0
   - Arch sum at 15: 15 + 0 + 0 = 15 -> max_sum = max(9, 15) = 15
   - Returns to parent (20): 15 + 0 = 15

3. Visit Node 7:
   - Left = 0, Right = 0
   - Arch sum at 7: 7 + 0 + 0 = 7 -> max_sum = 15
   - Returns to parent (20): 7 + 0 = 7

4. Visit Node 20:
   - Left gain from 15: max(0, 15) = 15
   - Right gain from 7: max(0, 7) = 7
   - Arch sum at 20: 20 + 15 + 7 = 42 -> max_sum = max(15, 42) = 42!
   - Returns to parent (-10): 20 + max(15, 7) = 20 + 15 = 35

5. Visit Root (-10):
   - Left gain from 9: max(0, 9) = 9
   - Right gain from 20: max(0, 35) = 35
   - Arch sum at -10: -10 + 9 + 35 = 34 -> max_sum = max(42, 34) = 42
   - Returns: -10 + 35 = 25

Global Max Path Sum = 42 (Path: 15 -> 20 -> 7).
```

---

### Solved Examples with Multiple Inputs

| Tree Structure | Postorder Gains Computed | Arch Sums Evaluated | Global Max Path Sum |
|---|---|---|---|
| `[1, 2, 3]` | Node 2: 2, Node 3: 3, Node 1: $1 + 2 + 3 = 6$ | $2, 3, 6$ | `6` ($2 \to 1 \to 3$) |
| `[-10, 9, 20, null, null, 15, 7]` | Shown in walkthrough | $9, 15, 7, 42, 34$ | `42` ($15 \to 20 \to 7$) |
| `[-3]` | Single negative node | Arch sum: $-3$ | `-3` |
| `[2, -1]` | Node -1 returns 0 (pruned); Node 2 returns 2 | Node 2 arch: $2 + 0 = 2$ | `2` (single node 2) |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        max_sum: float = float('-inf')
        
        def max_gain(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0
                
            # Recursively compute max gains from subtrees (ignore negatives)
            left_gain = max(0, max_gain(node.left))
            right_gain = max(0, max_gain(node.right))
            
            # Price of the new path turning at current node
            current_path = node.val + left_gain + right_gain
            max_sum = max(max_sum, current_path)
            
            # Return max branch extendable to parent
            return node.val + max(left_gain, right_gain)
            
        max_gain(root)
        return int(max_sum)
```

#### C++17
```cpp
#include <algorithm>
#include <climits>

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
private:
    int max_sum = INT_MIN;

    int maxGain(TreeNode* node) {
        if (!node) return 0;

        // Ignore branches that yield negative contributions
        int left_gain = std::max(0, maxGain(node->left));
        int right_gain = std::max(0, maxGain(node->right));

        // Evaluate path where current node is the highest turning point
        int current_arch = node->val + left_gain + right_gain;
        max_sum = std::max(max_sum, current_arch);

        // Return the single branch that extends upward to parent
        return node->val + std::max(left_gain, right_gain);
    }

public:
    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return max_sum;
    }
};
```

#### Java 17
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    private int maxSum = Integer.MIN_VALUE;

    private int maxGain(TreeNode node) {
        if (node == null) {
            return 0;
        }

        // Postorder: compute max positive branch gains
        int leftGain = Math.max(0, maxGain(node.left));
        int rightGain = Math.max(0, maxGain(node.right));

        // Path with current node as the peak
        int currentArch = node.val + leftGain + rightGain;
        maxSum = Math.max(maxSum, currentArch);

        // Return single branch to parent
        return node.val + Math.max(leftGain, rightGain);
    }

    public int maxPathSum(TreeNode root) {
        maxGain(root);
        return maxSum;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N$ is the number of nodes in the tree. Each node is visited exactly once in the postorder DFS traversal, performing constant-time arithmetic.
- **Space Complexity:** $\mathcal{O}(H)$, where $H$ is the height of the binary tree, consumed by the recursion call stack. In the worst case (skewed tree), $H = \mathcal{O}(N)$; for a balanced tree, $H = \mathcal{O}(\log N)$.

---

### Takeaway Pattern & Interview Traps

1. **Why Can't We Return `arch_sum` Upward?** A path cannot branch twice: you cannot enter a node from a parent and visit both its left and right children. The value returned to the parent must choose *at most one* branch: $node.val + \max(left\_gain, right\_gain)$.
2. **Negative Node Initialization:** Initializing `max_sum = 0` causes incorrect answers when all tree nodes are negative (e.g. `root = [-3]`). Always initialize `max_sum = -infinity` (or `Integer.MIN_VALUE`).
3. **Pruning Negative Branches:** The `max(0, gain)` clamp ensures we discard subtrees with net-negative sums, effectively allowing paths of length 1.