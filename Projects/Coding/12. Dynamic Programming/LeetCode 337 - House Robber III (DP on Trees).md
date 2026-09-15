---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 337: House Robber III (DP on Trees)"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - tree-dp
  - binary-tree
  - postorder-traversal
  - amazon
  - google
  - meta
  - uber
---

# LeetCode 337: House Robber III (DP on Trees)

**Target Companies:** Amazon, Google, Meta, Microsoft, Uber, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming on Trees / Postorder Traversal / State Reduction  

---

### Problem Statement

The thief has found himself a new place for his thievery so that he will not get too much attention. There is only one entrance to this area, called `root`.

Besides the `root`, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if **two directly-linked houses were broken into on the same night**.

Given the `root` of the binary tree, return the **maximum amount of money** the thief can rob without alerting the police.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]` — Root of the binary tree.
- **Output:** `int` — Maximum money robbed without robbing directly-linked nodes.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 10^4]$.
  - $0 \le \text{Node.val} \le 10^4$.

---

### Key Idea & Intuition

1. **Why Greedy / Level-Order Fails:**
   - A common misconception is to rob alternating levels (e.g., all even levels vs. all odd levels).
   - Counterexample: A node can be skipped to allow robbing both its grandchildren, or we can rob the left grandchild and the right direct child. Choosing alternating levels restricts valid independent choices across disjoint subtrees.

2. **Optimal Substructure on Trees (Tree DP):**
   - Unlike linear arrays, trees branch. However, subtrees are mutually disjoint and can be solved independently from bottom to top (postorder traversal).
   - At each node $u$, there are strictly two mutually exclusive states:
     - State 0: **Do NOT rob node $u$** ($\text{not\_rob}(u)$).
     - State 1: **DO rob node $u$** ($\text{rob}(u)$).

3. **State Transitions:**
   - **Case 1: Rob node $u$:**
     - Because adjacent nodes cannot be robbed, we **must NOT rob** the left child $L$ and the right child $R$:
       $$\text{rob}(u) = u.\text{val} + \text{not\_rob}(L) + \text{not\_rob}(R)$$
   - **Case 2: Do NOT rob node $u$:**
     - Since node $u$ is spared, its children can either be robbed or not robbed independently. We greedily pick the maximum possible money from each child's subtree:
       $$\text{not\_rob}(u) = \max(\text{not\_rob}(L), \text{rob}(L)) + \max(\text{not\_rob}(R), \text{rob}(R))$$
   - **Base Case:**
     - If node is `null`: return `(0, 0)` (0 money whether robbed or not).
   - **Final Answer:**
     - At the root node: $\max(\text{not\_rob}(\text{root}), \text{rob}(\text{root}))$.

4. **Space Optimization via Return Tuples:**
   - Instead of using a memoization hash map (`Map<TreeNode, Integer>`), each recursive call can return a 2-element tuple/array `[not_rob, rob]`. This reduces auxiliary heap space to zero and runs in pure $O(H)$ recursion stack space.

---

### Solution Approach (Step-by-Step)

1. **Define Helper Function `dfs(node) -> [not_rob, rob]`:**
   - If `node == null`, return `[0, 0]`.
2. **Postorder Recurrence:**
   - Recursively solve for left subtree: `left = dfs(node.left)`.
   - Recursively solve for right subtree: `right = dfs(node.right)`.
3. **Compute Current Node States:**
   - `rob_val = node.val + left[0] + right[0]` (cannot rob children).
   - `not_rob_val = max(left[0], left[1]) + max(right[0], right[1])` (free to rob or skip children).
4. **Return & Aggregate:**
   - Return `[not_rob_val, rob_val]`.
   - At the root, return `max(dfs(root)[0], dfs(root)[1])`.

---

### Visual Algorithm Walkthrough

Consider the binary tree:
```
        3
       / \
      2   3
       \   \
        3   1
```

**Postorder Bottom-Up Trace:**

```
1. Null nodes:
   Returns [0, 0]

2. Leaf Node (3) [right child of 2]:
   rob = 3 + 0 + 0 = 3
   not_rob = max(0, 0) + max(0, 0) = 0
   Returns: [0, 3]

3. Leaf Node (1) [right child of 3]:
   rob = 1 + 0 + 0 = 1
   not_rob = max(0, 0) + max(0, 0) = 0
   Returns: [0, 1]

4. Internal Node (2):
   left = [0, 0], right = [0, 3]
   rob = 2 + left[0] + right[0] = 2 + 0 + 0 = 2
   not_rob = max(left) + max(right) = max(0, 0) + max(0, 3) = 0 + 3 = 3
   Returns: [3, 2]

5. Internal Node (3) [right child of root]:
   left = [0, 0], right = [0, 1]
   rob = 3 + left[0] + right[0] = 3 + 0 + 0 = 3
   not_rob = max(left) + max(right) = max(0, 0) + max(0, 1) = 0 + 1 = 1
   Returns: [1, 3]

6. Root Node (3):
   left = [3, 2], right = [1, 3]
   rob = 3 + left[0] + right[0] = 3 + 3 + 1 = 7
   not_rob = max(3, 2) + max(1, 3) = 3 + 3 = 6
   Returns: [6, 7]

Final Result = max(6, 7) = 7
Optimal Houses Robbed: Root (3) + Grandchild (3) + Grandchild (1) = 7
```

---

### Solved Examples with Multiple Inputs

| Case | Tree Structure (`root`) | Postorder Calculations | Result | Explanation |
|---|---|---|---|---|
| **Standard Tree** | `[3, 2, 3, null, 3, null, 1]` | Root `rob=7, not_rob=6` | `7` | Rob root (3) + leaves (3, 1) = 7 |
| **All Levels Taken Alternating** | `[3, 4, 5, 1, 3, null, 1]` | Root `rob=3+4+0=7`, `not_rob=max(1,3)+max(0,1)+...=9` | `9` | Rob children (4 + 5) = 9 > root + grandchildren (3+1+3+1=8) |
| **Single Node** | `[10]` | `rob = 10, not_rob = 0` | `10` | Only one house to rob |
| **Degenerate Line** | `[4, 1, null, 2, null, 3]` | Node 3: `[0,3]`, Node 2: `[3,2]`, Node 1: `[3,3]`, Node 4: `[3,7]` | `7` | Rob nodes 4 and 3 ($4 + 3 = 7$) |
| **All Zeroes** | `[0, 0, 0]` | Every state yields `0` | `0` | No money in any house |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional, Tuple

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        # dfs returns a tuple: (not_robbed_val, robbed_val)
        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, 0)
            
            left_not_rob, left_rob = dfs(node.left)
            right_not_rob, right_rob = dfs(node.right)
            
            # If we rob this node, we cannot rob its children
            rob_curr = node.val + left_not_rob + right_not_rob
            
            # If we do not rob this node, we can choose to rob or skip each child
            not_rob_curr = max(left_not_rob, left_rob) + max(right_not_rob, right_rob)
            
            return (not_rob_curr, rob_curr)
        
        not_rob, rob = dfs(root)
        return max(not_rob, rob)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <algorithm>
#include <utility>

// Definition for a binary tree node.
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
    int rob(TreeNode* root) {
        auto [not_rob, rob_val] = dfs(root);
        return std::max(not_rob, rob_val);
    }

private:
    // Returns pair<not_robbed_val, robbed_val>
    std::pair<int, int> dfs(TreeNode* node) {
        if (!node) {
            return {0, 0};
        }

        auto [left_not, left_rob] = dfs(node.left);
        auto [right_not, right_rob] = dfs(node.right);

        // If robbing current node, cannot rob children
        int rob_curr = node->val + left_not + right_not;

        // If skipping current node, pick best for each subtree
        int not_rob_curr = std::max(left_not, left_rob) + std::max(right_not, right_rob);

        return {not_rob_curr, rob_curr};
    }
};
```

#### 3. Java (Modern, Typed)
```java
// Definition for a binary tree node.
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
    public int rob(TreeNode root) {
        int[] result = dfs(root);
        return Math.max(result[0], result[1]);
    }

    // Returns int[] where index 0 = not_robbed, index 1 = robbed
    private int[] dfs(TreeNode node) {
        if (node == null) {
            return new int[]{0, 0};
        }

        int[] left = dfs(node.left);
        int[] right = dfs(node.right);

        int robCurr = node.val + left[0] + right[0];
        int notRobCurr = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);

        return new int[]{notRobCurr, robCurr};
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  Every node in the binary tree is visited exactly once during the postorder traversal. At each node, only $O(1)$ constant additions and maximum comparisons are performed.
- **Space Complexity:** $\mathcal{O}(H)$  
  Where $H$ is the height of the tree. The only memory consumed is the recursion call stack, which is $O(\log N)$ for a balanced tree and $O(N)$ in the worst-case degenerate skew tree. No extra dynamic allocation is used.

---

### Takeaway Pattern & Interview Traps

1. **Tree DP State Bundling:**
   - In standard arrays, DP tables indexed by $i$ store results. In trees, passing the two states `(not_rob, rob)` upwards via postorder return values is the cleanest, idiomatic design pattern for tree DP.
2. **The "Grandparent Memoization" Trap:**
   - Beginners often write: `rob(root) = max(root.val + rob(root.left.left) + rob(root.left.right) + ..., rob(root.left) + rob(root.right))`.
   - Without memoization, this recalculates overlapping subtrees with exponential complexity $\mathcal{O}(2^N)$. Even with memoization (`HashMap<TreeNode, Integer>`), it incurs heavy pointer hashing overhead. Returning the state tuple `(not_rob, rob)` solves the problem cleanly in a single postorder pass without hash tables.
3. **Skipping Multiple Levels:**
   - An optimal schedule may skip two consecutive levels if nodes at the third level have overwhelmingly high values. The transition `not_rob_curr = max(left[0], left[1]) + ...` handles arbitrary skipping naturally.