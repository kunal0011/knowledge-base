---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 543: Diameter of Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - post-order
  - amazon
  - google
---

# LeetCode 543: Diameter of Binary Tree

**Target Companies:** Meta (All-Time Top #1 Tree Essential), Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Tree Post-order DFS / Bottom-Up Subtree Depth / Longest Path Calculation

---

### Problem Statement

Given the `root` of a binary tree, return the length of the **diameter** of the tree.

The **diameter** of a binary tree is the **length of the longest path between any two nodes** in a tree. This path may or may not pass through the `root`.

The **length** of a path between two nodes is represented by the **number of edges** between them.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — Length of the longest path (in number of edges).
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 10^4]$.
  - $-100 \le \text{Node.val} \le 100$

---

### Key Idea & Intuition

- **Diameter Through a Local Peak:**
  - Every valid path between two nodes has a single **highest ancestor** (the apex/peak of the path).
  - If a path's apex is node $u$, the path extends down $u$'s left branch and down $u$'s right branch.
  - The maximum length of such a path through $u$ is simply:
    $$\text{diameter}(u) = \text{depth}(u.\text{left}) + \text{depth}(u.\text{right})$$
    where $\text{depth}(\text{node})$ is the maximum number of edges from `node` down to a leaf.
- **Post-Order Bottom-Up Traversal:**
  - If we recursively compute the depth of each subtree bottom-up:
    - Base case: `depth(None) = 0`.
    - Recursively get `left_depth = depth(node.left)` and `right_depth = depth(node.right)`.
    - Update global diameter: `max_diameter = max(max_diameter, left_depth + right_depth)`.
    - Return upward to parent the depth of this node: `1 + max(left_depth, right_depth)`.
  - This visits every node once in $\mathcal{O}(N)$ time, avoiding duplicate work.

---

### Solution Approach (Step-by-Step)

1. Maintain global variable `max_diameter = 0`.
2. Define recursive helper `depth(node)`:
   - If `node is None`: return `0`.
   - `left_depth = depth(node.left)`
   - `right_depth = depth(node.right)`
   - Update `max_diameter = max(max_diameter, left_depth + right_depth)`.
   - Return `1 + max(left_depth, right_depth)`.
3. Call `depth(root)` and return `max_diameter`.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
          1
         / \
        2   3
       / \
      4   5

1. Node 4: Leaf -> left = 0, right = 0.
   - local diameter = 0 + 0 = 0.
   - returns: 1 + max(0,0) = 1.

2. Node 5: Leaf -> left = 0, right = 0.
   - local diameter = 0 + 0 = 0.
   - returns: 1 + max(0,0) = 1.

3. Node 2:
   - left_depth = 1, right_depth = 1.
   - local diameter = 1 + 1 = 2 (Path: 4 -> 2 -> 5).
   - returns: 1 + max(1,1) = 2.

4. Node 3: Leaf -> returns 1.

5. Node 1 (Root):
   - left_depth = 2 (via node 2), right_depth = 1 (via node 3).
   - local diameter = 2 + 1 = 3 (Path: 4/5 -> 2 -> 1 -> 3).
   - returns: 1 + max(2,1) = 3.

Maximum diameter recorded is 3 (3 edges: 4 -> 2 -> 1 -> 3).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Path Passes Through Root
- **Input:** `root = [1,2,3,4,5]`
- **Step Trace:**
  | Node | `left_depth` | `right_depth` | `local_diameter` ($L + R$) | `max_diameter` | Return to Parent ($1 + \max(L,R)$) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 4 | 0 | 0 | 0 | 0 | 1 |
  | 5 | 0 | 0 | 0 | 0 | 1 |
  | 2 | 1 | 1 | 2 | 2 | 2 |
  | 3 | 0 | 0 | 0 | 2 | 1 |
  | 1 | 2 | 1 | 3 | **3** | 3 |
- **Output:** `3`

#### Example 2: Path Lies Entirely Within a Subtree (Does Not Pass Through Root)
- **Input:**
  ```
          1
         /
        2
       / \
      3   4
     /     \
    5       6
  ```
- **Analysis:**
  - Left subtree of 2 has depth 2 (path 2 -> 3 -> 5).
  - Right subtree of 2 has depth 2 (path 2 -> 4 -> 6).
  - At Node 2: `diameter = 2 + 2 = 4` (Path: `5 -> 3 -> 2 -> 4 -> 6`).
  - At Root 1: `left_depth = 3`, `right_depth = 0` $\implies$ `diameter = 3 + 0 = 3`.
  - The maximum diameter is 4, which does **not** pass through root 1!
- **Output:** `4`

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
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_diameter = 0
        
        def depth(node: Optional[TreeNode]) -> int:
            nonlocal max_diameter
            if not node:
                return 0
            
            left_d = depth(node.left)
            right_d = depth(node.right)
            
            # Diameter at this node is the sum of depths of left and right subtrees
            max_diameter = max(max_diameter, left_d + right_d)
            
            # Return height of this subtree to parent
            return 1 + max(left_d, right_d)
            
        depth(root)
        return max_diameter
```

#### 2. C++ (C++17 / STL)
```cpp
#include <algorithm>

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
    int diameterOfBinaryTree(TreeNode* root) {
        int maxDiameter = 0;
        getDepth(root, maxDiameter);
        return maxDiameter;
    }

private:
    int getDepth(TreeNode* node, int& maxDiameter) {
        if (!node) return 0;

        int leftD = getDepth(node->left, maxDiameter);
        int rightD = getDepth(node->right, maxDiameter);

        maxDiameter = std::max(maxDiameter, leftD + rightD);

        return 1 + std::max(leftD, rightD);
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
    private int maxDiameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        maxDiameter = 0;
        getDepth(root);
        return maxDiameter;
    }

    private int getDepth(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftD = getDepth(node.left);
        int rightD = getDepth(node.right);

        maxDiameter = Math.max(maxDiameter, leftD + rightD);

        return 1 + Math.max(leftD, rightD);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Each node in the tree is visited exactly once during the post-order depth traversal.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion stack depth proportional to tree height $H$. For balanced trees $H = \mathcal{O}(\log N)$; for skewed trees $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Edges vs Nodes Count:**
   - The diameter is defined as the number of **edges**, which equals $\text{number of nodes} - 1$.
   - Using `left_depth + right_depth` directly yields the edge count without needing to subtract 1.
2. **Path Does Not Have to Include Root:** Example 2 highlights the most frequent interview trap: assuming the longest path must pass through the tree root. Always maintain a global maximum rather than simply returning `depth(root.left) + depth(root.right)`.
3. **Core Pattern Blueprint:** This exact post-order split-and-return template solves LeetCode 124 (*Binary Tree Maximum Path Sum*), LeetCode 687 (*Longest Univalue Path*), and LeetCode 1522 (*Diameter of N-Ary Tree*).