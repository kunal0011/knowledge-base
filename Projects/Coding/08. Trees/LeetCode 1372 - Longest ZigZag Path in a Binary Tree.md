---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 1372: Longest ZigZag Path in a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - dfs
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 1372: Longest ZigZag Path in a Binary Tree

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Tree DFS / Dynamic Programming on Trees / State Transition

---

### Problem Statement

You are given the `root` of a binary tree.

A **ZigZag path** for a binary tree is defined as follow:
- Choose any node in the binary tree and a direction (right or left).
- If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
- Change the direction from right to left or from left to right.
- Repeat the second and third steps until you cannot move in the tree.

ZigZag length is defined as the **number of nodes visited - 1** (i.e., the **number of edges** in the path).

Return the **longest ZigZag path** contained in the tree.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `int` — Maximum number of edges in a valid ZigZag path.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 5 \times 10^4]$.
  - $1 \le \text{Node.val} \le 100$

---

### Key Idea & Intuition

- **State Transition at Each Node:**
  - When moving to a child node, we have two choices:
    1. **Continue the ZigZag:** If we arrive at the current node from the opposite direction, the ZigZag chain extends: length becomes `prev_len + 1`.
    2. **Restart a new ZigZag:** If we move in the same direction as the previous edge, the ZigZag sequence breaks. However, this single step can serve as the first edge of a *new* ZigZag path: length resets to `1`.
- **Top-Down DFS Representation:**
  - We can define `dfs(node, is_left, length)`:
    - If `is_left == True` (we just moved left into `node`):
      - We can continue the ZigZag by moving right: `dfs(node.right, False, length + 1)`.
      - Or we can reset and move left: `dfs(node.left, True, 1)`.
    - If `is_left == False` (we just moved right into `node`):
      - We can continue the ZigZag by moving left: `dfs(node.left, True, length + 1)`.
      - Or we can reset and move right: `dfs(node.right, False, 1)`.
  - Maintain a global `max_len` updated with `length` at every visited node.

---

### Solution Approach (Step-by-Step)

1. Maintain global variable `max_zigzag = 0`.
2. Base case: If `node is None`, return.
3. Update `max_zigzag = max(max_zigzag, length)`.
4. If `is_left` is `True`:
   - To continue alternating: call `dfs(node.right, False, length + 1)`.
   - To reset direction: call `dfs(node.left, True, 1)`.
5. If `is_left` is `False`:
   - To continue alternating: call `dfs(node.left, True, length + 1)`.
   - To reset direction: call `dfs(node.right, False, 1)`.
6. Start from root:
   - `dfs(root.left, True, 1)`
   - `dfs(root.right, False, 1)`
7. Return `max_zigzag`.

---

### Visual Algorithm Walkthrough

```
Example Tree:
       1
        \
         1 (R)
        / \
   (L) 1   1 (R)
        \
         1 (R)
          \
           1 (R)

Tracing from Root (1):
- Move right to node: path = [R], length = 1
  - From here, move left: path = [R, L], length = 2
    - From here, move right: path = [R, L, R], length = 3!
      - From here, move right (breaks ZigZag): new path = [R], length = 1
  - From here, move right (breaks ZigZag): new path = [R], length = 1

Maximum ZigZag path found has length = 3 edges (Nodes visited = 4).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Tree
- **Input:** `root = [1,null,1,1,1,null,null,1,1,null,1,null,null,null,1]`
- **Step Trace:**
  | Node | Edge Taken | `is_left` | Incoming `length` | Continue Branch | Restart Branch | `max_zigzag` |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | Root | - | - | 0 | - | Left(1), Right(1) | 0 |
  | Right Child | Right | `False` | 1 | Left (`length = 2`) | Right (`length = 1`) | 1 |
  | Left Grandchild | Left | `True` | 2 | Right (`length = 3`) | Left (`length = 1`) | 2 |
  | Right G-Grandchild| Right | `False` | 3 | Left (`length = 4`) | Right (`length = 1`) | **3** |
- **Output:** `3`

#### Example 2: Minimal Tree (Single Node)
- **Input:** `root = [1]`
- **Step Trace:**
  - Neither `root.left` nor `root.right` exists.
  - `max_zigzag` remains `0`.
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
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        max_len = 0
        
        def dfs(node: Optional[TreeNode], is_left: bool, length: int) -> None:
            nonlocal max_len
            if not node:
                return
            
            max_len = max(max_len, length)
            
            if is_left:
                # Came from left: continue by going right, or reset by going left
                dfs(node.right, False, length + 1)
                dfs(node.left, True, 1)
            else:
                # Came from right: continue by going left, or reset by going right
                dfs(node.left, True, length + 1)
                dfs(node.right, False, 1)
                
        if root:
            if root.left:
                dfs(root.left, True, 1)
            if root.right:
                dfs(root.right, False, 1)
                
        return max_len
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
    int longestZigZag(TreeNode* root) {
        int maxLen = 0;
        if (!root) return 0;
        if (root->left) dfs(root->left, true, 1, maxLen);
        if (root->right) dfs(root->right, false, 1, maxLen);
        return maxLen;
    }

private:
    void dfs(TreeNode* node, bool isLeft, int length, int& maxLen) {
        if (!node) return;

        maxLen = std::max(maxLen, length);

        if (isLeft) {
            // Alternate right, or reset left
            dfs(node->right, false, length + 1, maxLen);
            dfs(node->left, true, 1, maxLen);
        } else {
            // Alternate left, or reset right
            dfs(node->left, true, length + 1, maxLen);
            dfs(node->right, false, 1, maxLen);
        }
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
    private int maxLen = 0;

    public int longestZigZag(TreeNode root) {
        maxLen = 0;
        if (root == null) return 0;

        if (root.left != null) {
            dfs(root.left, true, 1);
        }
        if (root.right != null) {
            dfs(root.right, false, 1);
        }

        return maxLen;
    }

    private void dfs(TreeNode node, boolean isLeft, int length) {
        if (node == null) {
            return;
        }

        maxLen = Math.max(maxLen, length);

        if (isLeft) {
            // Continue alternating to right
            dfs(node.right, false, length + 1);
            // Restart new sequence to left
            dfs(node.left, true, 1);
        } else {
            // Continue alternating to left
            dfs(node.left, true, length + 1);
            // Restart new sequence to right
            dfs(node.right, false, 1);
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Every node in the binary tree is visited at most twice during recursion (once continuing an alternating path, once as a restarted branch). Total work is linear $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(H)$ — Recursion call stack space proportional to tree height $H$. For balanced trees $H = \mathcal{O}(\log N)$, for skewed trees $H = \mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Edges vs Nodes Count:** Pay careful attention to the question: it asks for the number of *edges* (or nodes visited $- 1$). A single node path has length $0$, not $1$.
2. **Remember the Reset Branch:** A common bug is only recursing on the alternating child (`length + 1`) and omitting the same-direction child (`length = 1`). Doing so misses optimal paths that begin midway down a chain of same-direction edges!