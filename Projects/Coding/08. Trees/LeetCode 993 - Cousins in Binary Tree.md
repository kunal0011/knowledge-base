---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 993: Cousins in Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - bfs
  - dfs
  - amazon
  - google
---

# LeetCode 993: Cousins in Binary Tree

**Target Companies:** Amazon, Google, Microsoft, Meta  
**Difficulty:** Easy  
**Topic:** Tree BFS Level-Order / Depth & Parent Tracking / Early Exit

---

### Problem Statement

Given the `root` of a binary tree with unique values and the values of two different nodes of the tree `x` and `y`, return `true` if the nodes corresponding to the values `x` and `y` are **cousins** of each other, or `false` otherwise.

Two nodes of a binary tree are **cousins** if they have the **same depth** with **different parents**.

Note that in a binary tree, the root node is at the depth `0`, and children of each `k`-th depth node are at the depth `k + 1`.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`, `x: int`, `y: int`
- **Output:** `bool` — `true` if cousins, `false` otherwise.
- **Constraints:**
  - The number of nodes in the tree is in the range $[2, 100]$.
  - $1 \le \text{Node.val} \le 100$
  - Each node has a **unique** value.
  - $x \ne y$
  - $x$ and $y$ are exist in the tree.

---

### Key Idea & Intuition

- **Cousin Invariant:**
  - Two nodes $x$ and $y$ are cousins $\iff$
    $$\text{depth}(x) == \text{depth}(y) \quad \text{AND} \quad \text{parent}(x) \ne \text{parent}(y)$$
- **Why BFS Level-Order is Optimal:**
  - By definition, cousins must reside on the **exact same level**.
  - Using BFS with a queue storing `(node, parent)`:
    - For each horizontal level:
      - Track whether $x$ or $y$ appears on this level, along with their parents.
      - If **both** $x$ and $y$ are found on this level:
        - Return `parent_x != parent_y`.
      - If **only one** of $x$ or $y$ is found on this level:
        - They are at different depths! Return `False` immediately (early exit).
      - If **neither** is found:
        - Continue to the next level.

---

### Solution Approach (Step-by-Step)

1. Initialize `queue = deque([(root, None)])`.
2. While `queue` is not empty:
   - `level_size = len(queue)`
   - `parent_x = None`, `parent_y = None`
   - Loop `level_size` times:
     - Pop `(node, parent)`.
     - If `node.val == x`: `parent_x = parent`.
     - If `node.val == y`: `parent_y = parent`.
     - Push non-null children with `node` as parent:
       - `queue.append((node.left, node))`
       - `queue.append((node.right, node))`
   - After processing the level:
     - If both $x$ and $y$ were found: return `parent_x != parent_y`.
     - If exactly one was found: return `False`.
3. Return `False`.

---

### Visual Algorithm Walkthrough

```
Case 1: Cousins (True)
         1
       /   \
      2     3
       \     \
       [4]   [5]

Level 0: [1]
Level 1: [2 (parent 1), 3 (parent 1)]
Level 2: [4 (parent 2), 5 (parent 3)]
Both 4 and 5 found on Level 2!
Parents: parent(4) = 2 != parent(5) = 3 -> Return True.

---------------------------------------------------

Case 2: Siblings (False)
         1
       /   \
      2     3
     / \
   [4] [5]

Level 2: [4 (parent 2), 5 (parent 2)]
Both found on Level 2, but parent(4) == parent(5) == 2.
They are siblings, NOT cousins -> Return False.

---------------------------------------------------

Case 3: Different Depths (False)
         1
       /   \
     [2]    3
           /
         [4]

Level 1: [2, 3] -> 2 is found, 4 is NOT found.
Exactly one target found on Level 1 -> Return False immediately.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Cousins
- **Input:** `root = [1,2,3,null,4,null,5], x = 5, y = 4`
- **Level Trace:**
  | Level | Nodes & Parents | $x$ Found? | $y$ Found? | Decision |
  | :--- | :--- | :--- | :--- | :--- |
  | 0 | `(1, null)` | No | No | Continue |
  | 1 | `(2, 1), (3, 1)` | No | No | Continue |
  | 2 | `(4, 2), (5, 3)` | Yes ($p_y = 2$) | Yes ($p_x = 3$) | $p_x \ne p_y \implies$ **True** |
- **Output:** `true`

#### Example 2: Siblings
- **Input:** `root = [1,2,3,4], x = 4, y = 3`
- **Level Trace:**
  - Level 1: `(2, 1), (3, 1)` $\implies 3$ found on level 1, $4$ not found $\implies$ **False**.
- **Output:** `false`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional
from collections import deque

class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isCousins(self, root: Optional[TreeNode], x: int, y: int) -> bool:
        if not root:
            return False
            
        queue = deque([(root, None)])
        
        while queue:
            level_size = len(queue)
            parent_x = None
            parent_y = None
            
            for _ in range(level_size):
                node, parent = queue.popleft()
                
                if node.val == x:
                    parent_x = parent
                if node.val == y:
                    parent_y = parent
                    
                if node.left:
                    queue.append((node.left, node))
                if node.right:
                    queue.append((node.right, node))
                    
            # If both found on this level
            if parent_x and parent_y:
                return parent_x != parent_y
            # If only one found on this level, they have different depths
            if parent_x or parent_y:
                return False
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <queue>

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
    bool isCousins(TreeNode* root, int x, int y) {
        if (!root) return false;

        std::queue<std::pair<TreeNode*, TreeNode*>> q; // (node, parent)
        q.push({root, nullptr});

        while (!q.empty()) {
            int levelSize = q.size();
            TreeNode* parentX = nullptr;
            TreeNode* parentY = nullptr;

            for (int i = 0; i < levelSize; ++i) {
                auto [node, parent] = q.front();
                q.pop();

                if (node->val == x) parentX = parent;
                if (node->val == y) parentY = parent;

                if (node->left) q.push({node->left, node});
                if (node->right) q.push({node->right, node});
            }

            if (parentX && parentY) {
                return parentX != parentY;
            }
            if (parentX || parentY) {
                return false;
            }
        }

        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Queue;

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
    private static class NodeParent {
        TreeNode node;
        TreeNode parent;
        NodeParent(TreeNode node, TreeNode parent) {
            this.node = node;
            this.parent = parent;
        }
    }

    public boolean isCousins(TreeNode root, int x, int y) {
        if (root == null) return false;

        Queue<NodeParent> queue = new ArrayDeque<>();
        queue.offer(new NodeParent(root, null));

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            TreeNode parentX = null;
            TreeNode parentY = null;

            for (int i = 0; i < levelSize; i++) {
                NodeParent np = queue.poll();
                TreeNode node = np.node;
                TreeNode parent = np.parent;

                if (node.val == x) parentX = parent;
                if (node.val == y) parentY = parent;

                if (node.left != null) queue.offer(new NodeParent(node.left, node));
                if (node.right != null) queue.offer(new NodeParent(node.right, node));
            }

            if (parentX != null && parentY != null) {
                return parentX != parentY;
            }
            if (parentX != null || parentY != null) {
                return false;
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Every node in the binary tree is enqueued and dequeued at most once. As soon as the level containing the targets is evaluated, the algorithm terminates early.
- **Space Complexity:** $\mathcal{O}(W)$ where $W$ is the maximum width (diameter) of the tree, which is at most $N/2 = \mathcal{O}(N)$ nodes in the BFS queue.

---

### Takeaway Pattern & Interview Traps

1. **Siblings vs Cousins Distinction:** Siblings share the same parent and therefore are NOT cousins. The check `parent_x != parent_y` is essential.
2. **Early Exit Optimization:** If at the end of a level exactly one of `parent_x` or `parent_y` is non-null, we know for a fact that $x$ and $y$ exist at different depths. We can immediately return `False` without traversing deeper levels.