---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 199: Binary Tree Right Side View"
tags:
  - leetcode
  - coding
  - trees
  - amazon
  - google
---

# LeetCode 199: Binary Tree Right Side View

**Target Companies:** Amazon (Top #1 Tree Classic), Google, Meta  
**Difficulty:** Medium  
**Topic:** BFS Level-Order Traversal / DFS Right-First

---

### Problem Statement

Given the `root` of a binary tree, imagine yourself standing on the **right side** of it, return the values of the nodes you can see ordered from top to bottom.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `List[int]` containing the rightmost visible node at each depth.
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 100]$.
  - $-100 \le \text{Node.val} \le 100$

---

### Key Idea & Intuition

- **Level-Order Observation:**
  - When standing on the right side, at each horizontal level/depth $d$, exactly **one node** is visible: the rightmost node of that level.
  - If a right child does not exist, the left child becomes visible from the right!
- **Two Optimal Approaches:**
  1. **BFS (Queue Level-Order):** Process level by level. The last node popped in each level queue loop is appended to `result`.
  2. **DFS (Root -> Right -> Left):** Visit right subtree before left subtree. If the current `depth == len(result)`, this is the first time we visit this depth, so append `node.val`.

---

### Solution Approach (Step-by-Step - BFS)

1. If `not root`: return `[]`.
2. Initialize `queue = deque([root])` and `result = []`.
3. While `queue` is not empty:
   - `level_size = len(queue)`.
   - For `i` from $0$ to `level_size - 1`:
     - `node = queue.popleft()`.
     - If `i == level_size - 1`: `result.append(node.val)`.
     - If `node.left`: `queue.append(node.left)`.
     - If `node.right`: `queue.append(node.right)`.
4. Return `result`.

---

### Visual Algorithm Walkthrough

```
        1            <--- 1
      /   \
     2     3         <--- 3
      \     \
       5     4       <--- 4

Visible nodes from right side: [1, 3, 4]

Another Case (Left overhangs right):
        1            <--- 1
      /   \
     2     3         <--- 3
    /
   4                 <--- 4 (Visible because right side is empty!)

Visible nodes: [1, 3, 4]
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
            
        result = []
        queue = deque([root])
        
        while queue:
            level_len = len(queue)
            for i in range(level_len):
                node = queue.popleft()
                if i == level_len - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
        return result
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <queue>

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    std::vector<int> rightSideView(TreeNode* root) {
        if (!root) return {};
        
        std::vector<int> result;
        std::queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int levelSize = q.size();
            for (int i = 0; i < levelSize; ++i) {
                TreeNode* curr = q.front();
                q.pop();
                
                if (i == levelSize - 1) {
                    result.push_back(curr->val);
                }
                if (curr->left) q.push(curr->left);
                if (curr->right) q.push(curr->right);
            }
        }
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int val) { this.val = val; }
}

class Solution {
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        if (root == null) return result;

        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(root);

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode curr = queue.poll();
                if (i == levelSize - 1) {
                    result.add(curr.val);
                }
                if (curr.left != null) queue.offer(curr.left);
                if (curr.right != null) queue.offer(curr.right);
            }
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Every node is visited and queued exactly once.
- **Space Complexity:** $O(D)$ where $D$ is maximum tree diameter ($O(N)$ worst case for complete binary tree).
