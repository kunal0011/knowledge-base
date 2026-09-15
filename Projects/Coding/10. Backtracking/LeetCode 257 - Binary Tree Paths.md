---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 257: Binary Tree Paths"
tags:
  - leetcode
  - coding
  - backtracking
  - tree
  - dfs
  - binary-tree
  - amazon
  - google
---

# LeetCode 257: Binary Tree Paths

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** Backtracking / Depth-First Search on Binary Trees  

---

### Problem Statement

Given the `root` of a binary tree, return *all root-to-leaf paths in **any order***.

A **leaf** is a node with no children (`node.left == null` and `node.right == null`).

Each path should be formatted as a string with node values joined by `"->"`.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `List[str]` containing formatted path strings.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 100]$.
  - $-100 \le \text{Node.val} \le 100$

---

### Key Idea & Intuition

- **Root-to-Leaf Backtracking:**
  - A path starts at the root and finishes exclusively at a leaf node.
  - As we traverse downward, we append the node's value to our current path buffer.
  - When we reach a leaf node (both `left` and `right` children are null), we format the current path by joining elements with `"->"` and add the resulting string to our list of results.
  - Upon returning from exploring a subtree, we backtrack by removing the node from our path buffer (`path.pop()`), ensuring that sibling subtrees receive a clean state.
- **Why Mutable Path Array Beats String Concatenation:**
  - In many languages, passing a newly concatenated string (`path + "->" + str(node.val)`) down every recursive branch creates $\mathcal{O}(H)$ temporary string allocations at every single node, resulting in $\mathcal{O}(N \cdot H)$ space and GC overhead.
  - Maintaining a single mutable list of integers or node values and formatting only at the leaf nodes is significantly more cache-friendly and space-efficient.

---

### Solution Approach (Step-by-Step)

1. **Edge Case:** If `root is None`, return `[]`.
2. **Initialize:** `results = []` and a dynamic list `path = []`.
3. **Recursive DFS `dfs(node)`:**
   - Append `str(node.val)` to `path`.
   - **Leaf Check:** If `node.left is None` and `node.right is None`:
     - Append `"->".join(path)` to `results`.
   - If `node.left` exists:
     - `dfs(node.left)`
   - If `node.right` exists:
     - `dfs(node.right)`
   - **Backtrack:** `path.pop()`.
4. Call `dfs(root)` and return `results`.

---

### Visual Algorithm Walkthrough

Consider tree:
```
       1
     /   \
    2     3
     \
      5
```

Execution trace:
```
1. Visit Node(1): path = ["1"]
2. Visit Node(2): path = ["1", "2"]
   - Left is null.
   - Right is Node(5):
     3. Visit Node(5): path = ["1", "2", "5"]
        - Leaf reached! Format: "1->2->5" -> append to results.
        - Backtrack: pop "5", path = ["1", "2"]
   - Backtrack: pop "2", path = ["1"]
4. Visit Node(3): path = ["1", "3"]
   - Leaf reached! Format: "1->3" -> append to results.
   - Backtrack: pop "3", path = ["1"]
5. Backtrack: pop "1", path = []

Final Results: ["1->2->5", "1->3"]
```

---

### Solved Examples with Multiple Inputs

| Test Case | Tree (`root`) | Leaf Nodes | Output Paths |
| :--- | :--- | :--- | :--- |
| **Standard** | `[1, 2, 3, null, 5]` | `5`, `3` | `["1->2->5", "1->3"]` |
| **Single Node** | `[1]` | `1` | `["1"]` |
| **Skewed Left** | `[1, 2, null, 3]` | `3` | `["1->2->3"]` |
| **Negative Nodes** | `[-10, 5, 20]` | `5`, `20` | `["-10->5", "-10->20"]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        """
        Returns all root-to-leaf paths using DFS backtracking.
        """
        if not root:
            return []

        results: List[str] = []
        path: List[str] = []

        def dfs(node: TreeNode) -> None:
            path.append(str(node.val))

            # Leaf condition: neither left nor right child exists
            if not node.left and not node.right:
                results.append("->".join(path))
            else:
                if node.left:
                    dfs(node.left)
                if node.right:
                    dfs(node.right)

            # Backtrack
            path.pop()

        dfs(root)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

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
    std::vector<std::string> binaryTreePaths(TreeNode* root) {
        if (!root) return {};

        std::vector<std::string> results;
        std::vector<int> path;
        dfs(root, path, results);
        return results;
    }

private:
    void dfs(TreeNode* node, std::vector<int>& path, std::vector<std::string>& results) {
        path.push_back(node->val);

        if (!node->left && !node->right) {
            // Leaf node: format path
            std::string path_str = std::to_string(path[0]);
            for (size_t i = 1; i < path.size(); ++i) {
                path_str += "->" + std::to_string(path[i]);
            }
            results.push_back(path_str);
        } else {
            if (node->left) dfs(node->left, path, results);
            if (node->right) dfs(node->right, path, results);
        }

        // Backtrack
        path.pop_back();
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

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
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> results = new ArrayList<>();
        if (root == null) return results;

        List<Integer> path = new ArrayList<>();
        dfs(root, path, results);
        return results;
    }

    private void dfs(TreeNode node, List<Integer> path, List<String> results) {
        path.add(node.val);

        if (node.left == null && node.right == null) {
            // Leaf reached: construct formatted string
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < path.size(); i++) {
                if (i > 0) sb.append("->");
                sb.append(path.get(i));
            }
            results.add(sb.toString());
        } else {
            if (node.left != null) dfs(node.left, path, results);
            if (node.right != null) dfs(node.right, path, results);
        }

        // Backtrack
        path.remove(path.size() - 1);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot H)$ where $N$ is the number of nodes and $H$ is the height of the tree.
  - Every node is visited once during the DFS traversal.
  - When reaching each of the leaves (at most $\lceil N / 2 \rceil$), constructing the path string takes $\mathcal{O}(H)$ time.
  - In a balanced tree, $H = \mathcal{O}(\log N) \implies \mathcal{O}(N \log N)$. In a worst-case skewed tree, $H = \mathcal{O}(N) \implies \mathcal{O}(N^2)$.
  - Given $N \le 100$, runtime is $< 2 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(H)$ auxiliary space for the recursion call stack and `path` buffer.
  - $H = \mathcal{O}(\log N)$ on average, $\mathcal{O}(N)$ in the worst skewed case.

---

### Takeaway Pattern & Interview Traps

- **Leaf Definition:** Do not confuse reaching a `null` child with reaching a leaf node. If you process the leaf logic inside a `null` check, you will duplicate paths (once for `left == null` and once for `right == null`). Only format the path when `node.left == null && node.right == null`.
- **String Backtracking Hygiene:** Always push and pop from a single vector/list rather than concatenating strings on every recursive argument call.