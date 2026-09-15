---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 113: Path Sum II"
tags:
  - leetcode
  - coding
  - backtracking
  - tree
  - dfs
  - amazon
  - google
---

# LeetCode 113: Path Sum II

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Depth-First Search on Trees  

---

### Problem Statement

Given the `root` of a binary tree and an integer `targetSum`, return *all **root-to-leaf** paths where the sum of the node values in the path equals `targetSum`*. Each path should be returned as a list of the node values, not node references.

A **leaf** is a node with no children (`node.left == null` and `node.right == null`).

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`, `targetSum: int`
- **Output:** `List[List[int]]` containing all valid root-to-leaf paths.
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 5000]$.
  - $-1000 \le \text{Node.val} \le 1000$
  - $-1000 \le \text{targetSum} \le 1000$

---

### Key Idea & Intuition

- **Strict Root-to-Leaf Traversal:**
  - A path must start at the root node and terminate at a leaf node. Intermediate matches (paths that end before a leaf) do not count.
- **Backtracking State Maintenance:**
  - Instead of copying a new list at every recursive step ($\mathcal{O}(H^2)$ copying overhead per branch), maintain a single global/shared dynamic list `current_path`.
  - When entering a node: `current_path.append(node.val)`, subtract `node.val` from remaining sum (`remaining_sum - node.val`).
  - When at a leaf (`not node.left and not node.right`):
    - If `remaining_sum == node.val` (or after subtraction `remaining_sum == 0`), append a copy of `current_path` to the result list.
  - When returning from a node's recursive exploration: `current_path.pop()` to restore state for sibling subtrees.
- **Negative Values & Pruning Pitfall:**
  - Because node values and `targetSum` can be negative, **we cannot prune** when `current_sum > targetSum` or `remaining_sum < 0`. A path with positive sum can decrease later if negative nodes exist deeper down. Traversal must explore to every leaf unless the tree is empty.

---

### Solution Approach (Step-by-Step)

1. **Base Case:**
   - If `root is None`, return `[]`.
2. **Recursive Helper `dfs(node, remaining)`:**
   - Append `node.val` to `current_path`.
   - Check if `node` is a leaf:
     - If `node.left is None` and `node.right is None`:
       - If `remaining == node.val`, append a snapshot `list(current_path)` to `results`.
   - Recursively call `dfs(node.left, remaining - node.val)` if `node.left` exists.
   - Recursively call `dfs(node.right, remaining - node.val)` if `node.right` exists.
   - **Backtrack:** Pop `node.val` from `current_path`.
3. Return `results`.

---

### Visual Algorithm Walkthrough

Consider tree with `targetSum = 22`:
```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```

Execution trace for the left subtree:
```
1. Visit Node(5):  path = [5], remaining = 17
2. Visit Node(4):  path = [5, 4], remaining = 13
3. Visit Node(11): path = [5, 4, 11], remaining = 2
   - Visit Node(7): path = [5, 4, 11, 7], remaining = -5
     Leaf check: remaining (-5) != 0 -> Backtrack: pop 7, path = [5, 4, 11]
   - Visit Node(2): path = [5, 4, 11, 2], remaining = 0
     Leaf check: remaining == 0 -> Found Match! Add [5, 4, 11, 2] to results.
     Backtrack: pop 2, path = [5, 4, 11]
4. Backtrack: pop 11, path = [5, 4]
5. Backtrack: pop 4, path = [5]

Execution trace for right subtree:
6. Visit Node(8):  path = [5, 8], remaining = 9
   - Visit Node(13): path = [5, 8, 13], remaining = -4 (Leaf, no match) -> pop 13
   - Visit Node(4):  path = [5, 8, 4], remaining = 5
     - Visit Node(5): path = [5, 8, 4, 5], remaining = 0 -> Leaf match! Add [5, 8, 4, 5].
       pop 5
     - Visit Node(1): path = [5, 8, 4, 1], remaining = 4 -> Leaf, no match.
       pop 1
     pop 4
   pop 8
7. pop 5 -> Done!

Results: [[5, 4, 11, 2], [5, 8, 4, 5]]
```

---

### Solved Examples with Multiple Inputs

| Test Case | Tree (`root`) | `targetSum` | Valid Leaf Paths | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard Example** | `[5,4,8,11,null,13,4,7,2,null,null,5,1]` | `22` | `5->4->11->2` (sum 22), `5->8->4->5` (sum 22) | `[[5,4,11,2],[5,8,4,5]]` |
| **No Valid Path** | `[1,2,3]` | `5` | Leaves `2` (sum 3), `3` (sum 4) | `[]` |
| **Single Node Match** | `[1]` | `1` | `1` is leaf, sum 1 | `[[1]]` |
| **Negative Values** | `[1,-2,-3,1,3,-2,null,-1]` | `-1` | `1->-2->1->-1` (sum -1) | `[[1,-2,1,-1]]` |
| **Empty Tree** | `[]` | `0` | No nodes | `[]` |

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
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        """
        Finds all root-to-leaf paths that sum to targetSum using backtracking DFS.
        """
        results: List[List[int]] = []
        current_path: List[int] = []

        def dfs(node: Optional[TreeNode], remaining: int) -> None:
            if not node:
                return

            # Choose
            current_path.append(node.val)
            remaining -= node.val

            # Check if leaf node
            if node.left is None and node.right is None:
                if remaining == 0:
                    results.append(list(current_path))
            else:
                if node.left:
                    dfs(node.left, remaining)
                if node.right:
                    dfs(node.right, remaining)

            # Backtrack
            current_path.pop()

        dfs(root, targetSum)
        return results
```

#### C++17
```cpp
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
    std::vector<std::vector<int>> pathSum(TreeNode* root, int targetSum) {
        std::vector<std::vector<int>> results;
        std::vector<int> current_path;
        dfs(root, targetSum, current_path, results);
        return results;
    }

private:
    void dfs(TreeNode* node, int remaining, std::vector<int>& current_path, std::vector<std::vector<int>>& results) {
        if (!node) return;

        // Choose
        current_path.push_back(node->val);
        remaining -= node->val;

        // If leaf node
        if (!node->left && !node->right) {
            if (remaining == 0) {
                results.push_back(current_path);
            }
        } else {
            if (node->left) dfs(node->left, remaining, current_path, results);
            if (node->right) dfs(node->right, remaining, current_path, results);
        }

        // Backtrack
        current_path.pop_back();
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
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> currentPath = new ArrayList<>();
        dfs(root, targetSum, currentPath, results);
        return results;
    }

    private void dfs(TreeNode node, int remaining, List<Integer> currentPath, List<List<Integer>> results) {
        if (node == null) return;

        // Choose
        currentPath.add(node.val);
        remaining -= node.val;

        // Check leaf
        if (node.left == null && node.right == null) {
            if (remaining == 0) {
                results.add(new ArrayList<>(currentPath));
            }
        } else {
            if (node.left != null) dfs(node.left, remaining, currentPath, results);
            if (node.right != null) dfs(node.right, remaining, currentPath, results);
        }

        // Backtrack
        currentPath.remove(currentPath.size() - 1);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N$ is the number of nodes in the tree.
  - In the worst case (e.g., a balanced tree where every root-to-leaf path sums to `targetSum`), there are $\mathcal{O}(N/2)$ leaves and each path has length $\mathcal{O}(\log N)$ or $\mathcal{O}(N)$ in a skewed tree, resulting in $\mathcal{O}(N^2)$ time to copy the paths to the output list. For general trees, traversal visits each node once in $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(H)$ auxiliary space for the recursion stack and `current_path`, where $H$ is the height of the tree ($H = \mathcal{O}(\log N)$ balanced, $\mathcal{O}(N)$ skewed).
  - Total space including output is $\mathcal{O}(N \cdot H)$.

---

### Takeaway Pattern & Interview Traps

- **Pass-by-Reference with Backtracking vs Cloning:** Creating a new list at each recursive call (`dfs(node.left, path + [node.val])`) leads to unnecessary $\mathcal{O}(N^2)$ memory allocation. Always pass a single shared list and `pop()` upon backtrack.
- **Deep Copy on Result Insertion:** When adding `current_path` to `results`, remember to make a shallow copy (`list(current_path)` / `new ArrayList<>(currentPath)`); otherwise, subsequent `pop()` operations will mutate or empty the paths in `results`.
- **Leaf Definition Check:** A node with only one `null` child is **not** a leaf! Checking `remaining == 0` at a node that has a single child will mistakenly record a partial path. Both `node.left == null` and `node.right == null` must hold.