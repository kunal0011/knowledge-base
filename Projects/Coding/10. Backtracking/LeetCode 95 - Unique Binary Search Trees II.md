---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - backtracking
  - tree
  - binary-search-tree
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 95: Unique Binary Search Trees II

**Target Companies:** Amazon, Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Divide and Conquer / Catalan Trees / Memoization  

---

### Problem Statement

Given an integer `n`, return *all the structurally unique **BST**'s (binary search trees), which has exactly `n` nodes of unique values from `1` to `n`*. Return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[Optional[TreeNode]]` containing the root pointers to all unique BST topologies.
- **Constraints:**
  - $1 \le n \le 8$

---

### Key Idea & Intuition

- **BST Structural Invariant:**
  - In any Binary Search Tree, all values in the left subtree must be strictly less than the root, and all values in the right subtree must be strictly greater than the root.
- **Recursive Decomposition by Interval $[start, end]$:**
  - For an interval $[start, end]$ of candidate values:
    - Pick any value $i \in [start, end]$ to act as the root.
    - All values in $[start, i - 1]$ must form valid BSTs for the left subtree.
    - All values in $[i + 1, end]$ must form valid BSTs for the right subtree.
  - Recursively generate:
    - $\mathcal{L} = \text{generate}(start, i - 1)$
    - $\mathcal{R} = \text{generate}(i + 1, end)$
- **Cartesian Product Assembly:**
  - Every valid left subtree $L \in \mathcal{L}$ can pair with every valid right subtree $R \in \mathcal{R}$.
  - For each pair $(L, R)$, instantiate a new node `TreeNode(i, left=L, right=R)`.
- **Base Case Formulation:**
  - If $start > end$, there are no values in this subtree $\implies$ return `[None]`. Having `[None]` ensures that the nested loop executes at least once for nodes missing a left or right child.
- **Memoization (`memo[(start, end)]`):**
  - Sub-intervals $[start, end]$ are repeatedly requested across different subtrees. Caching `memo[(start, end)]` eliminates redundant construction.

---

### Solution Approach (Step-by-Step)

1. Create a memoization dictionary `memo = {}`.
2. Define `generate_trees(start, end)`:
   - If `start > end`: return `[None]`.
   - If `(start, end)` in `memo`: return `memo[(start, end)]`.
   - Initialize `all_trees = []`.
   - For `root_val` from `start` to `end`:
     - `left_subtrees = generate_trees(start, root_val - 1)`
     - `right_subtrees = generate_trees(root_val + 1, end)`
     - For `left_tree` in `left_subtrees`:
       - For `right_tree` in `right_subtrees`:
         - Create `curr_root = TreeNode(root_val, left_tree, right_tree)`.
         - Append `curr_root` to `all_trees`.
   - Store `memo[(start, end)] = all_trees`.
   - Return `all_trees`.
3. Call `generate_trees(1, n)` and return the list.

---

### Visual Algorithm Walkthrough

For $n = 3$, candidate roots: $\{1, 2, 3\}$.

```
1. Root = 1:
   - Left: [] -> None
   - Right: [2, 3] -> (2->3) and (3->2)
   => 1             1
       \             \
        2             3
         \           /
          3         2

2. Root = 2:
   - Left: [1] -> 1
   - Right: [3] -> 3
   =>   2
       / \
      1   3

3. Root = 3:
   - Left: [1, 2] -> (1->2) and (2->1)
   - Right: [] -> None
   =>     3            3
         /            /
        1            2
         \          /
          2        1

Total structurally unique BSTs = C_3 = 5 trees!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | Number of BSTs ($C_n$) | Topologies Generated |
| :--- | :--- | :--- | :--- |
| **n = 1** | `1` | $C_1 = 1$ | `[[1]]` |
| **n = 2** | `2` | $C_2 = 2$ | `1->2` and `2->1` |
| **n = 3** | `3` | $C_3 = 5$ | 5 distinct trees |
| **n = 4** | `4` | $C_4 = 14$ | 14 distinct trees |
| **n = 8 (Max)** | `8` | $C_8 = 1430$ | 1,430 distinct trees |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import Optional, List, Dict, Tuple

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        """
        Generates all structurally unique BSTs with values 1 to n.
        Uses divide-and-conquer with memoization on interval [start, end].
        """
        if n == 0:
            return []

        memo: Dict[Tuple[int, int], List[Optional[TreeNode]]] = {}

        def build(start: int, end: int) -> List[Optional[TreeNode]]:
            if start > end:
                return [None]

            if (start, end) in memo:
                return memo[(start, end)]

            trees: List[Optional[TreeNode]] = []

            for root_val in range(start, end + 1):
                # All possible left and right subtrees
                left_trees = build(start, root_val - 1)
                right_trees = build(root_val + 1, end)

                # Cartesian product assembly
                for left in left_trees:
                    for right in right_trees:
                        root = TreeNode(root_val, left, right)
                        trees.append(root)

            memo[(start, end)] = trees
            return trees

        return build(1, n)
```

#### C++17
```cpp
#include <vector>
#include <map>

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
    std::vector<TreeNode*> generateTrees(int n) {
        if (n == 0) return {};
        std::map<std::pair<int, int>, std::vector<TreeNode*>> memo;
        return build(1, n, memo);
    }

private:
    std::vector<TreeNode*> build(int start, int end,
                                 std::map<std::pair<int, int>, std::vector<TreeNode*>>& memo) {
        if (start > end) {
            return {nullptr};
        }

        auto key = std::make_pair(start, end);
        if (memo.count(key)) {
            return memo[key];
        }

        std::vector<TreeNode*> trees;

        for (int root_val = start; root_val <= end; ++root_val) {
            std::vector<TreeNode*> left_trees = build(start, root_val - 1, memo);
            std::vector<TreeNode*> right_trees = build(root_val + 1, end, memo);

            for (TreeNode* left : left_trees) {
                for (TreeNode* right : right_trees) {
                    TreeNode* root = new TreeNode(root_val, left, right);
                    trees.push_back(root);
                }
            }
        }

        memo[key] = trees;
        return trees;
    }
};
```

#### Java
```java
import java.util.*;

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
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new ArrayList<>();
        Map<String, List<TreeNode>> memo = new HashMap<>();
        return build(1, n, memo);
    }

    private List<TreeNode> build(int start, int end, Map<String, List<TreeNode>> memo) {
        List<TreeNode> trees = new ArrayList<>();
        if (start > end) {
            trees.add(null);
            return trees;
        }

        String key = start + "-" + end;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }

        for (int rootVal = start; rootVal <= end; rootVal++) {
            List<TreeNode> leftTrees = build(start, rootVal - 1, memo);
            List<TreeNode> rightTrees = build(rootVal + 1, end, memo);

            for (TreeNode left : leftTrees) {
                for (TreeNode right : rightTrees) {
                    TreeNode root = new TreeNode(rootVal, left, right);
                    trees.add(root);
                }
            }
        }

        memo.put(key, trees);
        return trees;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(C_n \cdot n) = \mathcal{O}\left(\frac{4^n}{\sqrt{n}}\right)$.
  - The number of structurally unique trees generated is the $n$-th Catalan number $C_n = \frac{1}{n+1}\binom{2n}{n}$.
  - Copying and allocating each tree with $n$ nodes takes $\mathcal{O}(n)$ time.
  - For $n = 8$, $C_8 = 1430$, yielding $1430 \times 8 \approx 1.1 \times 10^4$ operations, completing in $< 5 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(C_n \cdot n)$ to store all constructed tree nodes in memory, with recursion stack depth $\mathcal{O}(n)$.

---

### Takeaway Pattern & Interview Traps

- **Base Case `[None]` vs `[]`:**
  - If you return `[]` when `start > end`, the nested loops `for left in left_trees:` or `for right in right_trees:` will execute 0 times, completely dropping valid trees where a node has only one child!
  - Returning `[None]` ensures the product loops execute with a `None` pointer for empty subtrees.
- **Node Cloning in Tree Reuse:** When using memoization, identical subtree pointers are shared across multiple parent trees. This is standard in DAG tree representations, but in environments requiring deep clones, make copies during assignment.