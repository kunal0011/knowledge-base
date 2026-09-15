---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - trees
  - dynamic-programming
  - divide-and-conquer
  - bst
  - amazon
  - google
---

# LeetCode 95: Unique Binary Search Trees II

**Target Companies:** Google, Amazon, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Divide & Conquer / Catalan Number Generation / BST Construction with Memoization

---

### Problem Statement

Given an integer `n`, return *all the structurally unique **BST's** (binary search trees), which has exactly `n` nodes of unique values from `1` to `n`*. Return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[Optional[TreeNode]]` — A list of root nodes representing all structurally unique BSTs.
- **Constraints:**
  - $1 \le n \le 8$

---

### Key Idea & Intuition

- **Root Selection Drives Partitioning:**
  - By the BST property, choosing node $i$ ($1 \le i \le n$) as the root forces:
    - All nodes in its left subtree to come from the interval $[1, i - 1]$.
    - All nodes in its right subtree to come from the interval $[i + 1, n]$.
- **Cartesian Product of Subtrees:**
  - The construction of the left subtree from $[start, i - 1]$ and right subtree from $[i + 1, end]$ are completely independent subproblems.
  - If $[start, i - 1]$ produces $L$ unique subtrees and $[i + 1, end]$ produces $R$ unique subtrees, then choosing $i$ as root produces $L \times R$ unique BSTs.
- **Base Case:**
  - When $start > end$, the interval is empty, which represents an empty tree: return `[None]` (a list containing a single `null` element). This allows the nested loops over left and right children to execute once.
- **Memoization:**
  - Cache results for `(start, end)` ranges to avoid re-generating identical subtree structures.

---

### Solution Approach (Step-by-Step)

1. Define `generate(start, end)`:
   - If `start > end`: return `[None]`.
   - Initialize `all_trees = []`.
   - For root value $i$ from `start` to `end`:
     - `left_trees = generate(start, i - 1)`
     - `right_trees = generate(i + 1, end)`
     - For each `l` in `left_trees`:
       - For each `r` in `right_trees`:
         - `curr_root = TreeNode(i)`
         - `curr_root.left = l`
         - `curr_root.right = r`
         - `all_trees.append(curr_root)`
   - Return `all_trees`.
2. Call `generate(1, n)` and return the list.

---

### Visual Algorithm Walkthrough

```
For n = 3: Range [1, 3]

1. Choose i = 1 as root:
   - Left range: [1, 0] -> [None]
   - Right range: [2, 3] -> 2 unique subtrees:
     (a) 2 -> right 3       (b) 3 -> left 2
   - Combined Trees:
         1             1
          \             \
           2      and    3
            \           /
             3         2

2. Choose i = 2 as root:
   - Left range: [1, 1] -> [Node(1)]
   - Right range: [3, 3] -> [Node(3)]
   - Combined Tree:
         2
        / \
       1   3

3. Choose i = 3 as root:
   - Left range: [1, 2] -> 2 unique subtrees:
     (a) 1 -> right 2       (b) 2 -> left 1
   - Right range: [4, 3] -> [None]
   - Combined Trees:
         3             3
        /             /
       1      and    2
        \           /
         2         1

Total unique trees for n = 3: 2 + 1 + 2 = 5 (Catalan number C_3 = 5).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: $n = 1$
- **Input:** `n = 1`
- **Tracing:** `i = 1`, left returns `[None]`, right returns `[None]`. One tree: `[1]`.
- **Output:** `[[1]]`

#### Example 2: $n = 3$
- **Total Trees Generated:** 5
- **Representations:**
  1. `[1,null,2,null,3]`
  2. `[1,null,3,2]`
  3. `[2,1,3]`
  4. `[3,1,null,null,2]`
  5. `[3,2,null,1]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        if n == 0:
            return []
            
        memo = {}
        
        def generate(start: int, end: int) -> List[Optional[TreeNode]]:
            if start > end:
                return [None]
            if (start, end) in memo:
                return memo[(start, end)]
                
            trees = []
            for i in range(start, end + 1):
                # All possible left subtrees
                left_subtrees = generate(start, i - 1)
                # All possible right subtrees
                right_subtrees = generate(i + 1, end)
                
                # Cartesian product of left and right subtrees
                for left in left_subtrees:
                    for right in right_subtrees:
                        root = TreeNode(i)
                        root.left = left
                        root.right = right
                        trees.append(root)
                        
            memo[(start, end)] = trees
            return trees
            
        return generate(1, n)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <map>

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
        return generate(1, n);
    }

private:
    std::vector<TreeNode*> generate(int start, int end) {
        if (start > end) {
            return {nullptr};
        }

        std::vector<TreeNode*> allTrees;
        for (int i = start; i <= end; ++i) {
            std::vector<TreeNode*> leftSubtrees = generate(start, i - 1);
            std::vector<TreeNode*> rightSubtrees = generate(i + 1, end);

            for (auto left : leftSubtrees) {
                for (auto right : rightSubtrees) {
                    TreeNode* root = new TreeNode(i);
                    root->left = left;
                    root->right = right;
                    allTrees.push_back(root);
                }
            }
        }
        return allTrees;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

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
        if (n == 0) {
            return new ArrayList<>();
        }
        return generate(1, n);
    }

    private List<TreeNode> generate(int start, int end) {
        List<TreeNode> allTrees = new ArrayList<>();
        if (start > end) {
            allTrees.add(null);
            return allTrees;
        }

        for (int i = start; i <= end; i++) {
            List<TreeNode> leftSubtrees = generate(start, i - 1);
            List<TreeNode> rightSubtrees = generate(i + 1, end);

            for (TreeNode left : leftSubtrees) {
                for (TreeNode right : rightSubtrees) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    allTrees.add(root);
                }
            }
        }
        return allTrees;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \cdot C_n)$ where $C_n = \frac{1}{n+1} \binom{2n}{n}$ is the $n$-th Catalan number. For $n = 8$, $C_8 = 1430$, resulting in $\approx 8 \times 1430 \approx 1.1 \times 10^4$ operations.
- **Space Complexity:** $\mathcal{O}(n \cdot C_n)$ to store all $C_n$ distinct generated trees, each having $n$ nodes.

---

### Takeaway Pattern & Interview Traps

1. **Base Case Trap (`[None]` vs `[]`):** Returning an empty list `[]` when `start > end` causes the inner `for left in left_trees:` or `for right in right_trees:` loops to never execute, resulting in $0$ trees generated! You must return `[None]` so that `None` is attached as a null child.
2. **Distinction from LeetCode 96:**
   - LeetCode 96 asks only for the **count** of unique trees, solved via 1D dynamic programming in $\mathcal{O}(n^2)$ time and $\mathcal{O}(n)$ space.
   - LeetCode 95 asks for the **actual trees**, necessitating structural generation and Cartesian product branching.