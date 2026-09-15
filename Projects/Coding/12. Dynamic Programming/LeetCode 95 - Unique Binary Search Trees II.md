---
date: "2026-09-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 95: Unique Binary Search Trees II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - tree
  - binary-search-tree
  - recursion
  - divide-and-conquer
  - amazon
  - google
  - bloomberg
---

# LeetCode 95: Unique Binary Search Trees II

**Target Companies:** Amazon, Google, Bloomberg, Microsoft, Uber  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Divide & Conquer / Tree Construction

---

### Problem Statement

Given an integer `n`, return *all the structurally unique **BST's** (binary search trees), which has exactly `n` nodes of unique values from `1` to `n`*. Return the answer in **any order**.

Each tree node is defined as:
```
Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

---

### Input & Output Formats & Constraints

- **Input:** `n: int`
- **Output:** `List[Optional[TreeNode]]`
- **Constraints:**
  - `1 <= n <= 8`

---

### Key Idea & Intuition

The number of structurally unique Binary Search Trees with $n$ nodes is given by the $n$-th **Catalan number**:
$$C_n = \frac{1}{n+1}\binom{2n}{n}$$

For $n=8$, $C_8 = 1430$, so the problem asks us to physically construct all $C_n$ trees.

#### BST Property & Recursive Subproblem Decomposition:
In any valid BST constructed from values in the continuous interval $[start, end]$:
1. Any element $val \in [start, end]$ can be chosen as the `root`.
2. By the Binary Search Tree invariant:
   - All keys in the left subtree must be strictly less than `root.val`. Hence, the left subtree must be formed using values in $[start, val - 1]$.
   - All keys in the right subtree must be strictly greater than `root.val`. Hence, the right subtree must be formed using values in $[val + 1, end]$.
3. If $[start, val - 1]$ yields a list of valid left subtrees $L$ and $[val + 1, end]$ yields a list of valid right subtrees $R$, then every pair $(left\_tree, right\_tree) \in L \times R$ combined with `root = TreeNode(val)` produces a distinct, valid BST for $[start, end]$.

#### Memoization / DP Optimization:
Since intervals $[start, end]$ recur during tree generation, we can memoize the list of generated trees for each sub-interval $(start, end)$ using a cache table, preventing redundant tree reconstructions.

---

### Solution Approach (Step-by-Step)

1. **Base Case:**
   - If $start > end$, there are no nodes in this range. Return a list containing a single `None` (`[None]` in Python, `[nullptr]` in C++, `[null]` in Java).
   - *Crucial detail:* Returning `[null]` ensures that the Cartesian product loops execute exactly once with a null child instead of terminating with zero combinations.
2. **Memoization Check:**
   - Check if `(start, end)` is already in `memo`. If so, return `memo[(start, end)]`.
3. **Iterate Root Choice:**
   - For every integer $val$ from $start$ to $end$:
     - Recursively call `generate(start, val - 1)` to obtain all valid left subtrees `left_trees`.
     - Recursively call `generate(val + 1, end)` to obtain all valid right subtrees `right_trees`.
     - Perform Cartesian product: for every `left` in `left_trees` and every `right` in `right_trees`:
       - Create a new node `root = TreeNode(val, left, right)`.
       - Append `root` to `all_trees`.
4. **Cache & Return:**
   - Save `all_trees` into `memo[(start, end)]` and return it.

---

### Visual Algorithm Walkthrough

#### Construction for $n = 3$ (Interval $[1, 3]$)

```
Pick root = 1:
  Left: [1, 0] -> [null]
  Right: [2, 3] -> root 2: (null, 3) -> 2 - \ 3
                   root 3: (2, null) -> 3 - / 2
  Trees with root 1:
      1            1
       \            \
        2            3
         \          /
          3        2

Pick root = 2:
  Left: [1, 1] -> 1
  Right: [3, 3] -> 3
  Tree with root 2:
        2
       / \
      1   3

Pick root = 3:
  Left: [1, 2] -> root 1: (null, 2) -> 1 - \ 2
                   root 2: (1, null) -> 2 - / 1
  Right: [4, 3] -> [null]
  Trees with root 3:
        3            3
       /            /
      1            2
       \          /
        2        1

Total Trees for n = 3: 5 distinct trees (C_3 = 5).
```

---

### Solved Examples with Multiple Inputs

| $n$ | Total Unique Trees ($C_n$) | Root Breakdown | Subtree Configurations |
| :--- | :--- | :--- | :--- |
| `1` | $C_1 = 1$ | Root `1` | `[1]` |
| `2` | $C_2 = 2$ | Root `1`: right child `2`<br>Root `2`: left child `1` | `1-\2`, `2-/1` |
| `3` | $C_3 = 5$ | Root `1`: 2 trees<br>Root `2`: 1 tree<br>Root `3`: 2 trees | Total = 5 valid BST structures |
| `4` | $C_4 = 14$ | $C_0 C_3 + C_1 C_2 + C_2 C_1 + C_3 C_0 = 5 + 2 + 2 + 5 = 14$ | 14 trees |
| `8` | $C_8 = 1430$ | Full Catalan partition over $\{1..8\}$ | 1,430 trees |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Optional, Dict, Tuple

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
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
                # Generate all left and right subtrees
                left_subtrees = build(start, root_val - 1)
                right_subtrees = build(root_val + 1, end)
                
                # Cartesian product of left and right subtrees
                for left in left_subtrees:
                    for right in right_subtrees:
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
    std::map<std::pair<int, int>, std::vector<TreeNode*>> memo;

    std::vector<TreeNode*> build(int start, int end) {
        if (start > end) {
            return {nullptr};
        }

        std::pair<int, int> key = {start, end};
        if (memo.find(key) != memo.end()) {
            return memo[key];
        }

        std::vector<TreeNode*> trees;

        for (int rootVal = start; rootVal <= end; ++rootVal) {
            std::vector<TreeNode*> leftSubtrees = build(start, rootVal - 1);
            std::vector<TreeNode*> rightSubtrees = build(rootVal + 1, end);

            for (TreeNode* left : leftSubtrees) {
                for (TreeNode* right : rightSubtrees) {
                    TreeNode* root = new TreeNode(rootVal, left, right);
                    trees.push_back(root);
                }
            }
        }

        return memo[key] = trees;
    }

public:
    std::vector<TreeNode*> generateTrees(int n) {
        if (n == 0) return {};
        memo.clear();
        return build(1, n);
    }
};
```

#### Java 17
```java
import java.util.*;

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
    private Map<String, List<TreeNode>> memo = new HashMap<>();

    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return Collections.emptyList();
        }
        return build(1, n);
    }

    private List<TreeNode> build(int start, int end) {
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
            List<TreeNode> leftSubtrees = build(start, rootVal - 1);
            List<TreeNode> rightSubtrees = build(rootVal + 1, end);

            for (TreeNode left : leftSubtrees) {
                for (TreeNode right : rightSubtrees) {
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

- **Time Complexity:** $\mathcal{O}(n \cdot C_n) = \mathcal{O}\left(\frac{4^n}{\sqrt{n}}\right)$, where $C_n = \frac{1}{n+1}\binom{2n}{n}$ is the $n$-th Catalan number. Each of the $C_n$ trees has $n$ nodes, and building each node takes $\mathcal{O}(1)$ time. For $n \le 8$, $C_8 = 1430$, requiring $\approx 1.1 \times 10^4$ operations, well within the 1-second limit.
- **Space Complexity:** $\mathcal{O}(n \cdot C_n)$ to store all structurally unique trees. The maximum recursion call stack depth is $\mathcal{O}(n)$.

---

### Takeaway Pattern & Interview Traps

1. **Base Case Must Return `[null]`, Not `[]`:**
   - If `start > end` returns an empty list `[]`, the nested loops `for left in left_subtrees:` and `for right in right_subtrees:` will execute zero iterations, destroying valid combinations where either the left or right subtree is empty (e.g., node with single child).
   - Returning `[null]` allows the Cartesian product to pair the non-empty subtree with `null`.
2. **Subtree Sharing vs. Deep Copy:**
   - In standard competitive programming and LeetCode testing, subtrees can safely share child pointers across multiple root variations because the problem only requires returning root nodes of read-only trees.
3. **Contrast with LeetCode 96:**
   - LeetCode 96 asks for the *count* of unique BSTs, requiring only $\mathcal{O}(n^2)$ scalar DP or $\mathcal{O}(n)$ Catalan formula. LeetCode 95 requires actual object generation, bounding $n \le 8$.