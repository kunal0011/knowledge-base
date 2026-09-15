---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 987: Vertical Order Traversal of a Binary Tree"
tags:
  - leetcode
  - coding
  - trees
  - bfs
  - dfs
  - sorting
  - amazon
  - google
---

# LeetCode 987: Vertical Order Traversal of a Binary Tree

**Target Companies:** Meta (Top Tier Tree Question), Amazon, Google, Bloomberg  
**Difficulty:** Hard  
**Topic:** 2D Coordinate System Mapping / Tree Traversal / Multi-Key Sorting

---

### Problem Statement

Given the `root` of a binary tree, calculate the **vertical order traversal** of the binary tree.

For each node at position `(row, col)`, its left and right children will be at positions `(row + 1, col - 1)` and `(row + 1, col + 1)` respectively. The root of the tree is at `(0, 0)`.

The **vertical order traversal** of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their **values**.

Return the **vertical order traversal** of the binary tree.

---

### Input & Output Formats & Constraints

- **Input:** `root: Optional[TreeNode]`
- **Output:** `List[List[int]]` — Grouped lists of node values from leftmost column to rightmost column.
- **Constraints:**
  - The number of nodes in the tree is in the range $[1, 1000]$.
  - $0 \le \text{Node.val} \le 1000$

---

### Key Idea & Intuition

- **Coordinate System Assignment:**
  - Imagine the tree laid out on a 2D Cartesian grid:
    - Root has coordinates: $(r = 0, c = 0)$.
    - `node.left` has coordinates: $(r + 1, c - 1)$.
    - `node.right` has coordinates: $(r + 1, c + 1)$.
- **Multi-Level Ordering Rules:**
  - Group nodes by column index $c$ in ascending order (leftmost to rightmost).
  - Within the same column $c$:
    1. Order nodes by row index $r$ ascending (top to bottom).
    2. **Tie-breaker:** If two nodes have the exact same $(r, c)$, order them by their **node values** in ascending order.
- **Unified Triplet Sorting:**
  - During DFS or BFS, record each node as a tuple: `(col, row, val)`.
  - Sorting all tuples using standard lexicographical comparison automatically satisfies all three sorting requirements simultaneously!
  - Finally, group the sorted values by `col`.

---

### Solution Approach (Step-by-Step)

1. Initialize `node_list = []`.
2. Traverse tree using DFS: `dfs(node, row, col)`:
   - If `not node`: return.
   - Append `(col, row, node.val)` to `node_list`.
   - `dfs(node.left, row + 1, col - 1)`
   - `dfs(node.right, row + 1, col + 1)`
3. Sort `node_list` (in Python, sorting tuples `(col, row, val)` naturally sorts by `col`, then `row`, then `val`).
4. Iterate through sorted entries, grouping values sharing the same `col` into sublists.
5. Return the grouped lists.

---

### Visual Algorithm Walkthrough

```
Binary Tree:
          1 (0, 0)
        /   \
(1, -1) 2     3 (1, 1)
       / \   / \
(2, -2)4 5   6   7 (2, 2)
       (2, 0)(2, 0)

Nodes mapped to (col, row, val):
- Node 4: col = -2, row = 2
- Node 2: col = -1, row = 1
- Node 1: col =  0, row = 0
- Node 5: col =  0, row = 2
- Node 6: col =  0, row = 2
- Node 3: col =  1, row = 1
- Node 7: col =  2, row = 2

Tie-breaker at col = 0, row = 2:
Node 5 and Node 6 have identical (row, col) = (2, 0).
Sorted by value: 5 comes before 6.

Grouping by columns:
col = -2: [4]
col = -1: [2]
col =  0: [1, 5, 6]
col =  1: [3]
col =  2: [7]

Output: [[4], [2], [1, 5, 6], [3], [7]]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Overlapping Coordinates with Tie-breaking
- **Input:** `root = [1,2,3,4,5,6,7]`
- **Step Trace Table:**
  | Node Val | Row | Col | Tuple `(col, row, val)` |
  | :--- | :--- | :--- | :--- |
  | 4 | 2 | -2 | `(-2, 2, 4)` |
  | 2 | 1 | -1 | `(-1, 1, 2)` |
  | 1 | 0 | 0 | `(0, 0, 1)` |
  | 5 | 2 | 0 | `(0, 2, 5)` |
  | 6 | 2 | 0 | `(0, 2, 6)` |
  | 3 | 1 | 1 | `(1, 1, 3)` |
  | 7 | 2 | 2 | `(2, 2, 7)` |
- **Sorted Grouping:** `col = -2: [4]`, `col = -1: [2]`, `col = 0: [1, 5, 6]`, `col = 1: [3]`, `col = 2: [7]`
- **Output:** `[[4], [2], [1, 5, 6], [3], [7]]`

#### Example 2: Non-overlapping Skewed Columns
- **Input:** `root = [3,9,20,null,null,15,7]`
- **Grouping:**
  - `col = -1`: `[9]`
  - `col = 0`: `[3, 15]`
  - `col = 1`: `[20]`
  - `col = 2`: `[7]`
- **Output:** `[[9], [3, 15], [20], [7]]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Optional
from collections import defaultdict

class TreeNode:
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        node_list = []
        
        def dfs(node: Optional[TreeNode], row: int, col: int) -> None:
            if not node:
                return
            node_list.append((col, row, node.val))
            dfs(node.left, row + 1, col - 1)
            dfs(node.right, row + 1, col + 1)
            
        dfs(root, 0, 0)
        
        # Sort primarily by col, then by row, then by value
        node_list.sort()
        
        result = []
        curr_col = None
        for col, row, val in node_list:
            if col != curr_col:
                result.append([])
                curr_col = col
            result[-1].append(val)
            
        return result
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <tuple>
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
    std::vector<std::vector<int>> verticalTraversal(TreeNode* root) {
        std::vector<std::tuple<int, int, int>> nodes; // (col, row, val)
        dfs(root, 0, 0, nodes);

        // Sort by col, then row, then value
        std::sort(nodes.begin(), nodes.end());

        std::vector<std::vector<int>> result;
        int lastCol = -1e9;

        for (const auto& [col, row, val] : nodes) {
            if (col != lastCol) {
                result.push_back({});
                lastCol = col;
            }
            result.back().push_back(val);
        }

        return result;
    }

private:
    void dfs(TreeNode* node, int row, int col, std::vector<std::tuple<int, int, int>>& nodes) {
        if (!node) return;
        nodes.push_back({col, row, node->val});
        dfs(node->left, row + 1, col - 1, nodes);
        dfs(node->right, row + 1, col + 1, nodes);
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
    private static class NodeInfo implements Comparable<NodeInfo> {
        int col, row, val;

        NodeInfo(int col, int row, int val) {
            this.col = col;
            this.row = row;
            this.val = val;
        }

        @Override
        public int compareTo(NodeInfo o) {
            if (this.col != o.col) return Integer.compare(this.col, o.col);
            if (this.row != o.row) return Integer.compare(this.row, o.row);
            return Integer.compare(this.val, o.val);
        }
    }

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        List<NodeInfo> nodes = new ArrayList<>();
        dfs(root, 0, 0, nodes);

        Collections.sort(nodes);

        List<List<Integer>> result = new ArrayList<>();
        Integer lastCol = null;

        for (NodeInfo node : nodes) {
            if (lastCol == null || node.col != lastCol) {
                result.add(new ArrayList<>());
                lastCol = node.col;
            }
            result.get(result.size() - 1).add(node.val);
        }

        return result;
    }

    private void dfs(TreeNode node, int row, int col, List<NodeInfo> nodes) {
        if (node == null) return;
        nodes.add(new NodeInfo(col, row, node.val));
        dfs(node.left, row + 1, col - 1, nodes);
        dfs(node.right, row + 1, col + 1, nodes);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$ — Collecting coordinates during DFS takes $\mathcal{O}(N)$ time. Sorting the $N$ node triplets takes $\mathcal{O}(N \log N)$ time. Grouping the sorted entries takes linear time $\mathcal{O}(N)$. Overall runtime is dominated by sorting: $\mathcal{O}(N \log N)$.
- **Space Complexity:** $\mathcal{O}(N)$ — Storing the coordinate triples requires $\mathcal{O}(N)$ memory. The recursion stack consumes $\mathcal{O}(H) \le \mathcal{O}(N)$ frames.

---

### Takeaway Pattern & Interview Traps

1. **Difference with LeetCode 314 (*Binary Tree Vertical Order Traversal*):**
   - In LeetCode 314, when nodes share both row and column, their order is determined by left-to-right tree insertion order (standard BFS).
   - In LeetCode 987, when nodes share both row and column, they MUST be sorted by their **numerical values**. Sorting tuples `(col, row, val)` automatically handles this condition.
2. **Column and Row Progression:** Be careful with signs: going left decreases column ($c - 1$), going right increases column ($c + 1$), but descending always increases row ($r + 1$).