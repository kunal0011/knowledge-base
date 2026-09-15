---
date: "2026-09-15"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal"
tags:
  - leetcode
  - coding
  - trees
  - divide-and-conquer
  - hash-table
  - amazon
  - google
---

# LeetCode 105: Construct Binary Tree from Preorder and Inorder Traversal

**Target Companies:** Amazon (Top Tier Tree Interview Question), Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Tree Construction / Divide & Conquer / Hash Map Index Lookup

---

### Problem Statement

Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the *binary tree*.

---

### Input & Output Formats & Constraints

- **Input:** `preorder: List[int]`, `inorder: List[int]`
- **Output:** `Optional[TreeNode]` (Root of the reconstructed binary tree)
- **Constraints:**
  - $1 \le \text{preorder.length} \le 3000$
  - $\text{inorder.length} == \text{preorder.length}$
  - $-3000 \le \text{preorder}[i], \text{inorder}[i] \le 3000$
  - `preorder` and `inorder` consist of **unique** values.
  - Each value of `inorder` also appears in `preorder`.
  - `preorder` is **guaranteed** to be the preorder traversal of the tree.
  - `inorder` is **guaranteed** to be the inorder traversal of the tree.

---

### Key Idea & Intuition

- **Properties of Traversals:**
  - **Preorder:** `[Root, Left Subtree..., Right Subtree...]` $\implies$ The first unvisited element is ALWAYS the root of the current subtree.
  - **Inorder:** `[Left Subtree..., Root, Right Subtree...]` $\implies$ Finding the root value splits the tree strictly into its left subtree (elements before root) and right subtree (elements after root).
- **The $\mathcal{O}(N)$ Index Map Optimization:**
  - Naive implementations search for the root value in `inorder` linearly or slice lists (`inorder[:mid]`), incurring an expensive $\mathcal{O}(N^2)$ time overhead.
  - Because all values are **unique**, we can build an $\mathcal{O}(1)$ lookup Hash Map: `in_map = {val: idx for idx, val in enumerate(inorder)}`.
  - We maintain a global pointer or counter `pre_idx` advancing linearly through `preorder`, and recurse on the subarray boundaries `[in_left, in_right]` of `inorder`.

---

### Solution Approach (Step-by-Step)

1. Build a hash map `in_map` mapping each node value to its index in `inorder`.
2. Initialize `pre_idx = 0` to track the current root in `preorder`.
3. Define recursive function `build(in_left, in_right)`:
   - If `in_left > in_right`, return `None` (empty subtree).
   - Get the root value: `root_val = preorder[pre_idx]`, then increment `pre_idx += 1`.
   - Create node `root = TreeNode(root_val)`.
   - Find root's position in `inorder`: `mid = in_map[root_val]`.
   - **Crucial Order:** Construct left subtree first: `root.left = build(in_left, mid - 1)`.
   - Construct right subtree next: `root.right = build(mid + 1, in_right)`.
   - Return `root`.
4. Call `build(0, len(inorder) - 1)` and return the result.

---

### Visual Algorithm Walkthrough

```
preorder = [3, 9, 20, 15, 7]
inorder  = [9, 3, 15, 20, 7]

in_map: {9:0, 3:1, 15:2, 20:3, 7:4}

Step 1: pre_idx = 0 -> root_val = 3
        mid in inorder = 1
        Left inorder range:  [0, 0]  (values: [9])
        Right inorder range: [2, 4]  (values: [15, 20, 7])

         3
        / \
       ?   ?

Step 2 (Build Left): pre_idx = 1 -> root_val = 9
        mid in inorder = 0
        Left range: [0, -1] -> None
        Right range: [1, 0] -> None
        Node 9 is a leaf!

         3
        / \
       9   ?

Step 3 (Build Right): pre_idx = 2 -> root_val = 20
        mid in inorder = 3
        Left inorder range:  [2, 2]  (values: [15])
        Right inorder range: [4, 4]  (values: [7])

         3
        / \
       9   20
          /  \
         15   7
```

---

### Solved Examples with Multiple Inputs

#### Example 1: General Tree
- **Input:** `preorder = [3, 9, 20, 15, 7]`, `inorder = [9, 3, 15, 20, 7]`
- **Recursive Call Trace:**
  | Call | `pre_idx` | `root_val` | `(in_left, in_right)` | `mid` | Left Range | Right Range |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | 1 | 0 | 3 | (0, 4) | 1 | (0, 0) | (2, 4) |
  | 2 | 1 | 9 | (0, 0) | 0 | (0, -1) -> Null | (1, 0) -> Null |
  | 3 | 2 | 20 | (2, 4) | 3 | (2, 2) | (4, 4) |
  | 4 | 3 | 15 | (2, 2) | 2 | (2, 1) -> Null | (3, 2) -> Null |
  | 5 | 4 | 7 | (4, 4) | 4 | (4, 3) -> Null | (5, 4) -> Null |
- **Output:** `[3, 9, 20, null, null, 15, 7]`

#### Example 2: Skewed Tree (Single Branch)
- **Input:** `preorder = [-1]`, `inorder = [-1]`
- **Step Trace:**
  - `pre_idx = 0`, `root_val = -1`, `(in_left, in_right) = (0, 0)`, `mid = 0`.
  - Left child: `(0, -1)` $\implies$ `None`.
  - Right child: `(1, 0)` $\implies$ `None`.
- **Output:** `[-1]`

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
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        in_map = {val: idx for idx, val in enumerate(inorder)}
        pre_idx = 0
        
        def helper(in_left: int, in_right: int) -> Optional[TreeNode]:
            nonlocal pre_idx
            if in_left > in_right:
                return None
            
            root_val = preorder[pre_idx]
            pre_idx += 1
            root = TreeNode(root_val)
            
            mid = in_map[root_val]
            # Recursively construct left subtree before right subtree
            root.left = helper(in_left, mid - 1)
            root.right = helper(mid + 1, in_right)
            
            return root
        
        return helper(0, len(inorder) - 1)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

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
    TreeNode* buildTree(std::vector<int>& preorder, std::vector<int>& inorder) {
        std::unordered_map<int, int> in_map;
        for (int i = 0; i < inorder.size(); ++i) {
            in_map[inorder[i]] = i;
        }
        int pre_idx = 0;
        return helper(preorder, in_map, pre_idx, 0, inorder.size() - 1);
    }

private:
    TreeNode* helper(const std::vector<int>& preorder, 
                    const std::unordered_map<int, int>& in_map,
                    int& pre_idx, int in_left, int in_right) {
        if (in_left > in_right) return nullptr;

        int root_val = preorder[pre_idx++];
        TreeNode* root = new TreeNode(root_val);
        int mid = in_map.at(root_val);

        root->left = helper(preorder, in_map, pre_idx, in_left, mid - 1);
        root->right = helper(preorder, in_map, pre_idx, mid + 1, in_right);

        return root;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

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
    private int preIdx = 0;
    private Map<Integer, Integer> inMap = new HashMap<>();

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        preIdx = 0;
        inMap.clear();
        for (int i = 0; i < inorder.length; i++) {
            inMap.put(inorder[i], i);
        }
        return helper(preorder, 0, inorder.length - 1);
    }

    private TreeNode helper(int[] preorder, int inLeft, int inRight) {
        if (inLeft > inRight) {
            return null;
        }

        int rootVal = preorder[preIdx++];
        TreeNode root = new TreeNode(rootVal);
        int mid = inMap.get(rootVal);

        root.left = helper(preorder, inLeft, mid - 1);
        root.right = helper(preorder, mid + 1, inRight);

        return root;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ — Building the hash map takes $\mathcal{O}(N)$ time. Each recursive call consumes $\mathcal{O}(1)$ hash map lookup, and exactly $N$ nodes are created and connected. Total time is strictly linear $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$ — $\mathcal{O}(N)$ space for the hash map, plus $\mathcal{O}(H)$ for the recursion stack ($H = N$ worst-case skewed tree, $H = \log N$ for balanced tree).

---

### Takeaway Pattern & Interview Traps

1. **Subtree Ordering Invariant:** Because preorder visits `Root -> Left -> Right`, `root.left` MUST be recursively called before `root.right` when incrementing a single `pre_idx` counter. Calling `root.right` first would consume preorder elements intended for the left subtree!
2. **Postorder Equivalent (LeetCode 106):** In *Construct from Inorder and Postorder*, the root is at the *end* of postorder (`post_idx = len - 1`), so you must decrement `post_idx` and construct `root.right` FIRST, followed by `root.left`.
3. **Array Slicing Trap:** In Python, doing `helper(preorder[1:mid+1], inorder[:mid])` creates full list copies at each level, ballooning time to $\mathcal{O}(N^2)$ and memory to $\mathcal{O}(N^2)$, causing TLE/MLE on large inputs. Always pass boundary indices!