---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 257: Binary Tree Paths"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 257: Binary Tree Paths

## Problem Statement (LeetCode 257)

**Title:** Binary Tree Paths

**Problem:**  
Given the `root` of a binary tree, return **all root-to-leaf paths** in any order.

A **leaf** is a node with no left and no right child.

Each path should be represented as a string in the format:

```python
"root->node1->node2->...->leaf"
```

---

### Example

**Input**

```
1
       / \
      2   3
       \
        5
```

**Output**

```python
["1->2->5", "1->3"]
```

---

## 2. Key Observations (Very Important)

1. This is a **tree traversal problem**
2. We must generate **all paths**, not just one → **DFS**
3. Path grows as we go down and shrinks as we backtrack
4. **Leaf node = termination condition**
5. We must **remember the path so far** → classic **backtracking**

👉 This immediately points to:

* **DFS + Backtracking**
* Or **DFS with path copy**
* Or **Iterative DFS using stack**

---

## 3. Tree Node Definition (LeetCode Standard)

```python
from typing import Optional, List

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right
```

---

## 4. Approach 1 — DFS + Backtracking (BEST & INTERVIEW FAVORITE)

### Idea

* Maintain a list `path`
* Append node value when going down
* If leaf → convert path to string
* Pop when returning (backtracking)

---

### Code (Python 3 with typing)

```python
from typing import List, Optional

class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        result: List[str] = []
        path: List[str] = []

        def dfs(node: Optional[TreeNode]) -> None:
            if not node:
                return

            # Choose
            path.append(str(node.val))

            # If leaf node → record path
            if not node.left and not node.right:
                result.append("->".join(path))
            else:
                # Explore
                dfs(node.left)
                dfs(node.right)

            # Un-choose (backtracking)
            path.pop()

        dfs(root)
        return result
```

---

## 5. Backtracking Explained Using Tree Diagram

### Tree

```
1
       / \
      2   3
       \
        5
```

---

### Step-by-Step Backtracking Flow

#### Step 1: Start at root

```
path = []
dfs(1)
path = ["1"]
```

---

#### Step 2: Go left

```
dfs(2)
path = ["1", "2"]
```

---

#### Step 3: Go right from 2

```
dfs(5)
path = ["1", "2", "5"]
```

✔ Leaf node → Save path

```python
result = ["1->2->5"]
```

---

#### Step 4: Backtrack

```
path.pop() → ["1", "2"]
path.pop() → ["1"]
```

---

#### Step 5: Go right from root

```
dfs(3)
path = ["1", "3"]
```

✔ Leaf → Save path

```python
result = ["1->2->5", "1->3"]
```

---

#### Step 6: Final Backtrack

```
path.pop() → ["1"]
path.pop() → []
```

✔ Done

---

### Backtracking Tree Visualization

```
[]
                |
              ["1"]
             /     \
     ["1","2"]    ["1","3"]
          |
    ["1","2","5"]  (leaf → save)
```

---

## 6. Approach 2 — DFS with Path Copy (Simpler, Less Efficient)

### Idea

Instead of backtracking, pass a **new path list** to recursive calls.

---

### Code

```python
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []

        result: List[str] = []

        def dfs(node: TreeNode, path: List[str]) -> None:
            path.append(str(node.val))

            if not node.left and not node.right:
                result.append("->".join(path))
                return

            if node.left:
                dfs(node.left, path.copy())
            if node.right:
                dfs(node.right, path.copy())

        dfs(root, [])
        return result
```

---

### Trade-off

| Aspect | Backtracking | Path Copy |
| --- | --- | --- |
| Memory | Optimal | Higher |
| Performance | Better | Slightly worse |
| Clarity | Moderate | Very clear |

---

## 7. Approach 3 — Iterative DFS (Stack)

### Idea

Simulate recursion using stack.

---

### Code

```python
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:
        if not root:
            return []

        result: List[str] = []
        stack = [(root, str(root.val))]

        while stack:
            node, path = stack.pop()

            if not node.left and not node.right:
                result.append(path)

            if node.right:
                stack.append((node.right, path + "->" + str(node.right.val)))

            if node.left:
                stack.append((node.left, path + "->" + str(node.left.val)))

        return result
```

---

## 8. Time & Space Complexity

### Time

```
O(N) — every node visited once
```

### Space

```
O(H) recursion stack (H = tree height)
Worst case O(N) for skewed tree
```

---

## 9. When to Use Which Approach

| Scenario | Best Approach |
| --- | --- |
| Interviews | DFS + Backtracking |
| Clean & readable | Path Copy |
| Avoid recursion | Iterative DFS |

---

## 10. Pattern Recognition (For DSA Rounds)

This problem belongs to the pattern:

> **DFS + Path Backtracking on Tree**

Similar problems:

* Path Sum II
* Sum Root to Leaf Numbers
* All Paths From Source to Target (Graph)
* Subsets / Permutations (Backtracking)

---

If you want, I can next:

* Convert this into **LeetCode pattern notes**
* Show **dry run using stack frames**
* Explain **why BFS is worse here**
* Compare with **Path Sum II**

Just tell me.