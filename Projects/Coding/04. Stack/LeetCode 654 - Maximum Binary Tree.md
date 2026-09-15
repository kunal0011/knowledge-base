---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 654: Maximum Binary Tree"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - binary-tree
  - amazon
  - google
---

# LeetCode 654: Maximum Binary Tree

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Monotonic Stack / Cartesian Tree / Binary Tree Construction

---

### Problem Statement

You are given an integer array `nums` with no duplicates. A **maximum binary tree** can be built recursively from `nums` using the following algorithm:

1. Create a root node whose value is the maximum value in `nums`.
2. Recursively build the left subtree on the subarray prefix to the left of the maximum value.
3. Recursively build the right subtree on the subarray suffix to the right of the maximum value.

Return *the root node of the maximum binary tree*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 1000$.
  - $0 \le nums[i] \le 1000$.
  - All integers in `nums` are unique.
- **Output:**
  - `Optional[TreeNode]`: Root node of the constructed maximum binary tree.
- **Constraints:**
  - All elements in `nums` are pairwise distinct.
  - The tree must maintain the property that an in-order traversal yields the original array `nums`.

---

### Key Idea & Intuition

The naive divide-and-conquer approach finds the maximum element in each subarray range $[L, R]$, splitting the problem recursively into $[L, mid-1]$ and $[mid+1, R]$. On a strictly ascending or descending array, this takes $\mathcal{O}(N^2)$ time.

To build the tree in optimal **linear $\mathcal{O}(N)$ time**, we observe that this tree is precisely a **Cartesian Tree** (a heap-ordered binary tree where an in-order traversal recovers the original array):
- For every node $x$, its parent is the **smaller of its nearest greater element to the left and its nearest greater element to the right**.
- If no greater element exists, $x$ is the global root!

#### Monotonic Decreasing Stack Construction:
We maintain a stack of `TreeNode` references in strictly decreasing order of value:
1. When a new value `num` arrives:
   - Create `curr = TreeNode(num)`.
   - While `stack` is non-empty and `stack[-1].val < num`:
     - Because `stack[-1]` appeared earlier than `num` and is smaller than `num`, `stack[-1]` must belong to the **left subtree** of `curr`!
     - Therefore: `curr.left = stack.pop()`.
   - If the stack is still non-empty after popping all smaller elements:
     - The node `stack[-1]` appeared earlier than `num` and is larger than `num`. Thus, `curr` must be the **right child** of `stack[-1]`.
     - Therefore: `stack[-1].right = curr`.
   - Push `curr` onto `stack`.
2. At the end of the array, the very bottom element of the stack (`stack[0]`) has the maximum value in the entire array and is therefore the global root.

---

### Solution Approach (Step-by-Step)

1. Initialize `stack = []` of `TreeNode` pointers.
2. For each number `num` in `nums`:
   - Instantiate `curr = TreeNode(num)`.
   - While `stack` and `stack[-1].val < num`:
     - `curr.left = stack.pop()`
   - If `stack`:
     - `stack[-1].right = curr`
   - `stack.append(curr)`
3. Return `stack[0]` (the root of the Cartesian tree).

---

### Visual Algorithm Walkthrough

Let `nums = [3, 2, 1, 6, 0, 5]`:

```
Processing elements:
-------------------------------------------------------------------------
1. num = 3:
   stack = [Node(3)]

2. num = 2:
   stack top is 3 (3 > 2).
   Node(3).right = Node(2)
   stack = [Node(3), Node(2)]

3. num = 1:
   stack top is 2 (2 > 1).
   Node(2).right = Node(1)
   stack = [Node(3), Node(2), Node(1)]

4. num = 6:
   6 > 1 -> pop Node(1), Node(6).left = Node(1)
   6 > 2 -> pop Node(2), Node(6).left = Node(2) (Node(2).right is still 1)
   6 > 3 -> pop Node(3), Node(6).left = Node(3) (Node(3).right is still 2)
   Stack is empty.
   Push Node(6).
   stack = [Node(6)]

5. num = 0:
   6 > 0 -> Node(6).right = Node(0)
   stack = [Node(6), Node(0)]

6. num = 5:
   5 > 0 -> pop Node(0), Node(5).left = Node(0)
   6 > 5 -> Node(6).right = Node(5)
   Push Node(5).
   stack = [Node(6), Node(5)]

Finished!
Root is stack[0] = Node(6).

Resulting Tree:
        6
       / \
      3   5
       \  /
        2 0
         \
          1
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Array

- **Input:** `nums = [3, 2, 1, 6, 0, 5]`
- **Output:** Root 6 with left child 3, right child 5.

#### Example 2: Strictly Ascending Array

- **Input:** `nums = [1, 2, 3]`
- **Tracing:**
  - 1 pushed.
  - 2 pops 1 $\implies$ `2.left = 1`.
  - 3 pops 2 $\implies$ `3.left = 2`.
- **Output:**
  ```
      3
     /
    2
   /
  1
  ```

#### Example 3: Strictly Descending Array

- **Input:** `nums = [3, 2, 1]`
- **Output:**
  ```
  3
   \
    2
     \
      1
  ```

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        stack: List[TreeNode] = []

        for num in nums:
            curr = TreeNode(num)

            # Pop all smaller nodes: they fall into the left subtree of curr
            while stack and stack[-1].val < num:
                curr.left = stack.pop()

            # The top remaining node is larger and to the left: curr is its right child
            if stack:
                stack[-1].right = curr

            stack.append(curr)

        # The bottom of the stack holds the maximum element of the entire array (root)
        return stack[0] if stack else None
```

#### C++17

```cpp
#include <vector>

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
    TreeNode* constructMaximumBinaryTree(const std::vector<int>& nums) {
        std::vector<TreeNode*> stack;

        for (int num : nums) {
            TreeNode* curr = new TreeNode(num);

            while (!stack.empty() && stack.back()->val < num) {
                curr->left = stack.back();
                stack.pop_back();
            }

            if (!stack.empty()) {
                stack.back()->right = curr;
            }

            stack.push_back(curr);
        }

        return stack.empty() ? nullptr : stack.front();
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public static class TreeNode {
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

    public TreeNode constructMaximumBinaryTree(int[] nums) {
        Deque<TreeNode> stack = new ArrayDeque<>();

        for (int num : nums) {
            TreeNode curr = new TreeNode(num);

            while (!stack.isEmpty() && stack.peek().val < num) {
                curr.left = stack.pop();
            }

            if (!stack.isEmpty()) {
                stack.peek().right = curr;
            }

            stack.push(curr);
        }

        return stack.peekLast(); // Bottom-most element is the global root
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Each node is pushed into the stack exactly once and popped at most once.
  - Tree pointer updates (`curr.left` and `stack[-1].right`) take $\mathcal{O}(1)$ time.
  - Total Time: $\mathcal{O}(N)$, an optimal improvement over the naive $\mathcal{O}(N^2)$ recursion.
- **Space Complexity:** $\mathcal{O}(N)$
  - The stack stores at most $N$ node references in the worst case (strictly descending values).

---

### Takeaway Pattern & Interview Traps

1. **Cartesian Tree Pattern:**
   - Any problem asking to build a binary tree where the in-order traversal matches array order and parent values exceed child values is equivalent to building a **Cartesian Tree**.
2. **Left vs Right Child Assignment:**
   - Nodes popped from the stack are *smaller* than `curr` and were located *before* `curr` in the array. Thus, the last popped node becomes `curr.left`.
   - The remaining node at `stack[-1]` is *larger* than `curr` and was located *before* `curr`. Thus, `curr` becomes `stack[-1].right`.