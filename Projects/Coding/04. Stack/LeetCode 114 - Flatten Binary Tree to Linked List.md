---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 114: Flatten Binary Tree to Linked List"
tags:
  - leetcode
  - coding
  - stack
  - binary-tree
  - amazon
  - google
---

# LeetCode 114: Flatten Binary Tree to Linked List

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Stack / Tree Traversal / Morris Traversal

---

### Problem Statement

Given the `root` of a binary tree, flatten the tree into a "linked list":

1. The "linked list" should use the same `TreeNode` class where the `right` child pointer points to the next node in the list and the `left` child pointer is always `null`.
2. The "linked list" should be in the same order as a **pre-order traversal** of the binary tree.

You must modify the tree **in-place** without allocating new nodes.

---

### Input & Output Formats & Constraints

- **Input:**
  - `root`: `Optional[TreeNode]`, root of a binary tree.
- **Output:**
  - `None`: Modify the binary tree in-place.
- **Constraints:**
  - The number of nodes in the tree is in the range $[0, 2000]$.
  - $-100 \le \text{Node.val} \le 100$.
- **Follow-up:**
  - Can you flatten the tree in-place using $\mathcal{O}(1)$ extra space?

---

### Key Idea & Intuition

The required output sequence is the standard **pre-order traversal**:
$$\text{root} \to \text{left subtree} \to \text{right subtree}$$

After flattening, each node's `left` must be `null` and `right` points to the next node in pre-order.

#### Approach 1: Iterative Stack ($\mathcal{O}(N)$ Time, $\mathcal{O}(H)$ Space)
In pre-order traversal, when we visit a node, its left child comes before its right child.
Using a LIFO stack:
1. Push `root` to stack.
2. In each step, pop `curr`.
3. Push `curr.right` first, then `curr.left` second (so left is popped and processed first).
4. If a previous node `prev` exists, wire `prev.right = curr` and `prev.left = None`.
5. Update `prev = curr`.

#### Approach 2: Morris-Style In-Place Splicing ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space - Optimal Follow-up)
Consider any node `curr` that has a left child:
- In pre-order traversal, all nodes in `curr.left` are visited *before* the first node in `curr.right`.
- The very last node visited in the left subtree is its **rightmost node** (the in-order predecessor of `curr.right`).
- Therefore, we can find `curr`'s left subtree's rightmost node, attach `curr.right` to its `right` pointer, move `curr.left` over to `curr.right`, and set `curr.left = None`.
- Repeat this for each node as we move along `curr = curr.right`. This achieves true $\mathcal{O}(1)$ auxiliary space without recursion or an explicit stack.

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Stack-Based Traversal
1. If `root` is `null`, return immediately.
2. Initialize `stack = [root]` and `prev = null`.
3. While `stack` is not empty:
   - Pop `curr = stack.pop()`.
   - If `prev != null`:
     - `prev.right = curr`
     - `prev.left = null`
   - If `curr.right` exists, push `curr.right`.
   - If `curr.left` exists, push `curr.left`.
   - Set `prev = curr`.

#### Algorithm 2: Constant Space $\mathcal{O}(1)$ Morris In-Place
1. Initialize `curr = root`.
2. While `curr != null`:
   - If `curr.left != null`:
     - Find the rightmost node of `curr.left`:
       ```python
       predecessor = curr.left
       while predecessor.right:
           predecessor = predecessor.right
       ```
     - Attach `curr.right` to `predecessor.right`:
       `predecessor.right = curr.right`
     - Move left child to right:
       `curr.right = curr.left`
       `curr.left = None`
   - Move to next node: `curr = curr.right`.

---

### Visual Algorithm Walkthrough

Let the input tree be:
```
       1
      / \
     2   5
    / \   \
   3   4   6
```

#### Constant-Space Pointer Rewiring:
```
1. At curr = 1 (has left child 2):
   - Find rightmost node of left subtree: node 4.
   - Attach curr.right (5) to 4.right:
            1
           /
          2
         / \
        3   4
             \
              5
               \
                6
   - Move curr.left (2) to curr.right, set curr.left = null:
        1
         \
          2
         / \
        3   4
             \
              5
               \
                6

2. Advance curr to 2 (has left child 3):
   - Rightmost node of left subtree: node 3.
   - Attach curr.right (4) to 3.right:
        1
         \
          2
           \
            3
             \
              4
               \
                5
                 \
                  6

3. Advance curr through 3 -> 4 -> 5 -> 6:
   - None of these have left children, so curr advances to the end.
   - Finished! Fully flattened right-skewed list.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Tree

- **Input:** `root = [1, 2, 5, 3, 4, null, 6]`
- **Step Tracing (Stack Approach):**

| Step | Current Node | Popped From Stack | Push to Stack | `prev.right` Wired |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 1 | Stack initially `[1]` | Push 5, Push 2 | None (`prev` initialized to 1) |
| 2 | 2 | Popped 2 | Push 4, Push 3 | $1 \to 2$, $1.\text{left} = \text{null}$ |
| 3 | 3 | Popped 3 | None (leaves) | $2 \to 3$, $2.\text{left} = \text{null}$ |
| 4 | 4 | Popped 4 | None (leaves) | $3 \to 4$, $3.\text{left} = \text{null}$ |
| 5 | 5 | Popped 5 | Push 6 | $4 \to 5$, $4.\text{left} = \text{null}$ |
| 6 | 6 | Popped 6 | None | $5 \to 6$, $5.\text{left} = \text{null}$ |

- **Output:** `[1, null, 2, null, 3, null, 4, null, 5, null, 6]`

#### Example 2: Tree with Only Left Children

- **Input:** `root = [1, 2, null, 3]`
  ```
      1
     /
    2
   /
  3
  ```
- **Execution:**
  - `curr = 1`: predecessor is 3. Attach `1.right` (`null`) to `3.right`. Move `1.left` (2) to `1.right`.
  - `curr = 2`: predecessor is 3. Move `2.left` (3) to `2.right`.
  - Output: `1 -> 2 -> 3`.

#### Example 3: Empty Tree or Single Node

- **Input:** `root = []`
- **Output:** `[]` (immediately returns without action).

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(1)$ Space (Morris-Style)
```python
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional['TreeNode'] = None,
                 right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Flattens the binary tree to a linked list in-place using O(1) auxiliary space.
        """
        curr = root
        while curr:
            if curr.left:
                # Find the rightmost node of the left subtree
                predecessor = curr.left
                while predecessor.right:
                    predecessor = predecessor.right
                
                # Rewire predecessor's right to current's right
                predecessor.right = curr.right
                # Move left subtree to right and nullify left
                curr.right = curr.left
                curr.left = None
            
            curr = curr.right
```

##### Stack-Based Approach ($\mathcal{O}(H)$ Space)
```python
class SolutionStack:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root:
            return

        stack = [root]
        prev: Optional[TreeNode] = None

        while stack:
            curr = stack.pop()
            if prev:
                prev.right = curr
                prev.left = None

            # Push right first so that left is popped and processed first
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)

            prev = curr
```

#### C++17

```cpp
#include <stack>

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
    // Optimal O(1) space Morris traversal approach
    void flatten(TreeNode* root) {
        TreeNode* curr = root;
        while (curr != nullptr) {
            if (curr->left != nullptr) {
                // Find rightmost node of the left subtree
                TreeNode* predecessor = curr->left;
                while (predecessor->right != nullptr) {
                    predecessor = predecessor->right;
                }
                
                // Splice current right subtree onto predecessor right
                predecessor->right = curr->right;
                curr->right = curr->left;
                curr->left = nullptr;
            }
            curr = curr->right;
        }
    }
};
```

#### Java

```java
public class Solution {
    // Definition for a binary tree node.
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

    // Optimal O(1) auxiliary space approach
    public void flatten(TreeNode root) {
        TreeNode curr = root;
        while (curr != null) {
            if (curr.left != null) {
                // Locate rightmost node of left subtree
                TreeNode predecessor = curr.left;
                while (predecessor.right != null) {
                    predecessor = predecessor.right;
                }
                
                // Rewire pointers
                predecessor.right = curr.right;
                curr.right = curr.left;
                curr.left = null;
            }
            curr = curr.right;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - In the Morris approach, each edge is traversed at most twice (once to find the predecessor, once when advancing `curr`). Hence, the time complexity is strictly $\mathcal{O}(N)$, where $N$ is the number of nodes in the tree.
  - In the stack approach, each node is pushed and popped exactly once, taking $\mathcal{O}(N)$ time.
- **Space Complexity:**
  - **Morris Traversal Approach:** $\mathcal{O}(1)$ auxiliary space as no recursion or explicit data structures are used.
  - **Stack Approach:** $\mathcal{O}(H)$ space, where $H$ is the height of the binary tree ($\mathcal{O}(N)$ in the worst case of a skewed tree, $\mathcal{O}(\log N)$ for a balanced tree).

---

### Takeaway Pattern & Interview Traps

1. **Follow-up Expectation:**
   - When asked LC 114 in an interview (especially at Amazon or Google), the interviewer will almost certainly ask: *"Can you do this in $\mathcal{O}(1)$ auxiliary space without recursion?"*
   - Knowing the Morris-style predecessor linking pattern directly satisfies this follow-up.
2. **Pre-order Stack Ordering:**
   - If implementing with a stack, remember to push `curr.right` *before* `curr.left` so that the left child emerges on top of the stack first.
3. **Nullifying the Left Pointer:**
   - Forgetting to explicitly set `curr.left = null` is a classic bug that leads to invalid tree structures and memory cycles.