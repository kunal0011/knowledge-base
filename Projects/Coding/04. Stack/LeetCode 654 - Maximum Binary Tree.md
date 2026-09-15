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
---

# LeetCode 654: Maximum Binary Tree

Below is a **complete, structured explanation** of **LeetCode 654 – Maximum Binary Tree**, aligned with how this problem is typically expected to be understood in interviews and competitive programming.

---

## 1. Problem Statement (LeetCode 654)

You are given an integer array `nums` with **distinct elements**.

Construct a **Maximum Binary Tree** defined as follows:

1. The **root** is the **maximum number** in the array.
2. The **left subtree** is the maximum binary tree constructed from elements **left of the maximum number**.
3. The **right subtree** is the maximum binary tree constructed from elements **right of the maximum number**.

Return the **root node** of the constructed binary tree.

---

## 2. Key Observation

### Recursive Definition

The problem definition itself is **recursive**:

* Pick the maximum element → root
* Recursively apply the same logic to the left and right subarrays

### Naive Approach (Divide & Conquer)

* Find max in subarray → `O(n)`
* Do this for each recursive call
* **Worst-case time complexity:** `O(n²)` (e.g., sorted array)

This is **not optimal**.

---

## 3. Stack-Based Key Insight (Monotonic Stack)

### Critical Insight

Each element’s **parent** in the Maximum Binary Tree is:

> The **nearest greater element** on either the **left or right**,  
> whichever is **smaller among the two greater elements**.

This is identical to **Next Greater Element** logic.

### Why a Stack Works

* We maintain a **monotonic decreasing stack**
* While processing elements from left to right:

  * If current value is greater than stack top:

    * Pop smaller elements → they become **left children**
  * If stack is not empty after popping:

    * Current element becomes the **right child** of stack top

### Result

* Each element is pushed and popped **once**
* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

## 4. Algorithm (Stack Approach)

1. Initialize an empty stack
2. Iterate through `nums`
3. For each value:

   * Create a tree node
   * Pop stack while top value < current value

     * Last popped node becomes **left child**
   * If stack is not empty:

     * Current node becomes **right child** of stack top
   * Push current node onto stack
4. The **bottom-most element in stack** is the root

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None

class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        stack: List[TreeNode] = []

        for num in nums:
            curr = TreeNode(num)

            # Pop smaller elements and assign as left child
            while stack and stack[-1].val < num:
                curr.left = stack.pop()

            # Assign current node as right child
            if stack:
                stack[-1].right = curr

            stack.append(curr)

        # Root is the bottom element of stack
        return stack[0]
```

---

## 6. Worked-Out Example

### Input

```text
nums = [3, 2, 1, 6, 0, 5]
```

---

### Step-by-Step Stack Processing

| Current | Stack (top → bottom) | Action |
| --- | --- | --- |
| 3 | [] | push 3 |
| 2 | [3] | 3.right = 2 |
| 1 | [3,2] | 2.right = 1 |
| 6 | [3,2,1] | pop 1 → pop 2 → pop 3 → 6.left = 3 |
| 0 | [6] | 6.right = 0 |
| 5 | [6,0] | pop 0 → 5.left = 0 → 6.right = 5 |

---

### Final Tree Structure

```
        6
       / \
      3   5
       \  /
        2 0
         \
          1
```

---

## 7. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | `O(n)` |
| Space Complexity | `O(n)` |
| Technique Used | Monotonic Decreasing Stack |

---

## 8. Interview Takeaway

* This is **not just a tree problem**
* It is fundamentally a **Next Greater Element / Monotonic Stack** problem
* Recognizing the **parent selection rule** is the key leap

If you want, I can also:

* Compare this with the recursive `O(n²)` approach
* Show how this connects to **Cartesian Trees**
* Draw a **stack-to-tree construction diagram** step by step