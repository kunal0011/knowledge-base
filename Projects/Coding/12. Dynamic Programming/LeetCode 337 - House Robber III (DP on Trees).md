---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 337: House Robber III (DP on Trees)"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 337: House Robber III (DP on Trees)

## LeetCode 337 — House Robber III (DP on Trees)

---

### Problem Statement

You are given the root of a binary tree where each node represents a house with some money.  
If you rob a house, you **cannot rob its direct children**.

Return the **maximum amount of money** you can rob without alerting the police.

---

## Why this is a DP Problem (Key Insight)

* Linear House Robber uses DP with previous states.
* A **tree has no linear order**, so we apply **Dynamic Programming on Trees**.
* At each node, you must decide:

  * **Rob this node** → cannot rob children
  * **Do not rob this node** → children may be robbed or not

This leads naturally to **two DP states per node**.

---

## DP State Definition (Most Important Part)

For every node `u`, define:

```
dp[u][0] → Maximum money if we DO NOT rob node u
dp[u][1] → Maximum money if we DO rob node u
```

These two states fully capture all valid configurations.

---

## DP Transition (State Relations)

Let:

* `L` = left child
* `R` = right child

### Case 1: Rob current node (`dp[u][1]`)

If we rob `u`, we **cannot rob its children**:

```
dp[u][1] = u.val + dp[L][0] + dp[R][0]
```

---

### Case 2: Do NOT rob current node (`dp[u][0]`)

If we skip `u`, we are free to choose the best option for each child:

```
dp[u][0] = max(dp[L][0], dp[L][1]) + max(dp[R][0], dp[R][1])
```

---

## Base Case

For a `null` node:

```
dp[null][0] = 0
dp[null][1] = 0
```

---

## Final Answer

At the root `r`:

```
answer = max(dp[r][0], dp[r][1])
```

---

## Example Walkthrough (DP Table Creation)

### Example Tree

```
        3
       / \
      2   3
       \   \
        3   1
```

---

### Step 1: Leaf Nodes

#### Node 3 (left-right leaf)

```
dp[3][1] = 3
dp[3][0] = 0
```

#### Node 1 (right-right leaf)

```
dp[1][1] = 1
dp[1][0] = 0
```

---

### Step 2: Internal Nodes

#### Node 2

Children: right child = 3

```
dp[2][1] = 2 + dp[3][0] = 2
dp[2][0] = max(0,3) = 3
```

---

#### Node 3 (right child of root)

Children: right child = 1

```
dp[3][1] = 3 + dp[1][0] = 3
dp[3][0] = max(0,1) = 1
```

---

### Step 3: Root Node (3)

Children: 2 and 3

```
dp[3][1] = 3 + dp[2][0] + dp[3][0]
         = 3 + 3 + 1
         = 7

dp[3][0] = max(3,2) + max(1,3)
         = 3 + 3
         = 6
```

---

### Final Result

```
max(7, 6) = 7
```

---

## DP Table Summary

| Node | dp[0] (Not Rob) | dp[1] (Rob) |
| --- | --- | --- |
| 3 (leaf) | 0 | 3 |
| 1 | 0 | 1 |
| 2 | 3 | 2 |
| 3 | 1 | 3 |
| Root 3 | 6 | 7 |

---

## Python 3 Solution (Typed)

```python
from typing import Optional, Tuple

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:

        def dfs(node: Optional[TreeNode]) -> Tuple[int, int]:
            if not node:
                return (0, 0)

            left_not, left_rob = dfs(node.left)
            right_not, right_rob = dfs(node.right)

            rob = node.val + left_not + right_not
            not_rob = max(left_not, left_rob) + max(right_not, right_rob)

            return (not_rob, rob)

        not_rob, rob = dfs(root)
        return max(not_rob, rob)
```

---

## Time and Space Complexity

* **Time:** `O(N)` — each node processed once
* **Space:** `O(H)` — recursion stack (`H = height of tree`)

---

## Core Takeaway

> Tree DP almost always means **“define states per node”** and combine children bottom-up.

If you want, I can:

* Convert this into a **postorder DP tree diagram**
* Compare with **LeetCode 198 (linear)**
* Explain **why greedy fails here**
* Show an **iterative version**

Just tell me.