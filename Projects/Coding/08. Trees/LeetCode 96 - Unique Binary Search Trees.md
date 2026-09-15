---
date: "2025-12-24"
type: leetcode-solution
category: "Trees"
folder: "08. Trees"
title: "LeetCode 96: Unique Binary Search Trees"
tags:
  - leetcode
  - coding
  - trees
---

# LeetCode 96: Unique Binary Search Trees

Below is a structured, interview-ready explanation of **LeetCode 96 – Unique Binary Search Trees**, aligned with your usual format.

---

## 📘 LeetCode 96: Unique Binary Search Trees

### 🔹 Problem Statement

Given an integer `n`, return the **number of structurally unique Binary Search Trees (BSTs)** that can be formed using values from `1` to `n`.

A BST must satisfy:

* All nodes in the left subtree are **less than** the root.
* All nodes in the right subtree are **greater than** the root.
* Both subtrees are themselves BSTs.

---

## 🔹 Key Observation & Core Concepts

### 1. **Choice of Root Decomposes the Problem**

If we choose a number `i` (1 ≤ i ≤ n) as the root:

* Left subtree will contain values `[1 … i-1]`
* Right subtree will contain values `[i+1 … n]`

Let:

* `L = number of nodes in left subtree = i - 1`
* `R = number of nodes in right subtree = n - i`

The number of unique BSTs with root `i` is:

```
(number of left BSTs) × (number of right BSTs)
= dp[L] × dp[R]
```

---

### 2. **Independence of Subproblems**

* Left and right subtrees are **independent**
* Any valid left subtree can be combined with any valid right subtree

This is a classic **Cartesian product** situation.

---

### 3. **Catalan Number Structure**

The recurrence relation matches the **Catalan numbers**:

[  
dp[n] = \sum\_{i=1}^{n} dp[i-1] \times dp[n-i]  
]

Where:

* `dp[n]` = number of unique BSTs using `n` nodes

---

### 4. **Base Case**

* `dp[0] = 1` → empty tree is a valid BST
* `dp[1] = 1` → only one possible tree

---

## 🔹 Dynamic Programming State Definition

| State | Meaning |
| --- | --- |
| `dp[k]` | Number of unique BSTs formed using `k` nodes |

---

## 🔹 State Transition

For each `n`:

```
dp[n] = Σ dp[left_nodes] × dp[right_nodes]
       where left_nodes = i - 1
             right_nodes = n - i
             i ranges from 1 to n
```

---

## 🔹 Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def numTrees(self, n: int) -> int:
        # dp[i] = number of unique BSTs with i nodes
        dp: List[int] = [0] * (n + 1)
        
        dp[0] = 1  # empty tree
        dp[1] = 1  # single node
        
        for nodes in range(2, n + 1):
            total = 0
            for root in range(1, nodes + 1):
                left = root - 1
                right = nodes - root
                total += dp[left] * dp[right]
            dp[nodes] = total
        
        return dp[n]
```

---

## 🔹 Worked Example: `n = 3`

### Step 1: Initialize

```
dp[0] = 1
dp[1] = 1
```

---

### Step 2: Compute `dp[2]`

| Root | Left Nodes | Right Nodes | Count |
| --- | --- | --- | --- |
| 1 | 0 | 1 | 1 × 1 = 1 |
| 2 | 1 | 0 | 1 × 1 = 1 |

```
dp[2] = 2
```

---

### Step 3: Compute `dp[3]`

| Root | Left Nodes | Right Nodes | Count |
| --- | --- | --- | --- |
| 1 | 0 | 2 | 1 × 2 = 2 |
| 2 | 1 | 1 | 1 × 1 = 1 |
| 3 | 2 | 0 | 2 × 1 = 2 |

```
dp[3] = 5
```

---

## 🔹 Conceptual Illustration (Tree Structures for n = 3)

![Image](https://assets.leetcode.com/uploads/2021/01/18/uniquebstn3.jpg)

![Image](https://media.geeksforgeeks.org/wp-content/uploads/20250904151404252799/bst2.webp)

![Image](https://ds055uzetaobb.cloudfront.net/brioche/uploads/TJiRcSyFDm-binarytrees.png?width=1200)

Each distinct shape corresponds to a valid BST configuration.  
The count of such shapes is exactly the Catalan number `C₃ = 5`.

---

## 🔹 Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n²)** |
| Space Complexity | **O(n)** |

---

## 🔹 Interview Takeaways

* This problem is a **canonical DP + Catalan number** question.
* The crucial insight is **fixing the root and multiplying independent subtree counts**.
* Appears frequently as a foundation for harder BST construction problems (e.g., LeetCode 95).

If you want, I can also:

* Derive the **direct Catalan formula**
* Show a **recursive + memoization** version
* Connect this to **LeetCode 95 (Generate BSTs)** structurally