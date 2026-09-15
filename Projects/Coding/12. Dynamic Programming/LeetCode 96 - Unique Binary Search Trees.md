---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 96: Unique Binary Search Trees"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 96: Unique Binary Search Trees

**LeetCode 96 – Unique Binary Search Trees**, with **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 96 — Unique Binary Search Trees

### Problem Statement

Given an integer `n`, return the number of **structurally unique Binary Search Trees (BSTs)** that store values `1` to `n`.

---

## Key Insight (Why DP Works)

A BST is defined by:

* Choosing a **root**
* Recursively forming a BST from values **less than the root** (left subtree)
* Recursively forming a BST from values **greater than the root** (right subtree)

The **exact values do not matter**, only **how many nodes** go into the left and right subtrees.

---

## DP State Definition

Let:

```
dp[i] = number of unique BSTs that can be formed using i nodes
```

We want `dp[n]`.

---

## Base Case

```
dp[0] = 1   // empty tree
dp[1] = 1   // single node
```

Why `dp[0] = 1`?

* An empty subtree is considered **one valid possibility** (important for multiplication logic).

---

## State Transition

Suppose we are computing `dp[i]`.

Choose each node `k` (`1 ≤ k ≤ i`) as the **root**.

* Left subtree has `k - 1` nodes
* Right subtree has `i - k` nodes

Number of BSTs with `k` as root:

```
dp[k - 1] * dp[i - k]
```

So:

```
dp[i] = Σ (dp[k - 1] * dp[i - k])   for k = 1 to i
```

This is the **Catalan number recurrence**.

---

## DP Table Construction Order

We build the table **bottom-up**:

```
dp[0] → dp[1] → dp[2] → ... → dp[n]
```

Each `dp[i]` depends only on **smaller indices**.

---

## Example Walkthrough (n = 3)

### Step 1: Initialize

```
dp[0] = 1
dp[1] = 1
```

DP array:

```
i:   0   1   2   3
dp: [1,  1,  0,  0]
```

---

### Step 2: Compute dp[2]

Possible roots: `1`, `2`

| Root | Left nodes | Right nodes | Count |
| --- | --- | --- | --- |
| 1 | 0 | 1 | dp[0] \* dp[1] = 1 |
| 2 | 1 | 0 | dp[1] \* dp[0] = 1 |

```
dp[2] = 1 + 1 = 2
```

DP array:

```
[1, 1, 2, 0]
```

---

### Step 3: Compute dp[3]

Possible roots: `1`, `2`, `3`

| Root | Left nodes | Right nodes | Count |
| --- | --- | --- | --- |
| 1 | 0 | 2 | dp[0] \* dp[2] = 2 |
| 2 | 1 | 1 | dp[1] \* dp[1] = 1 |
| 3 | 2 | 0 | dp[2] \* dp[0] = 2 |

```
dp[3] = 2 + 1 + 2 = 5
```

DP array:

```
[1, 1, 2, 5]
```

---

## Final Answer

```
Unique BSTs for n = 3 → 5
```

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def numTrees(self, n: int) -> int:
        dp: List[int] = [0] * (n + 1)
        
        dp[0] = 1
        dp[1] = 1

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

## Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n²)** |
| Space | **O(n)** |

---

## Mental Model (Interview-Friendly)

* Choose each value as root
* Multiply combinations of left × right subtree
* Sum across all root choices
* Build answers from **smaller subproblems**

---

If you want next:

* **Visual recursion tree**
* **Relation to Catalan Numbers (math proof)**
* **Why BST property removes value dependency**
* **Memoized recursion version**

Tell me which one to continue.