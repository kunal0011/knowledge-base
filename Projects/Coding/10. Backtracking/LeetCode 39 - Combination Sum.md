---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 39: Combination Sum"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 39: Combination Sum

## LeetCode 39 — Combination Sum

---

### Problem Statement

Given an array of **distinct integers** `candidates` and an integer `target`, return **all unique combinations** of `candidates` where the chosen numbers sum to `target`.

**Rules**

* Each number in `candidates` may be chosen **unlimited times**.
* The solution set must not contain duplicate combinations.
* You may return the answer in **any order**.

**Constraints**

* `1 ≤ candidates.length ≤ 30`
* `1 ≤ candidates[i] ≤ 200`
* `1 ≤ target ≤ 500`
* All elements in `candidates` are **distinct**

**Example**

```text
Input: candidates = [2,3,6,7], target = 7
Output:
[
  [2,2,3],
  [7]
]
```

---

## Key Observations

1. This is a **combination** problem, not a permutation:

   * Order does **not** matter → `[2,2,3]` is the same as `[3,2,2]`.
2. Each candidate can be used **multiple times**.
3. To avoid duplicates:

   * Maintain a **start index** so we only move forward.
4. Since numbers are positive:

   * If the current sum exceeds `target`, we can **prune** the branch.
5. This is a classic **unbounded knapsack (DFS)** style backtracking problem.

---

## Approach (Backtracking)

### State

* `start`: index from which we are allowed to choose numbers
* `path`: current combination
* `remaining`: remaining sum to reach `target`

### Choices

* Choose `candidates[i]`
* Recurse with the **same index `i`** (because reuse is allowed)

### Base Cases

* `remaining == 0` → valid combination
* `remaining < 0` → invalid path (prune)

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                result.append(path.copy())
                return

            if remaining < 0:
                return

            for i in range(start, len(candidates)):
                path.append(candidates[i])
                # reuse allowed → i (not i + 1)
                backtrack(i, remaining - candidates[i])
                path.pop()

        backtrack(0, target)
        return result
```

---

## Example Walkthrough

### Input

```
candidates = [2,3,6,7]
target = 7
```

---

### Step-by-Step Exploration

1. Choose `2` → remaining `5`

   * Choose `2` → remaining `3`

     * Choose `2` → remaining `1` ❌
     * Choose `3` → remaining `0` ✅ `[2,2,3]`
2. Choose `3` → remaining `4`

   * Choose `3` → remaining `1` ❌
3. Choose `6` → remaining `1` ❌
4. Choose `7` → remaining `0` ✅ `[7]`

---

## Backtracking Tree Structure

![https://miro.medium.com/1%2ATQ0-N3WZezH7AFNIi-FKsA.png?utm_source=chatgpt.com](https://miro.medium.com/1%2ATQ0-N3WZezH7AFNIi-FKsA.png?utm_source=chatgpt.com)

![https://www.interviewbit.com/blog/wp-content/uploads/2021/11/Recursion-Tree-1-1024x640.png?utm_source=chatgpt.com](https://www.interviewbit.com/blog/wp-content/uploads/2021/11/Recursion-Tree-1-1024x640.png?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/20250312122618748210/Recursion-Tree-for-01-KnapSack-2.webp?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20250312122618748210/Recursion-Tree-for-01-KnapSack-2.webp?utm_source=chatgpt.com)

### Conceptual Tree

```
[]
                -----------------------------------------------
                |                |              |             |
               2                3              6             7
           -----------        --------        -------        [7]
           |         |        |      |
          2           3       3      6
          |           |       |
          2           3       ❌
          |
          3
          |
       [2,2,3]
```

---

## Why Duplicates Are Avoided

* We **never go backwards** in indices.
* Once we start from index `i`, all future choices come from `i` onward.
* This enforces a canonical (sorted-by-index) order.

---

## Complexity Analysis

* **Time Complexity:**  
  Exponential — roughly `O(2^target)` in the worst case.
* **Space Complexity:**  
  `O(target)` recursion depth (worst-case path like `[1,1,1,...]`).

---

## Pattern Recognition

LeetCode 39 represents the **Unbounded Combination Backtracking** pattern:

* Reuse allowed
* Index-based DFS
* Sum-based pruning

This pattern appears in:

* Coin Change (DFS variants)
* Knapsack-style enumeration
* Resource allocation problems

---

### One-Line Interview Explanation

> “We use backtracking with an index pointer. At each step, we either reuse the current number or move forward, pruning when the sum exceeds the target.”

If you want, I can next:

* Contrast **LC 39 vs LC 40** trees explicitly
* Show how this relates to **unbounded knapsack DP**
* Provide a **generic reusable combination-sum template**