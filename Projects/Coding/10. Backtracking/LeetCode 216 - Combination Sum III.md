---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 216: Combination Sum III"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 216: Combination Sum III

## LeetCode 216 — Combination Sum III

---

### Problem Statement

Find **all valid combinations** of `k` numbers that sum up to `n`, such that:

* Only numbers from **1 to 9** are used
* Each number is used **at most once**
* Exactly **k numbers** are chosen
* The sum of the chosen numbers equals `n`

Return the list of all possible combinations.  
The order of combinations does **not** matter.

**Constraints**

* `2 ≤ k ≤ 9`
* `1 ≤ n ≤ 60`

**Example**

```text
Input: k = 3, n = 7
Output: [[1,2,4]]
```

---

## Key Observations

1. The candidate set is **fixed**: numbers `1..9`.
2. Each number can be used **only once** → combinations, not permutations.
3. We must satisfy **two constraints simultaneously**:

   * Exact length = `k`
   * Exact sum = `n`
4. Because numbers are strictly increasing, **duplicates cannot occur**.
5. Strong pruning is possible using:

   * Remaining count
   * Remaining sum bounds

---

## Approach (Backtracking with Pruning)

### State Definition

* `start`: next number allowed to pick
* `path`: current combination
* `remaining`: remaining sum to reach `n`

### Decisions

* Pick a number `i` from `[start .. 9]`
* Recurse with:

  * `start = i + 1`
  * `remaining = remaining - i`

### Base Conditions

* If `remaining == 0` **and** `len(path) == k` → valid solution
* If `remaining < 0` or `len(path) > k` → prune

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            # Valid combination
            if remaining == 0 and len(path) == k:
                result.append(path.copy())
                return

            # Pruning
            if remaining < 0 or len(path) > k:
                return

            for i in range(start, 10):
                # Prune if remaining numbers are insufficient
                if len(path) + (10 - i) < k:
                    break

                path.append(i)
                backtrack(i + 1, remaining - i)
                path.pop()

        backtrack(1, n)
        return result
```

---

## Example Walkthrough (`k = 3`, `n = 9`)

### Goal

Pick **3 numbers** from `1..9` that sum to `9`.

### Valid Solution

```
[1,2,6]
[1,3,5]
[2,3,4]
```

### Exploration

* Start with `1`

  * Try `2`

    * Try `6` → sum = 9 ✓
    * Try `7` → sum > 9 ✗
  * Try `3`

    * Try `5` → sum = 9 ✓
* Start with `2`

  * Try `3`

    * Try `4` → sum = 9 ✓
* Start with `3` → cannot form valid size-3 sum → pruned

---

## Backtracking Tree Structure

![https://codeanddebug.in/blog/wp-content/uploads/2025/07/combination-sum-3-featured-image.png?utm_source=chatgpt.com](https://codeanddebug.in/blog/wp-content/uploads/2025/07/combination-sum-3-featured-image.png?utm_source=chatgpt.com)

![https://www.interviewbit.com/blog/wp-content/uploads/2021/11/Recursion-Tree-1-1024x640.png?utm_source=chatgpt.com](https://www.interviewbit.com/blog/wp-content/uploads/2021/11/Recursion-Tree-1-1024x640.png?utm_source=chatgpt.com)

![https://miro.medium.com/0%2Ap4GiBEhJkO_zARlv.png?utm_source=chatgpt.com](https://miro.medium.com/0%2Ap4GiBEhJkO_zARlv.png?utm_source=chatgpt.com)

### Conceptual Tree (Simplified)

```
[]
                 ------------------------------------------------
                 |                |                |
                1                2                3
          -------------       -----------        ----------
          |           |       |         |
         2             3      3          4
      --------       -------     |
      |      |       |     |     |
     6        7     5       6    4
  [1,2,6] ✓   ✗   [1,3,5] ✓ ✗  [2,3,4] ✓
```

---

## Why Pruning Is Effective

1. Numbers are **positive and increasing**
2. Once `remaining < 0`, deeper recursion only worsens it
3. If remaining slots `< needed elements`, branch cannot succeed

---

## Complexity Analysis

* **Time Complexity:** `O(C(9, k))`
* **Space Complexity:** `O(k)` recursion depth

---

## Pattern Recognition

LeetCode 216 is a **fixed-candidate, fixed-length combination** problem:

* Similar to LC 77 (Combinations)
* Similar to LC 40 (Combination Sum II)
* Strong example of **multi-constraint backtracking**

---

### One-Line Interview Explanation

> “We backtrack over numbers 1 to 9, selecting increasing values, and prune branches when the sum or length constraints are violated.”

If you want next, I can:

* Show **tight mathematical pruning using min/max sum bounds**
* Compare **LC 39 vs 40 vs 216** trees
* Convert this into a **generic k-sum backtracking template**