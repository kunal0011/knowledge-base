---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 491: Non-decreasing Subsequences"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 491: Non-decreasing Subsequences

## LeetCode 491 — Non-decreasing Subsequences

---

### Problem Statement

Given an integer array `nums`, return **all the different non-decreasing subsequences** of the array with **length ≥ 2**.

* A subsequence is obtained by deleting some (or none) elements **without changing the order**.
* The result must **not contain duplicate subsequences**.

**Constraints**

* `1 ≤ nums.length ≤ 15`
* `-100 ≤ nums[i] ≤ 100`

**Example**

```text
Input: nums = [4,6,7,7]
Output:
[
  [4,6],
  [4,7],
  [4,6,7],
  [4,6,7,7],
  [6,7],
  [6,7,7],
  [7,7],
  [4,7,7]
]
```

---

## Key Observations (Critical for LC 491)

1. This is a **subsequence** problem, not a permutation or combination:

   * Order must be preserved.
   * Elements are chosen or skipped as we move forward.
2. The subsequence must be **non-decreasing**:

   ```
   next_element >= last_element_in_path
   ```
3. The input **contains duplicates**, but:

   * We cannot globally skip duplicates like LC 47.
   * Duplicate handling must be done **per recursion level**.
4. We use a **local set (`used`) at each depth** to avoid generating the same value twice **at the same tree level**.

---

## Core Backtracking Strategy

### State

* `start`: current index in `nums`
* `path`: current subsequence being built

### Rules

* Add `path` to result **only if length ≥ 2**
* At each recursion level:

  * Use a `set` to record values already used at this level
  * Skip values that:

    * Break non-decreasing order
    * Have already been used at this level

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            if len(path) >= 2:
                result.append(path.copy())

            used = set()  # level-based duplicate control

            for i in range(start, len(nums)):
                # Enforce non-decreasing order
                if path and nums[i] < path[-1]:
                    continue

                # Skip duplicates at the same tree level
                if nums[i] in used:
                    continue

                used.add(nums[i])
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()

        backtrack(0)
        return result
```

---

## Example Explanation (`nums = [4,6,7,7]`)

### Valid subsequences (length ≥ 2)

* Starting from `4`:

  * `[4,6]`
  * `[4,6,7]`
  * `[4,6,7,7]`
  * `[4,7]`
  * `[4,7,7]`
* Starting from `6`:

  * `[6,7]`
  * `[6,7,7]`
* Starting from `7`:

  * `[7,7]`

Duplicates such as `[4,7]` are produced **only once**, even though there are two `7`s.

---

## Backtracking Tree Structure

### (Complete Conceptual Tree — `nums = [4,6,7,7]`)

![https://i.ytimg.com/vi/C3HjKuU4xjg/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/C3HjKuU4xjg/maxresdefault.jpg?utm_source=chatgpt.com)

![https://afteracademy.com/images/longest-increasing-subsequence-recursion-tree-740cf7ef02136d4a.png?utm_source=chatgpt.com](https://afteracademy.com/images/longest-increasing-subsequence-recursion-tree-740cf7ef02136d4a.png?utm_source=chatgpt.com)

![https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com](https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com)

### Conceptual Navigation Tree

```
[]
                    ----------------------------------------------------------------
                    |                         |                       |             |
                   4                          6                       7             7 ❌
                   |                          |                       |
        -----------------------         ------------------          ---------
        |            |          |        |                |          |
       6              7          7 ❌     7                7 ❌       7
       |              |                  |                            |
   ---------        ---------          ---------                    [7,7]
   |       |        |       |          |
  7         7 ❌    7         7 ❌       7
  |                    |
 [4,6,7]          [4,7,7]
  |
 [4,6,7,7]
```

### How to Read This Tree

* **Vertical direction (downwards)**:

  * Allowed duplicates (e.g., `[7,7]`)
* **Horizontal direction (same level)**:

  * Duplicates skipped using `used` set
* ❌ indicates **pruned branches** due to:

  * Duplicate at same level
  * Breaking non-decreasing order

---

## Why Level-Based `used` Is Required (Key Insight)

Consider this level:

```
path = [4]
choices = [6,7,7]
```

If we allow both `7`s:

* `[4,7]` would be generated **twice**

Hence:

```
used = set()  # reset every recursion level
```

This is **not global duplicate elimination** (unlike LC 47).

---

## Complexity Analysis

* **Time Complexity:** `O(2^n)` (subsequence enumeration with pruning)
* **Space Complexity:** `O(n)` recursion depth (excluding output)

---

## Pattern Recognition

LeetCode 491 represents the **“Subsequence Backtracking with Level Deduplication”** pattern:

* Linear index progression
* Order-preserving
* Local duplicate control per recursion level

This pattern also appears in:

* LeetCode 90 (Subsets II)
* LeetCode 40 (Combination Sum II)
* LeetCode 47 (Permutations II — conceptually similar)

---

### One-Line Interview Explanation

> “We generate subsequences using backtracking, enforce non-decreasing order, and avoid duplicates by tracking values used at each recursion level.”

If you want next, I can:

* Compare **LC 491 vs LC 78 vs LC 90**
* Show **exact duplicate outputs** if `used` is removed
* Provide a **generic subsequence backtracking template**