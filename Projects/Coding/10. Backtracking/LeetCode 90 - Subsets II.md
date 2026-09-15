---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 90: Subsets II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 90: Subsets II

## LeetCode 90 — Subsets II

---

### Problem Statement

Given an integer array `nums` that **may contain duplicates**, return **all possible subsets (the power set)**.

The solution set **must not contain duplicate subsets**.  
Return the solution in **any order**.

**Constraints**

* `1 ≤ nums.length ≤ 10`
* `-10 ≤ nums[i] ≤ 10`

**Example**

```text
Input: nums = [1,2,2]
Output:
[
  [],
  [1],
  [1,2],
  [1,2,2],
  [2],
  [2,2]
]
```

---

## Key Observations

1. This is a **subset (power set)** problem → each element is either chosen or not.
2. **Duplicates in input** cause duplicate subsets if handled naively.
3. To eliminate duplicates:

   * **Sort the array**
   * Skip duplicates **at the same recursion level**
4. Unlike permutations, **order does not matter**.
5. Every node in the backtracking tree represents a **valid subset**.

---

## Core Duplicate-Skipping Rule

```
if i > start and nums[i] == nums[i - 1]:
    continue
```

Meaning:

> “If we already considered this value at this tree level, skip it.”

---

## Approach (Backtracking)

### State Definition

* `start`: index from which we can choose elements
* `path`: current subset

### Steps

1. Add the current `path` to the result (every node is valid).
2. For each index `i` from `start` to end:

   * Skip duplicates at the same level
   * Include `nums[i]`
   * Recurse with `start = i + 1`
   * Backtrack

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            # Every path is a valid subset
            result.append(path.copy())

            for i in range(start, len(nums)):
                # Skip duplicates at the same recursion level
                if i > start and nums[i] == nums[i - 1]:
                    continue

                path.append(nums[i])
                backtrack(i + 1)
                path.pop()

        backtrack(0)
        return result
```

---

## Example Walkthrough (`nums = [1,2,2]`)

### Sorted Input

```
[1,2,2]
```

---

### Generated Subsets (in order of traversal)

1. `[]`
2. `[1]`
3. `[1,2]`
4. `[1,2,2]`
5. `[2]`
6. `[2,2]`

---

## Backtracking Tree Structure

![https://i.ytimg.com/vi/xIlOhGmfOqc/hq720.jpg?rs=AOn4CLA7cVDTa41LjfhEwaEYa4bMD2CqTQ&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com](https://i.ytimg.com/vi/xIlOhGmfOqc/hq720.jpg?rs=AOn4CLA7cVDTa41LjfhEwaEYa4bMD2CqTQ&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com)

![https://afteracademy.com/images/print-all-subsets-of-a-given-set-recursion-tree-9bbcd4be963c54c8.jpg?utm_source=chatgpt.com](https://afteracademy.com/images/print-all-subsets-of-a-given-set-recursion-tree-9bbcd4be963c54c8.jpg?utm_source=chatgpt.com)

![https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com](https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com)

### Conceptual Tree

```
[]
                --------------------------------
                |                              |
               [1]                            [2]
            ----------                       ----------
            |        |                       |        |
         [1,2]     ❌ [1,2]               [2,2]      ❌
            |
        [1,2,2]
```

### Explanation of Pruning

* The second `2` at the same level is skipped
* Vertical duplicates (deeper levels) are allowed

---

## Why This Works

### Horizontal vs Vertical Duplicates

| Direction | Allowed | Reason |
| --- | --- | --- |
| Vertical (downwards) | ✅ | Builds valid larger subsets |
| Horizontal (same level) | ❌ | Causes duplicate subsets |

---

## Complexity Analysis

* **Time Complexity:** `O(2^n)` (each subset generated once)
* **Space Complexity:** `O(n)` recursion depth

---

## Comparison with Related Problems

| Problem | Output Type | Duplicate Handling |
| --- | --- | --- |
| LC 78 | Subsets | No duplicates in input |
| LC 90 | Subsets | Skip duplicates |
| LC 40 | Combination Sum | Skip duplicates + sum constraint |
| LC 47 | Permutations | used[] + duplicate check |

---

## Pattern Recognition

LeetCode 90 exemplifies the **“Subset with Duplicates (Level Pruning)”** pattern:

* Sort input
* Skip duplicates at the same recursion level
* Add path at every node

---

### One-Line Interview Explanation

> “We sort the array and generate subsets using backtracking, skipping duplicate elements at the same recursion level to avoid repeated subsets.”

If you want next, I can:

* Show **binary include/exclude** formulation and why it fails with duplicates
* Compare **LC 77 vs LC 90** trees
* Provide a **generic subset backtracking template**