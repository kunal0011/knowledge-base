---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 78: Subsets"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 78: Subsets

## LeetCode 78 — Subsets

---

### Problem Statement

Given an integer array `nums` of **unique elements**, return **all possible subsets** (the power set).

The solution set must **not contain duplicate subsets**.  
You may return the answer in **any order**.

**Constraints**

* `1 ≤ nums.length ≤ 10`
* All elements in `nums` are **distinct**

**Example**

```text
Input: nums = [1,2,3]
Output:
[
  [],
  [1],
  [2],
  [3],
  [1,2],
  [1,3],
  [2,3],
  [1,2,3]
]
```

---

## Key Observations

1. Each element has **two choices**: include it or exclude it.
2. Order inside a subset does **not matter**.
3. Total number of subsets = `2^n`.
4. This problem is the **foundation of all combination problems**.
5. Backtracking works by **adding current state at every node**, not only at leaves.

---

## Approach 1 — Backtracking (DFS)

### Idea

* Start with an empty subset.
* At each index:

  * Add the current subset to the result.
  * Try adding each remaining element and recurse.

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            # Every node is a valid subset
            result.append(path.copy())

            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()

        backtrack(0)
        return result
```

---

## Example Walkthrough (`nums = [1,2,3]`)

Traversal order:

1. `[]`
2. `[1]`
3. `[1,2]`
4. `[1,2,3]`
5. `[1,3]`
6. `[2]`
7. `[2,3]`
8. `[3]`

---

## Backtracking Tree Structure

![https://i.ytimg.com/vi/VdnvmfzA1pw/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/VdnvmfzA1pw/maxresdefault.jpg?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/uploads/20230911132238/print-all-subsets.png?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20230911132238/print-all-subsets.png?utm_source=chatgpt.com)

![https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com](https://adeveloperdiary.com/assets/img/subset.jpg?utm_source=chatgpt.com)

### Conceptual Tree

```
[]
              --------------------------------
              |              |              |
            [1]            [2]            [3]
           --------        --------         |
           |      |        |      |         |
        [1,2]   [1,3]   [2,3]   (end)      (end)
           |
        [1,2,3]
```

### Key Insight

* Every node is added to the result.
* Depth-first traversal ensures all subsets are covered.

---

## Approach 2 — Binary Decision (Include / Exclude)

### Idea

At each index:

* Either include the element
* Or skip it

---

### Python Code

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []

        def backtrack(index: int, path: List[int]) -> None:
            if index == len(nums):
                result.append(path.copy())
                return

            # Exclude
            backtrack(index + 1, path)

            # Include
            path.append(nums[index])
            backtrack(index + 1, path)
            path.pop()

        backtrack(0, [])
        return result
```

---

## Binary Tree View

```
[]
                 /        \
              skip 1      take 1
               /              \
           skip 2            take 2
           /    \             /    \
        skip3  take3       skip3  take3
```

Leaves = final subsets.

---

## Approach 3 — Bitmasking

### Idea

* Each subset corresponds to a binary number of length `n`.

---

### Python Code

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []
        n = len(nums)

        for mask in range(1 << n):
            subset: List[int] = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
            result.append(subset)

        return result
```

---

## Complexity Analysis

| Approach | Time | Space |
| --- | --- | --- |
| Backtracking | `O(n × 2^n)` | `O(n)` |
| Binary recursion | `O(n × 2^n)` | `O(n)` |
| Bitmasking | `O(n × 2^n)` | `O(1)` |

---

## Pattern Recognition

LeetCode 78 represents the **Power Set / Subset Pattern**, which extends to:

* LeetCode 90 (Subsets II – duplicates)
* LeetCode 77 (Combinations)
* LeetCode 39 / 40 (Combination Sum)

---

### One-Line Interview Explanation

> “We generate all subsets by exploring each element with include/exclude decisions, producing `2^n` subsets.”

If you want next:

* Subsets **with duplicates (LC 90)** tree comparison
* Relationship between **subsets and combinations**
* A unified **backtracking decision tree template**