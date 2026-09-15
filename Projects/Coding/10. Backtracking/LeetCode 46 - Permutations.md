---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 46: Permutations"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 46: Permutations

## LeetCode 46 — Permutations

---

### Problem Statement

Given an array `nums` of **distinct integers**, return **all possible permutations**.  
You may return the answer in **any order**.

**Constraints**

* `1 ≤ nums.length ≤ 6`
* All integers in `nums` are **unique**

**Example**

```text
Input: nums = [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
```

---

## Key Observations

1. A permutation is an **ordering** of elements → order matters.
2. Each element must be used **exactly once** in each permutation.
3. Since all numbers are distinct, **no duplicate handling** is required.
4. The problem naturally maps to a **backtracking / DFS** tree:

   * Depth = `len(nums)`
   * Branching factor decreases as elements get used
5. Total permutations = `n!`, which defines the lower bound of runtime.

---

## Primary Approach — Backtracking (DFS)

### Idea

* Build permutations incrementally.
* Track which elements are already used.
* When the path length equals `n`, record the permutation.

---

### Python 3 Code (with Typing)

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []
        used = [False] * len(nums)

        def backtrack(path: List[int]) -> None:
            if len(path) == len(nums):
                result.append(path.copy())
                return

            for i in range(len(nums)):
                if used[i]:
                    continue

                used[i] = True
                path.append(nums[i])
                backtrack(path)
                path.pop()
                used[i] = False

        backtrack([])
        return result
```

---

## Example Explanation (`nums = [1,2,3]`)

### Step-by-step

1. Start with `[]`
2. Choose `1` → `[1]`

   * Choose `2` → `[1,2]`

     * Choose `3` → `[1,2,3]` ✓
   * Backtrack, choose `3` → `[1,3]`

     * Choose `2` → `[1,3,2]` ✓
3. Choose `2` → `[2]`

   * Choose `1` → `[2,1]`

     * Choose `3` → `[2,1,3]` ✓
   * Choose `3` → `[2,3]`

     * Choose `1` → `[2,3,1]` ✓
4. Choose `3` → `[3]`

   * Choose `1` → `[3,1]`

     * Choose `2` → `[3,1,2]` ✓
   * Choose `2` → `[3,2]`

     * Choose `1` → `[3,2,1]` ✓

---

## Backtracking Tree Illustration

![https://i.ytimg.com/vi/jspmoGnvDIg/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/jspmoGnvDIg/maxresdefault.jpg?utm_source=chatgpt.com)

![https://rangerway.com/images/algorithm/permutation-tree.png?utm_source=chatgpt.com](https://rangerway.com/images/algorithm/permutation-tree.png?utm_source=chatgpt.com)

![https://i.ytimg.com/vi/Jh6C7sU3hEw/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/Jh6C7sU3hEw/maxresdefault.jpg?utm_source=chatgpt.com)

### Conceptual Tree

```
[]
            ----------------------------------------
            |                 |                  |
           [1]               [2]                [3]
        ---------         ---------          ---------
        |       |         |       |          |       |
     [1,2]   [1,3]     [2,1]   [2,3]      [3,1]   [3,2]
        |       |         |       |          |       |
   [1,2,3] [1,3,2]   [2,1,3] [2,3,1]   [3,1,2] [3,2,1]
```

---

## All Solution Approaches

### 1. Backtracking with `used[]` (Most Common)

* Shown above
* Easy to understand
* Time: `O(n × n!)`

---

### 2. Backtracking by In-Place Swapping

#### Idea

* Fix one position at a time
* Swap the current element with every possible choice

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = []

        def backtrack(start: int) -> None:
            if start == len(nums):
                result.append(nums.copy())
                return

            for i in range(start, len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start], nums[i] = nums[i], nums[start]

        backtrack(0)
        return result
```

**Pros**

* No extra `used[]` array
* Space efficient

**Cons**

* Modifies input array (must swap back)

---

### 3. Iterative (Insertion-Based)

#### Idea

* Build permutations by inserting each new number into all positions

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result: List[List[int]] = [[]]

        for num in nums:
            new_result = []
            for perm in result:
                for i in range(len(perm) + 1):
                    new_result.append(perm[:i] + [num] + perm[i:])
            result = new_result

        return result
```

**Pros**

* No recursion
* Clean functional approach

**Cons**

* Higher memory usage
* Less intuitive in interviews

---

### 4. Python Built-in (Not Interview-Friendly)

```python
import itertools
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(map(list, itertools.permutations(nums)))
```

**Note:** Usually **not acceptable** in interviews unless explicitly allowed.

---

## Complexity Analysis

| Approach | Time | Space |
| --- | --- | --- |
| Backtracking (used[]) | `O(n × n!)` | `O(n)` |
| In-place swap | `O(n × n!)` | `O(1)` |
| Iterative | `O(n × n!)` | `O(n × n!)` |

---

## Pattern Recognition

LeetCode 46 is the **canonical permutation problem** and forms the base for:

* LeetCode 47 (Permutations II – duplicates)
* LeetCode 31 (Next Permutation)
* LeetCode 784 (Letter Case Permutation)

---

### One-Line Interview Summary

> “We generate permutations by fixing one element at each position using backtracking, marking elements as used, and exploring all `n!` possible orderings.”

If you want next, I can:

* Contrast **LC 46 vs LC 47** trees explicitly
* Show how this evolves into **next permutation**
* Provide a **generic permutation backtracking template**