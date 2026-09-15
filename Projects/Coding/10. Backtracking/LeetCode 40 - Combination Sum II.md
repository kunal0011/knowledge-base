---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 40: Combination Sum II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 40: Combination Sum II

## LeetCode 40 — Combination Sum II

---

### Problem Statement

Given a collection of **candidate numbers** `candidates` (which may contain duplicates) and a target number `target`, return **all unique combinations** where the chosen numbers sum to `target`.

**Rules**

* Each number in `candidates` may be used **at most once** in a combination.
* The solution set must **not contain duplicate combinations**.
* You may return the answer in **any order**.

**Constraints**

* `1 ≤ candidates.length ≤ 100`
* `1 ≤ candidates[i] ≤ 50`
* `1 ≤ target ≤ 30`

**Example**

```text
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output:
[
  [1,1,6],
  [1,2,5],
  [1,7],
  [2,6]
]
```

---

## Key Observations (Very Important)

1. **Duplicates exist** in the input → naive backtracking will generate duplicate combinations.
2. Each element can be used **only once**, unlike LeetCode 39.
3. To handle duplicates correctly:

   * **Sort the array first**
   * Skip duplicates **at the same recursion level**
4. This is a **combination** problem:

   * Order does **not** matter
   * Enforce increasing index (`start`) to avoid reusing elements

---

## Core Duplicate-Skipping Rule

After sorting:

```
if i > start and candidates[i] == candidates[i - 1]:
    continue
```

Meaning:

> “At the same tree level, if we already tried this value once, skip it.”

---

## Approach (Backtracking + Pruning)

### State Definition

* `start`: next index to consider
* `path`: current combination
* `remaining`: remaining sum to reach `target`

### Decisions

* Choose `candidates[i]`
* Move to `i + 1` (because each number can be used once)

### Pruning

* If `remaining < 0` → stop exploring
* If `remaining == 0` → record solution

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                result.append(path.copy())
                return

            for i in range(start, len(candidates)):
                # Skip duplicates at the same tree level
                if i > start and candidates[i] == candidates[i - 1]:
                    continue

                if candidates[i] > remaining:
                    break  # pruning (sorted array)

                path.append(candidates[i])
                backtrack(i + 1, remaining - candidates[i])
                path.pop()

        backtrack(0, target)
        return result
```

---

## Example Walkthrough

### Input

```
candidates = [1,1,2,5,6,7,10]
target = 8
```

---

### Valid Paths

1. `1 → 1 → 6` → `[1,1,6]`
2. `1 → 2 → 5` → `[1,2,5]`
3. `1 → 7` → `[1,7]`
4. `2 → 6` → `[2,6]`

---

## Backtracking Tree Structure (Navigation View)

![https://i.ytimg.com/vi/IS5zvcszrRw/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/IS5zvcszrRw/maxresdefault.jpg?utm_source=chatgpt.com)

![https://assets.algo.monster/liteProblems/comb_sum_ii.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/IfZZ6nicu5mVUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw4uyE8vMfRwM4r0dorUDckO8C9JNC0yCnP3SNVNzvFzr0r3D0iy9M4wLTVIMU3JCHVLUysGAHOqJeY?utm_source=chatgpt.com)

![https://miro.medium.com/v2/resize%3Afit%3A1200/1%2Acs9X_sBCwXL7hbkP4w_TqQ.png?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A1200/1%2Acs9X_sBCwXL7hbkP4w_TqQ.png?utm_source=chatgpt.com)

### Conceptual Tree (Simplified)

```
[]
                ------------------------------------------------
                |                |               |            |
               1                2               5            6
           -----------         -------         -------        |
           |         |         |     |         |     |      [6,?]
          1           2        5     6         6     7
          |           |        |     |         |
       [1,1,6]     [1,2,5]   ❌   [2,6]       ❌
```