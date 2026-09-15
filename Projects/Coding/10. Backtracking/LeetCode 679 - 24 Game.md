---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 679: 24 Game"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 679: 24 Game

## LeetCode 679 — 24 Game

---

### Problem Statement

You are given an integer array `cards` of length **4**. Each card has a value from `1` to `9`.

You can use the operations:

* `+`, `-`, `*`, `/`

**Rules**

* You must use **all four numbers exactly once**
* You may use parentheses
* Division is **real division**, not integer division

Return `True` if you can get the value **24**, otherwise return `False`.

---

### Key Observations (Critical)

1. This is **not** a typical permutation problem — it is a **state-reduction backtracking** problem.
2. At each step:

   * Pick **any two numbers**
   * Replace them with **one result**
3. The problem size reduces:

   ```
   4 numbers → 3 → 2 → 1
   ```
4. Order of picking numbers **matters for `-` and `/`**.
5. Floating-point precision issues require comparing results with a **small epsilon**.
6. Branching factor is high, but depth is **fixed (3 levels)** → brute-force backtracking is feasible.

---

## Core Backtracking Idea

Instead of building a path, we:

* Repeatedly **merge two numbers into one**
* Recurse on the smaller list
* Stop when only one number remains

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        EPS = 1e-6

        def backtrack(nums: List[float]) -> bool:
            if len(nums) == 1:
                return abs(nums[0] - 24) < EPS

            for i in range(len(nums)):
                for j in range(len(nums)):
                    if i == j:
                        continue

                    remaining = []
                    for k in range(len(nums)):
                        if k != i and k != j:
                            remaining.append(nums[k])

                    a, b = nums[i], nums[j]

                    for val in (a + b, a - b, a * b):
                        if backtrack(remaining + [val]):
                            return True

                    if abs(b) > EPS:
                        if backtrack(remaining + [a / b]):
                            return True

            return False

        return backtrack([float(x) for x in cards])
```

---

## Example Explanation

### Input

```
cards = [4, 1, 8, 7]
```

One valid sequence:

```
(8 - 4) = 4
(7 - 1) = 6
4 * 6 = 24
```

Return:

```
True
```

---

## Complete Conceptual Backtracking Tree

**(Navigation View — State Reduction Tree)**

> Each node represents the **current list of numbers**.  
> Each edge represents **choosing two numbers and applying an operation**.

---

### Level 0 (4 numbers)

```
[4, 1, 8, 7]
```

---

### Level 1 (Pick any two → reduce to 3 numbers)

Examples (not exhaustive):

```python
[ (4,1)->5 , 8, 7 ]
[ (4,1)->3 , 8, 7 ]
[ (4,1)->4 , 8, 7 ]
[ (8,4)->2 , 1, 7 ]
[ (8,7)->56, 4, 1 ]
[ (7,1)->6 , 4, 8 ]
...
```

Each `(a,b)->result` comes from:

```
a + b
a - b
b - a
a * b
a / b
b / a   (if denominator ≠ 0)
```

---

### Level 2 (3 numbers → 2 numbers)

Example branch:

```
[4, 8, 6]
```

Possible reductions:

```python
(4,8)->12 → [12,6]
(8,4)->4  → [4,6]
(8,6)->2  → [2,4]
(6,4)->24 → [24]
...
```

---

### Level 3 (2 numbers → 1 number)

```
[4, 6]
```

Operations:

```
4 + 6 = 10
4 - 6 = -2
6 - 4 = 2
4 * 6 = 24  ✅
4 / 6 ≈ 0.66
6 / 4 = 1.5
```

---

### Level 4 (Leaf Node)

```
[24]  → SUCCESS
```

---

## Tree Characteristics (Important Insight)

| Aspect | Description |
| --- | --- |
| Tree type | **State-reduction tree** |
| Depth | Fixed = 3 |
| Branching | High (pair selection × operations) |
| Node value | List of remaining numbers |
| Goal | Reduce to `[24]` |

This is **not** a “build-a-path” tree (like permutations), but a **merge-and-reduce** tree.

---

## Why This Problem Is Hard

* No natural ordering
* Non-commutative operations
* Floating-point precision
* High branching but shallow depth

---

## One-Line Interview Explanation

> “We repeatedly pick two numbers, apply all valid operations, and recurse on the reduced list until only one number remains, checking if it equals 24.”