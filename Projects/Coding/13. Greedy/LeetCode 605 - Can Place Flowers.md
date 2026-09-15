---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 605: Can Place Flowers"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 605: Can Place Flowers

**LeetCode 605 – Can Place Flowers**, covering the problem statement, greedy insight, implementation details, and a fully worked example with step-by-step processing.

---

## 1. Problem Statement

You are given an integer array `flowerbed` containing `0`s and `1`s:

* `1` → a flower is already planted
* `0` → empty plot

You are also given an integer `n`, representing the number of new flowers to plant.

**Rules:**

* Flowers **cannot be planted in adjacent plots**.
* Return `True` if `n` new flowers can be planted following the rule, otherwise return `False`.

---

### Example

```text
Input:
flowerbed = [1,0,0,0,1], n = 1

Output:
True
```

---

## 2. Key Observation

To plant a flower at index `i`:

```
flowerbed[i] == 0
AND
(left plot is empty or does not exist)
AND
(right plot is empty or does not exist)
```

Formally:

```
flowerbed[i] == 0
AND
(i == 0 OR flowerbed[i-1] == 0)
AND
(i == len(flowerbed)-1 OR flowerbed[i+1] == 0)
```

---

## 3. Why a Greedy Approach Works

This problem is **locally optimal → globally optimal**, which is the hallmark of a greedy solution.

### Greedy Strategy

* Traverse the flowerbed **left to right**
* Whenever you **can** plant a flower safely, **plant it immediately**
* Decrease `n`
* Modify the array in place to reflect the planting

### Why this is correct

* Planting earlier **never blocks a better future option**
* Delaying a valid placement can only reduce remaining space
* No backtracking or DP is needed

---

## 4. Greedy Solution Tricks (Interview Notes)

1. **In-place modification**

   * Once you plant a flower, mark it as `1`
   * Prevents accidentally planting adjacent flowers later
2. **Early exit**

   * As soon as `n == 0`, return `True`
3. **Boundary handling**

   * Treat missing neighbors as `0`
   * Avoid special casing by using index checks

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        length = len(flowerbed)

        for i in range(length):
            if flowerbed[i] == 0:
                left_empty = (i == 0) or (flowerbed[i - 1] == 0)
                right_empty = (i == length - 1) or (flowerbed[i + 1] == 0)

                if left_empty and right_empty:
                    flowerbed[i] = 1
                    n -= 1

                    if n == 0:
                        return True

        return n <= 0
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
flowerbed = [1,0,0,0,1]
n = 1
```

---

### Step-by-Step Processing

| Index | Flowerbed State | Check | Action | n |
| --- | --- | --- | --- | --- |
| 0 | [1,0,0,0,1] | Already 1 | Skip | 1 |
| 1 | [1,0,0,0,1] | Left=1 ❌ | Skip | 1 |
| 2 | [1,0,0,0,1] | Left=0, Right=0 ✅ | Plant | 0 |
| 3 | [1,0,1,0,1] | n == 0 | Stop | 0 |

---

### Final State

```
[1,0,1,0,1]
```

### Output

```
True
```

---

## 7. Edge Case Walkthrough

### Case 1: Single plot

```
flowerbed = [0], n = 1
```

* Left and right both treated as empty
* Can plant → `True`

---

### Case 2: No space

```
flowerbed = [1,0,1,0,1], n = 1
```

* No index satisfies adjacency rule
* Output → `False`

---

## 8. Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** (in-place) |

---

## 9. Summary (What Interviewers Look For)

* Correct adjacency condition
* Greedy justification
* Clean boundary handling
* Early termination optimization

If you want, I can also:

* Convert this into a **mathematical counting approach**
* Show **why DP is unnecessary**
* Provide a **one-pass formula-based solution**
* Compare with similar greedy problems (e.g., Jump Game)

Just tell me.