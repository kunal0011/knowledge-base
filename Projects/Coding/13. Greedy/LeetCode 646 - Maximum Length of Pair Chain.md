---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 646: Maximum Length of Pair Chain"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 646: Maximum Length of Pair Chain

**LeetCode 646 – Maximum Length of Pair Chain**, aligned with interview and competitive-programming expectations.

---

## LeetCode 646: Maximum Length of Pair Chain

### Problem Statement

You are given an array of pairs `pairs`, where  
`pairs[i] = [a, b]` and `a < b`.

A **pair** `[a, b]` can be followed by another pair `[c, d]` **if and only if**:

```
b < c
```

A **pair chain** is a sequence of pairs where each pair can follow the previous one.

Return the **maximum length** of such a chain.

---

### Example

```text
Input:  pairs = [[1,2],[2,3],[3,4]]
Output: 2

Explanation:
[1,2] → [3,4] is a valid chain
[1,2] → [2,3] is NOT valid because 2 < 2 is false
```

---

## Key Observation

This problem is **structurally identical** to:

> **Activity Selection / Interval Scheduling (Greedy)**

### Why?

* Each pair `[a, b]` behaves like an **interval**
* We want to select the **maximum number of non-overlapping intervals**
* Condition `b < c` is stricter than `b ≤ c`
* Greedy works when we **always pick the interval that ends earliest**

---

## Why Greedy Works (Critical Insight)

If we pick the pair with the **smallest ending value `b`**, we:

* Leave **maximum room** for future pairs
* Avoid blocking better chains later
* Ensure local optimal choice leads to global optimum

This is the same proof logic as:

* Interval Scheduling
* Activity Selection
* Non-overlapping intervals

---

## Greedy Strategy (Step-by-Step)

1. **Sort pairs by their ending value (`b`)**
2. Initialize:

   * `current_end = -∞`
   * `chain_length = 0`
3. Traverse sorted pairs:

   * If `pair.start > current_end`

     * Accept the pair
     * Update `current_end = pair.end`
     * Increment `chain_length`

---

## Python 3 Solution (Greedy)

```python
from typing import List

class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # Step 1: Sort pairs by second element (ending value)
        pairs.sort(key=lambda x: x[1])

        chain_length = 0
        current_end = float('-inf')

        # Step 2: Greedily select pairs
        for start, end in pairs:
            if start > current_end:
                chain_length += 1
                current_end = end

        return chain_length
```

---

## Complete Worked Example (All Processing Steps)

### Input

```
pairs = [[1,2],[7,8],[4,5],[2,3]]
```

---

### Step 1: Sort by ending value

```
Sorted pairs = [[1,2], [2,3], [4,5], [7,8]]
```

---

### Step 2: Initialize

```
current_end = -∞
chain_length = 0
```

---

### Step 3: Process Each Pair

| Pair | Condition (`start > current_end`) | Action Taken | New `current_end` | Chain |
| --- | --- | --- | --- | --- |
| [1,2] | 1 > -∞ → TRUE | Pick | 2 | 1 |
| [2,3] | 2 > 2 → FALSE | Skip | 2 | 1 |
| [4,5] | 4 > 2 → TRUE | Pick | 5 | 2 |
| [7,8] | 7 > 5 → TRUE | Pick | 8 | 3 |

---

### Final Answer

```
Maximum Chain Length = 3
Chain = [1,2] → [4,5] → [7,8]
```

---

## Why Sorting by Start Will Fail

If you sort by `start` instead of `end`, you may pick:

```
[1,10] → blocks all future pairs
```

Even though many short chains could exist.

Sorting by **end time minimizes blocking**.

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | `O(n log n)` (sorting) |
| Space Complexity | `O(1)` (in-place) |

---

## Comparison with DP (Why Greedy is Better Here)

| Approach | Time | Reason |
| --- | --- | --- |
| DP (LIS-style) | `O(n²)` | Checks all previous pairs |
| Greedy | `O(n log n)` | Optimal due to interval structure |

---

## Final Takeaway (Interview Ready)

* This is **interval scheduling**
* **Sort by end**
* **Pick earliest finishing pair**
* Local greedy choice leads to global optimum

If you want, I can also:

* Show **DP formulation** for comparison
* Draw a **conceptual greedy selection tree**
* Map this problem to **LIS vs Interval Scheduling pattern**

Just tell me.