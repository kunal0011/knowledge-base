---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 769: Max Chunks To Make Sorted"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 769: Max Chunks To Make Sorted

Below is a **complete, structured explanation of LeetCode 769 — *Max Chunks To Make Sorted***, aligned with how this problem is typically solved in interviews using **greedy reasoning**.

---

## 1. Problem Statement

You are given an integer array `arr` of length `n` that is a **permutation of `[0, 1, 2, ..., n-1]`**.

You want to split the array into some number of **chunks** (contiguous subarrays) such that:

* After **sorting each chunk individually**
* And **concatenating them in the same order**

The final array becomes **fully sorted**.

### Goal

Return the **maximum number of chunks** you can split the array into.

---

## 2. Key Observations

### Observation 1: What does “sorted after chunk sorting” really mean?

If the array were already sorted:

```
[0, 1, 2, 3, 4]
```

You could split it into `n` chunks of size 1.

But in general, for a chunk ending at index `i`:

* After sorting that chunk,
* All elements in that chunk **must belong to indices ≤ i** in the final sorted array.

---

### Observation 2: Maximum value determines chunk validity

At index `i`, look at:

* `max_so_far` = maximum value in `arr[0..i]`

If:

```
max_so_far == i
```

then:

* All values seen so far are within `[0, i]`
* They can be rearranged to exactly fill positions `0..i`
* Hence, we can safely **end a chunk here**

---

### Why this works (critical insight)

Because:

* The array is a **permutation**
* No duplicates
* Values uniquely map to sorted positions

So when `max(arr[0..i]) == i`, everything needed for positions `0..i` is already present.

---

## 3. Greedy Strategy (Core Trick)

### Greedy Decision Rule

Iterate from left to right:

1. Track `max_so_far`
2. Whenever `max_so_far == current_index`

   * Finalize a chunk
   * Increment chunk count

This greedy choice is **optimal** because:

* Ending a chunk as early as possible gives room for more chunks later
* Delaying chunk boundaries never increases the total count

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        max_so_far = 0
        chunks = 0

        for i, value in enumerate(arr):
            max_so_far = max(max_so_far, value)
            
            # If all elements so far fit in positions 0..i
            if max_so_far == i:
                chunks += 1

        return chunks
```

---

## 5. Complete Worked Example (Step-by-Step)

### Example Input

```
arr = [1, 0, 2, 3, 4]
```

---

### Initialization

```
max_so_far = 0
chunks = 0
```

---

### Step-by-Step Processing

| Index `i` | Value `arr[i]` | max\_so\_far | Condition `max_so_far == i` | Action |
| --- | --- | --- | --- | --- |
| 0 | 1 | max(0, 1) = 1 | ❌ (1 ≠ 0) | continue |
| 1 | 0 | max(1, 0) = 1 | ✅ (1 == 1) | chunk++ |
| 2 | 2 | max(1, 2) = 2 | ✅ (2 == 2) | chunk++ |
| 3 | 3 | max(2, 3) = 3 | ✅ (3 == 3) | chunk++ |
| 4 | 4 | max(3, 4) = 4 | ✅ (4 == 4) | chunk++ |

---

### Chunk Formation Visualization

```
[1, 0] | [2] | [3] | [4]
```

* Sort each chunk:

```
[0, 1] | [2] | [3] | [4]
```

* Concatenate:

```
[0, 1, 2, 3, 4]
```

---

### Final Answer

```
Total Chunks = 4
```

---

## 6. Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** |

---

## 7. Common Pitfalls

1. **Works only because it’s a permutation**

   * This logic fails if duplicates exist (see LeetCode 768)
2. **Do not sort subarrays explicitly**

   * Sorting is conceptual, not part of the algorithm
3. **Do not use prefix sums**

   * Maximum tracking is sufficient and optimal

---

## 8. One-Line Intuition (Interview Ready)

> A chunk can end at index `i` if the maximum element seen so far equals `i`, meaning all required values for positions `0..i` are already present.

---

If you want, I can also:

* Contrast this with **LeetCode 768**
* Provide a **proof of correctness**
* Show a **counterexample where greedy fails if conditions change**