---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - greedy
---

**LeetCode 321 – Create Maximum Number**, aligned with how this problem is expected to be reasoned about in senior-level interviews.

---

# LeetCode 321 — Create Maximum Number

## Problem Statement

You are given two integer arrays `nums1` and `nums2` of lengths `m` and `n` respectively, and an integer `k`.

You must create a number of length `k` by selecting digits from `nums1` and `nums2`.  
The relative order of digits **must be preserved within each array**.

Return the **maximum possible number** (as an array of digits).

---

### Example

```text
nums1 = [3,4,6,5]
nums2 = [9,1,2,5,8,3]
k = 5

Output: [9,8,6,5,3]
```

---

## Key Observations

### 1. This is **not** a simple merge problem

* You are **choosing subsequences**, not contiguous subarrays.
* Order must be preserved, but elements may be skipped.

---

### 2. The problem decomposes into **three greedy subproblems**

1. **Pick the best subsequence of length `x` from `nums1`**
2. **Pick the best subsequence of length `k-x` from `nums2`**
3. **Merge the two subsequences lexicographically to form the largest number**

We try **all valid splits**:

```
x from max(0, k - len(nums2)) to min(k, len(nums1))
```

---

### 3. Core Greedy Insight

> The largest number is determined by **lexicographical order**, not numeric addition.

When merging:

* If digits are equal, compare the **remaining suffix**, not just the next digit.

---

## Greedy Trick #1: Maximum Subsequence of Length `k`

This is the same idea as **monotonic stack**.

### Rule

* Maintain a stack
* While:

  * stack not empty
  * current digit > stack[-1]
  * we still can drop elements
* Pop from stack
* Push current digit

---

### Helper Function: `max_subsequence(nums, k)`

```python
def max_subsequence(nums: list[int], k: int) -> list[int]:
    stack: list[int] = []
    drop = len(nums) - k

    for num in nums:
        while drop and stack and stack[-1] < num:
            stack.pop()
            drop -= 1
        stack.append(num)

    return stack[:k]
```

---

## Greedy Trick #2: Lexicographically Largest Merge

When merging two sequences:

* Always pick from the sequence that is **lexicographically larger** from current position

### Why?

Because equal digits can hide a better future suffix.

---

### Helper Function: `merge(a, b)`

```python
def merge(a: list[int], b: list[int]) -> list[int]:
    res: list[int] = []
    while a or b:
        if a > b:
            res.append(a.pop(0))
        else:
            res.append(b.pop(0))
    return res
```

> Python list comparison is lexicographical — this is intentional and correct.

---

## Full Algorithm

1. Iterate over all valid splits of `k`
2. For each split:

   * Extract max subsequence from `nums1`
   * Extract max subsequence from `nums2`
   * Merge them greedily
3. Track the best lexicographical result

---

## Python 3 Solution (Typed)

```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:

        def max_subsequence(nums: List[int], k: int) -> List[int]:
            stack: List[int] = []
            drop = len(nums) - k

            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)

            return stack[:k]

        def merge(a: List[int], b: List[int]) -> List[int]:
            res: List[int] = []
            while a or b:
                if a > b:
                    res.append(a.pop(0))
                else:
                    res.append(b.pop(0))
            return res

        best: List[int] = []

        start = max(0, k - len(nums2))
        end = min(k, len(nums1))

        for i in range(start, end + 1):
            part1 = max_subsequence(nums1, i)
            part2 = max_subsequence(nums2, k - i)
            candidate = merge(part1[:], part2[:])
            best = max(best, candidate)

        return best
```

---

## Complete Worked Example (Step-by-Step)

### Input

```
nums1 = [3,4,6,5]
nums2 = [9,1,2,5,8,3]
k = 5
```

---

### Valid splits of `k = 5`

| i (nums1) | k-i (nums2) |
| --- | --- |
| 0 | 5 |
| 1 | 4 |
| 2 | 3 |
| 3 | 2 |
| 4 | 1 |

---

### Split: i = 2

#### Step 1: max subsequence from nums1 (length 2)

```
nums1 = [3,4,6,5]
drop = 2

3 → stack = [3]
4 → pop 3 → [4]
6 → pop 4 → [6]
5 → append → [6,5]

Result: [6,5]
```

---

#### Step 2: max subsequence from nums2 (length 3)

```
nums2 = [9,1,2,5,8,3]
drop = 3

Result: [9,8,3]
```

---

#### Step 3: Merge [6,5] and [9,8,3]

| Compare | Pick | Result |
| --- | --- | --- |
| [6,5] vs [9,8,3] | 9 | [9] |
| [6,5] vs [8,3] | 8 | [9,8] |
| [6,5] vs [3] | 6 | [9,8,6] |
| [5] vs [3] | 5 | [9,8,6,5] |
| [] vs [3] | 3 | [9,8,6,5,3] |

---

### Final Answer

```
[9,8,6,5,3]
```

---

## Time & Space Complexity

| Aspect | Complexity |
| --- | --- |
| Subsequence extraction | O(m + n) |
| Merge | O(k²) worst (due to list pop) |
| Total | O(k \* (m + n)) |
| Space | O(k) |

> This is acceptable under constraints (`m, n ≤ 500`).

---

## Interview Takeaways

* This is a **compound greedy problem**
* Requires:

  * Monotonic stack
  * Lexicographical comparison
  * Careful case analysis
* Very high signal problem for senior interviews

If you want:

* Optimization discussion (deque-based merge)
* C++ version
* Visualization with trees
* Similar problems linkage (LC 402, 316, 1081)

Tell me what to cover next.