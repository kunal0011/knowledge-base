---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 321: Create Maximum Number

Below is a **complete, interview-ready breakdown** of **LeetCode 321 – Create Maximum Number**, structured exactly as you requested.

---

## LeetCode 321 — Create Maximum Number

### Problem Statement

You are given two integer arrays `nums1` and `nums2`, of lengths `m` and `n` respectively, and an integer `k`.

You must create a number of length `k` by selecting digits from `nums1` and `nums2`, **preserving the relative order within each array**, such that the resulting number is **as large as possible**.

Return the resulting number as an array of digits.

---

### Key Observations

1. **You must preserve relative order**

   * You cannot reorder digits within `nums1` or `nums2`.
2. **Split the problem**

   * Decide how many digits to take from `nums1` (`i`)
   * Take `k - i` digits from `nums2`
   * Try **all valid splits**
3. **Two core subproblems**

   * **Pick the maximum subsequence** of length `x` from one array
   * **Merge two subsequences** to form the lexicographically largest number
4. **Greedy + Two Pointer**

   * Greedy selection ensures maximal prefix
   * Two pointers help merge while comparing remaining suffixes

---

### Core Techniques Used

| Technique | Purpose |
| --- | --- |
| Monotonic Stack | Pick max subsequence |
| Two Pointers | Merge sequences |
| Lexicographic Comparison | Decide next digit |
| Greedy Enumeration | Try all splits |

---

## Step 1: Pick Maximum Subsequence (Greedy Stack)

**Idea**  
Use a stack to remove smaller digits when a larger digit appears and removals are allowed.

```python
def max_subsequence(nums: list[int], k: int) -> list[int]:
    drop = len(nums) - k
    stack = []

    for num in nums:
        while drop and stack and stack[-1] < num:
            stack.pop()
            drop -= 1
        stack.append(num)

    return stack[:k]
```

---

## Step 2: Merge Two Subsequences (Two Pointer Technique)

**Key Insight**  
When digits are equal, compare **remaining suffixes**, not just current digits.

```python
def merge(a: list[int], b: list[int]) -> list[int]:
    res = []
    i = j = 0

    while i < len(a) or j < len(b):
        if a[i:] > b[j:]:
            res.append(a[i])
            i += 1
        else:
            res.append(b[j])
            j += 1

    return res
```

---

## Final Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:

        def max_subsequence(nums: List[int], k: int) -> List[int]:
            drop = len(nums) - k
            stack = []

            for num in nums:
                while drop and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)

            return stack[:k]

        def merge(a: List[int], b: List[int]) -> List[int]:
            res = []
            while a or b:
                if a > b:
                    res.append(a.pop(0))
                else:
                    res.append(b.pop(0))
            return res

        best = []

        for i in range(max(0, k - len(nums2)), min(k, len(nums1)) + 1):
            part1 = max_subsequence(nums1, i)
            part2 = max_subsequence(nums2, k - i)
            candidate = merge(part1[:], part2[:])
            best = max(best, candidate)

        return best
```

---

## Worked Out Example

### Input

```
nums1 = [3,4,6,5]
nums2 = [9,1,2,5,8,3]
k = 5
```

---

### Try all valid splits

#### Case 1: Take 2 from `nums1`, 3 from `nums2`

**Max subsequence from nums1 (2 digits)**  
`[6,5]`

**Max subsequence from nums2 (3 digits)**  
`[9,8,3]`

---

### Merge (Two Pointers)

| a | b | Pick |
| --- | --- | --- |
| [6,5] | [9,8,3] | 9 |
| [6,5] | [8,3] | 8 |
| [6,5] | [3] | 6 |
| [5] | [3] | 5 |
| [] | [3] | 3 |

Result:

```
[9,8,6,5,3]
```

---

### Final Output

```
[9,8,6,5,3]
```

---

## Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(k * (m + n))` |
| Space | `O(k)` |

---

## Why This Solution Is Optimal

* Exhausts all valid splits
* Greedy ensures maximal digit selection
* Two pointer merge guarantees lexicographically maximum result
* Fully compliant with order constraints

---

If you want:

* Dry-run visualization
* Optimization to avoid `pop(0)`
* Java version
* Interview explanation cheat-sheet

Tell me how deep you want to go.