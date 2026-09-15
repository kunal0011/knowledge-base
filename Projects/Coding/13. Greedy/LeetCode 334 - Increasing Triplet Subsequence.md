---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 334: Increasing Triplet Subsequence"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 334: Increasing Triplet Subsequence

**LeetCode 334 – Increasing Triplet Subsequence**, structured exactly as requested.

---

## 1. Problem Statement

**LeetCode 334 – Increasing Triplet Subsequence**

Given an integer array `nums`, return `true` if there exists a **triplet** of indices  
`i < j < k` such that:

```
nums[i] < nums[j] < nums[k]
```

Otherwise, return `false`.

### Constraints

* `1 ≤ nums.length ≤ 5 * 10^5`
* `-2^31 ≤ nums[i] ≤ 2^31 - 1`

---

## 2. Key Observation

You **do not need to find the actual triplet**.  
You only need to determine **whether such a triplet exists**.

### Critical Insight

If we can maintain:

* the **smallest possible first number**
* the **smallest possible second number greater than the first**

and later find **any number greater than both**, then an increasing triplet **must exist**.

This is fundamentally a **greedy decision problem**, not a DP or backtracking problem.

---

## 3. Why Greedy Works Here

The greedy strategy works because:

* Smaller values give **more room** for future larger values.
* Once you lock in the **best possible first and second elements**, the existence of a third larger element guarantees success.

### Greedy Invariant

At every step:

* `first` = smallest number seen so far
* `second` = smallest number greater than `first`

If we ever find a number:

```
num > second
```

→ triplet found.

---

## 4. Greedy Solution Tricks (Interview Gold)

### Trick 1: Do NOT track indices

Ordering (`i < j < k`) is guaranteed by **left-to-right traversal**.

---

### Trick 2: Always minimize `first` and `second`

Even if you already have a candidate:

* Replace it with a **better (smaller)** value when possible.

---

### Trick 3: Use `<=`, not `<`

This avoids false positives when duplicate values exist.

---

## 5. Algorithm (Step-by-Step Logic)

1. Initialize:

   * `first = +∞`
   * `second = +∞`
2. Traverse the array:

   * If `num <= first`: update `first`
   * Else if `num <= second`: update `second`
   * Else:

     * `num > second` → **triplet found** → return `True`
3. If traversal ends → return `False`

---

## 6. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float('inf')
        second = float('inf')

        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                # num > first and num > second
                return True

        return False
```

---

## 7. Complete Worked Example (Step-by-Step Processing)

### Example Input

```text
nums = [2, 1, 5, 0, 4, 6]
```

---

### Initial State

```
first  = ∞
second = ∞
```

---

### Step-by-Step Execution

| Index | num | Action Taken | first | second | Explanation |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | `num <= first` | 2 | ∞ | New smallest value |
| 1 | 1 | `num <= first` | 1 | ∞ | Better first |
| 2 | 5 | `num <= second` | 1 | 5 | Second element candidate |
| 3 | 0 | `num <= first` | 0 | 5 | Even better first |
| 4 | 4 | `num <= second` | 0 | 4 | Better second |
| 5 | 6 | `num > second` | 0 | 4 | ✅ Triplet found |

---

### Triplet Identified

```
0 < 4 < 6
```

Indices respect `i < j < k`.

---

## 8. Why This Is Optimal

### Time Complexity

* **O(n)** — single pass

### Space Complexity

* **O(1)** — constant extra space

---

## 9. Common Mistakes to Avoid

1. ❌ Trying to store all subsequences
2. ❌ Using DP (`O(n²)` will TLE)
3. ❌ Resetting `second` incorrectly
4. ❌ Using `<` instead of `<=`

---

## 10. Mental Model for Interviews

> “I’m not searching for a triplet —  
> I’m **proving its existence** by greedily minimizing the first two elements.”

---

If you want:

* a **visual greedy decision tree**
* comparison with **LIS (Longest Increasing Subsequence)**
* or an explanation of **why DP is unnecessary**

tell me and I will extend this explanation accordingly.