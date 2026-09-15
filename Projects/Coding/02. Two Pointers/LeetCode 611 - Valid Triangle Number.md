---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 611: Valid Triangle Number"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 611: Valid Triangle Number

Below is a complete, interview-ready explanation of **LeetCode 611 – Valid Triangle Number**, structured exactly as requested.

---

## 📌 Problem Statement (LeetCode 611 – Valid Triangle Number)

You are given an integer array `nums`.  
Return the number of **triplets** `(i, j, k)` such that:

* `i < j < k`
* `nums[i]`, `nums[j]`, and `nums[k]` can form a **valid triangle**

### Triangle validity rule

For three sides `a`, `b`, `c` (assuming `a ≤ b ≤ c`):

```
a + b > c
```

---

## 🔑 Key Observation

1. **Sorting simplifies the triangle condition**

   * After sorting, for indices `i < j < k`, we have:

     ```
     nums[i] ≤ nums[j] ≤ nums[k]
     ```
   * We only need to check:

     ```
     nums[i] + nums[j] > nums[k]
     ```
2. **Monotonic behavior after sorting**

   * Fix the **largest side** `nums[k]`
   * If `nums[i] + nums[j] > nums[k]` is true for some `i`,  
     then it will also be true for all indices `i' > i` (because values increase).

This monotonicity enables a **two-pointer optimization**.

---

## 🧠 Two Pointer Technique (Core Insight)

### Strategy

1. **Sort** the array
2. Fix `k` as the **largest side** (iterate from right to left)
3. Use two pointers:

   * `i = 0` (smallest)
   * `j = k - 1` (second largest)

### Pointer logic

* If `nums[i] + nums[j] > nums[k]`

  * Then **all indices from `i` to `j-1`** also form valid triangles with `j` and `k`
  * Add `(j - i)` to the answer
  * Decrease `j`
* Else

  * Increase `i` to make the sum larger

### Time Complexity

* Sorting: `O(n log n)`
* Two-pointer scan: `O(n²)`
* **Overall:** `O(n²)`

---

## 🧩 Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0

        # Fix the largest side nums[k]
        for k in range(n - 1, 1, -1):
            i, j = 0, k - 1

            while i < j:
                if nums[i] + nums[j] > nums[k]:
                    # All pairs (i, i+1, ..., j-1) with j are valid
                    count += (j - i)
                    j -= 1
                else:
                    i += 1

        return count
```

---

## 🧪 Worked-Out Example

### Input

```text
nums = [2, 2, 3, 4]
```

### Step 1: Sort

```text
nums = [2, 2, 3, 4]
```

---

### Step 2: Fix k = 3 → nums[k] = 4

| i | j | nums[i] + nums[j] | Condition | Action |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 + 3 = 5 | > 4 ✅ | count += (2 - 0) = 2, j-- |
| 0 | 1 | 2 + 2 = 4 | > 4 ❌ | i++ |

Valid triangles found:

```
(2,3,4), (2,3,4)
```

---

### Step 3: Fix k = 2 → nums[k] = 3

| i | j | nums[i] + nums[j] | Condition | Action |
| --- | --- | --- | --- | --- |
| 0 | 1 | 2 + 2 = 4 | > 3 ✅ | count += (1 - 0) = 1 |

Valid triangle:

```
(2,2,3)
```

---

### ✅ Final Count

```
Total valid triangles = 3
```

---

## 🧠 Why This Works (Interview Takeaway)

* Sorting converts a **3-sum-like brute force** into a **two-pointer counting problem**
* Key trick: when a condition works for `(i, j)`, it works for a **range of indices**
* Common pattern used in:

  * Triangle problems
  * Pair counting
  * Optimized combinatorics

---

If you want, I can also provide:

* Dry-run visualization
* Edge-case analysis
* Comparison with brute force
* Similar problems using the same two-pointer pattern