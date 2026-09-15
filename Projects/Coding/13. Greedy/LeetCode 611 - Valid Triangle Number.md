---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 611: Valid Triangle Number"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 611: Valid Triangle Number

**LeetCode 611 – Valid Triangle Number**, aligned with how this problem is typically expected to be solved in interviews and competitive programming.

---

## 1. Problem Statement (LeetCode 611)

You are given an integer array `nums`.

Return the number of **triplets** `(i, j, k)` such that:

* `i < j < k`
* `nums[i]`, `nums[j]`, and `nums[k]` **can form a triangle**

### Triangle Validity Rule

For any three sides `a ≤ b ≤ c`, they form a triangle **if and only if**:

```
a + b > c
```

---

## 2. Key Observation

### Observation 1: Sorting simplifies triangle checks

If the array is sorted:

```
nums[i] ≤ nums[j] ≤ nums[k]
```

You **only need to check one condition**:

```
nums[i] + nums[j] > nums[k]
```

The other two triangle conditions are **always satisfied** due to sorting.

---

### Observation 2: Fix the largest side

Instead of choosing all three indices independently:

* Fix `k` as the **largest side**
* Find how many pairs `(i, j)` (`i < j < k`) satisfy:

```
nums[i] + nums[j] > nums[k]
```

This reduces the problem to a **two-pointer search**.

---

## 3. Greedy + Two-Pointer Trick

### Core Greedy Insight

When the array is sorted:

* If `nums[left] + nums[right] > nums[k]`
* Then **all elements between `left` and `right`** will also form valid triangles with `nums[right]` and `nums[k]`

Why?

```
nums[left] ≥ nums[left], nums[left+1], ..., nums[right-1]
```

So:

```
(nums[left+1] + nums[right]) > nums[k]
(nums[left+2] + nums[right]) > nums[k]
...
```

This allows counting **multiple triangles in O(1)**.

---

## 4. Algorithm (Step-by-Step)

1. Sort the array
2. Iterate `k` from right to left (largest side)
3. Use two pointers:

   * `left = 0`
   * `right = k - 1`
4. While `left < right`:

   * If `nums[left] + nums[right] > nums[k]`

     * Add `(right - left)` to answer
     * Move `right--`
   * Else

     * Move `left++`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        count = 0

        # Fix the largest side
        for k in range(n - 1, 1, -1):
            left, right = 0, k - 1

            while left < right:
                if nums[left] + nums[right] > nums[k]:
                    # All pairs (left ... right-1, right) are valid
                    count += (right - left)
                    right -= 1
                else:
                    left += 1

        return count
```

---

## 6. Complete Worked Example (All Steps)

### Input

```text
nums = [2, 2, 3, 4]
```

### Step 1: Sort

```text
nums = [2, 2, 3, 4]
```

---

### Step 2: Fix k = 3 (nums[k] = 4)

```
left = 0, right = 2
```

| left | right | nums[left] + nums[right] | > 4? | Action |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 + 3 = 5 | Yes | count += (2 - 0) = 2 |
|  |  |  |  | right-- → 1 |
| 0 | 1 | 2 + 2 = 4 | No | left++ → 1 |

Stop (`left == right`)

**Triangles counted so far: 2**

* (2,3,4)
* (2,3,4) (different index)

---

### Step 3: Fix k = 2 (nums[k] = 3)

```
left = 0, right = 1
```

| left | right | nums[left] + nums[right] | > 3? | Action |
| --- | --- | --- | --- | --- |
| 0 | 1 | 2 + 2 = 4 | Yes | count += (1 - 0) = 1 |
|  |  |  |  | right-- |

Stop

**Triangles added: 1**

* (2,2,3)

---

### Final Count

```
Total Valid Triangles = 3
```

---

## 7. Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | `O(n²)` |
| Space Complexity | `O(1)` (excluding sort) |

---

## 8. Why This is a Greedy Problem

* You **greedily fix the largest side**
* You **greedily count multiple valid pairs at once**
* You exploit **monotonicity due to sorting**

This transforms a naive `O(n³)` problem into an efficient `O(n²)` solution.

---

If you want, I can also:

* Draw the **two-pointer movement as a visual table**
* Compare this with a **binary search approach**
* Explain **why brute force TLEs**
* Show **edge cases** (zeros, duplicates)

Tell me how deep you want to go.