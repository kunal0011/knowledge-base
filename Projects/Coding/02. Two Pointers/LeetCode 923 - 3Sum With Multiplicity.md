---
date: "2025-12-23"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 923: 3Sum With Multiplicity"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 923: 3Sum With Multiplicity

Below is a **complete, structured explanation** of **LeetCode 923 – 3Sum With Multiplicity**, tailored for algorithmic clarity and interview readiness.

---

## 1. Problem Statement

You are given an integer array `arr` and an integer `target`.

Return the **number of tuples** `(i, j, k)` such that:

* `0 <= i < j < k < len(arr)`
* `arr[i] + arr[j] + arr[k] == target`

Since the answer can be very large, return it **modulo 10⁹ + 7**.

---

## 2. Key Observations

1. **Order matters by index, not by value**

   * We count distinct index triples, not unique value triples.
2. **Array length is small enough to sort**

   * `len(arr) ≤ 3000`
   * Sorting enables a **two-pointer strategy**.
3. **Duplicates are the core challenge**

   * Unlike classic 3Sum, duplicates must be counted using **combinatorics**.
4. **Brute force is infeasible**

   * `O(n³)` → ~27 billion operations in worst case.

---

## 3. Two-Pointer Technique (Core Idea)

### Strategy

1. **Sort the array**
2. Fix one element `arr[i]`
3. Use two pointers:

   * `left = i + 1`
   * `right = n - 1`
4. Move pointers based on comparison with `target`

---

### Case Analysis When a Valid Triplet is Found

Let:

```
arr[i] + arr[left] + arr[right] == target
```

#### Case 1: `arr[left] != arr[right]`

* Count how many times `arr[left]` repeats → `count_left`
* Count how many times `arr[right]` repeats → `count_right`
* Number of valid combinations:

```
count_left * count_right
```

Move both pointers inward.

---

#### Case 2: `arr[left] == arr[right]`

All values between `left` and `right` are equal.

Let:

```
k = right - left + 1
```

Number of ways to choose 2 indices:

```
C(k, 2) = k * (k - 1) // 2
```

Break (no further pairs possible).

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        MOD = 10**9 + 7
        arr.sort()
        n = len(arr)
        result = 0

        for i in range(n):
            left = i + 1
            right = n - 1

            while left < right:
                total = arr[i] + arr[left] + arr[right]

                if total < target:
                    left += 1
                elif total > target:
                    right -= 1
                else:
                    # Case 1: different values
                    if arr[left] != arr[right]:
                        left_val = arr[left]
                        right_val = arr[right]

                        count_left = 0
                        count_right = 0

                        while left < right and arr[left] == left_val:
                            count_left += 1
                            left += 1

                        while left <= right and arr[right] == right_val:
                            count_right += 1
                            right -= 1

                        result += count_left * count_right

                    # Case 2: same values
                    else:
                        k = right - left + 1
                        result += k * (k - 1) // 2
                        break

        return result % MOD
```

---

## 5. Worked-Out Example

### Input

```
arr = [1,1,2,2,3,3,4,4,5,5]
target = 8
```

---

### Step-by-Step Breakdown

#### Sorted Array

```
[1,1,2,2,3,3,4,4,5,5]
```

---

### Fix `i = 0` → `arr[i] = 1`

| left | right | sum | Action |
| --- | --- | --- | --- |
| 1 (1) | 9 (5) | 7 | left++ |
| 2 (2) | 9 (5) | 8 | valid |

* `2` appears **twice**
* `5` appears **twice**
* Add `2 × 2 = 4`

---

### Fix `i = 2` → `arr[i] = 2`

Valid combinations:

* `(2,2,4)`
* `2 occurs twice, 4 occurs twice`
* Add `4`

---

### Fix `i = 4` → `arr[i] = 3`

* `(3,3,2)` not valid due to order
* `(3,3,2)` excluded
* `(3,3,2)` skipped

But:

* `(3,3,2)` is avoided due to pointer rules

---

### Fix `i = 4` with `left = 5`, `right = 6`

```
3 + 3 + 2 != 8
```

Later:

* `(3,3,2)` skipped
* `(3,3,2)` avoided

---

### Total Count

```
20
```

---

## 6. Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | `O(n²)` |
| Space Complexity | `O(1)` (excluding sort) |

---

## 7. Key Takeaways

* Sorting enables deterministic pointer movement
* Duplicate handling requires **count aggregation**
* Combination math avoids redundant iteration
* This problem is a **hybrid of 3Sum + Counting**

---

If you want:

* A **visual pointer movement diagram**
* Or a **comparison with hashmap-based solution**
* Or an **optimized explanation for interviews**

Tell me how you want to proceed.