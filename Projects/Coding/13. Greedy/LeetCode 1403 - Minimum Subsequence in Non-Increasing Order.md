---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1403: Minimum Subsequence in Non-Increasing Order"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1403: Minimum Subsequence in Non-Increasing Order

Below is a complete, structured explanation of **LeetCode 1403 – Minimum Subsequence in Non-Increasing Order**, aligned with interview expectations and greedy-problem reasoning.

---

## 1. Problem Statement

You are given an array of integers `nums`.

Your task is to return the **minimum subsequence** (not necessarily contiguous) such that:

1. The **sum of the subsequence** is **strictly greater** than the sum of the remaining elements.
2. The subsequence is returned in **non-increasing order**.
3. If multiple answers exist, return the one with the **maximum total sum**.
4. A subsequence keeps the relative order, but since we can reorder for output, sorting is allowed.

---

## 2. Key Observations

### Observation 1: Greedy Choice

To exceed the sum of remaining elements using the **fewest elements**, we should **pick the largest numbers first**.

This is a classic **“take biggest contributors first”** greedy pattern.

---

### Observation 2: Sorting Enables Optimal Greedy

If we sort `nums` in **descending order**, then:

* Each step adds the maximum possible value to our subsequence.
* We minimize the subsequence size.
* The output order requirement (non-increasing) is automatically satisfied.

---

### Observation 3: Stop Condition

Let:

* `total_sum` = sum of all elements
* `curr_sum` = sum of selected elements

We stop as soon as:

```
curr_sum > total_sum - curr_sum
```

At this point:

* Subsequence sum is strictly greater than the remaining sum.
* Adding more elements would violate the “minimum subsequence” requirement.

---

## 3. Why This Is a Greedy Problem

This problem satisfies the **greedy-choice property**:

* Choosing the largest available element **never harms** the ability to reach an optimal solution.
* Any solution that skips a larger element in favor of a smaller one would require **more elements** to exceed the remaining sum.

Hence, sorting + greedy accumulation is optimal.

---

## 4. Greedy Strategy (Step-by-Step)

1. Compute the total sum of the array.
2. Sort the array in descending order.
3. Initialize an empty result list and a running sum.
4. Iterate through the sorted array:

   * Add the current number to the result.
   * Update the running sum.
   * Stop once the running sum exceeds the remaining sum.

---

## 5. Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def minSubsequence(self, nums: List[int]) -> List[int]:
        total_sum: int = sum(nums)
        nums.sort(reverse=True)

        curr_sum: int = 0
        result: List[int] = []

        for num in nums:
            curr_sum += num
            result.append(num)

            if curr_sum > total_sum - curr_sum:
                break

        return result
```

---

## 6. Complete Worked Example (Step-by-Step Processing)

### Input

```text
nums = [4, 3, 10, 9, 8]
```

---

### Step 1: Compute Total Sum

```
total_sum = 4 + 3 + 10 + 9 + 8 = 34
```

---

### Step 2: Sort in Descending Order

```text
nums = [10, 9, 8, 4, 3]
```

---

### Step 3: Greedy Accumulation

| Step | Picked | curr\_sum | remaining\_sum | Condition |
| --- | --- | --- | --- | --- |
| 1 | 10 | 10 | 24 | 10 ≤ 24 ❌ |
| 2 | 9 | 19 | 15 | 19 > 15 ✅ |

Stop immediately.

---

### Step 4: Result

```
[10, 9]
```

* Subsequence sum = 19
* Remaining sum = 15
* 19 > 15 ✔
* Minimum number of elements ✔
* Non-increasing order ✔

---

## 7. Time and Space Complexity

### Time Complexity

* Sorting: **O(n log n)**
* Single pass: **O(n)**
* Overall: **O(n log n)**

### Space Complexity

* Result list: **O(n)** (worst case)

---

## 8. Interview Takeaway

This problem is a textbook example of:

* **Greedy with sorting**
* **Prefix sum comparison**
* **“Take largest first” heuristic**

If you recognize this pattern, the solution becomes immediate and robust.

If you want, I can also:

* Prove correctness formally
* Show why reverse greedy (smallest first) fails
* Generalize this pattern to similar LeetCode problems