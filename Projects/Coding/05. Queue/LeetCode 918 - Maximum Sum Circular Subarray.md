---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 918: Maximum Sum Circular Subarray"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 918: Maximum Sum Circular Subarray

Below is a **structured, interview-ready explanation** of **LeetCode 918 – Maximum Sum Circular Subarray**, aligned with how this problem is typically reasoned about in competitive programming and system interviews.

---

## LeetCode 918: Maximum Sum Circular Subarray

### Problem Statement

Given a **circular integer array** `nums`, return the **maximum possible sum of a non-empty subarray**.

* The array is **circular**, meaning the element after the last element is the first element.
* A subarray can wrap around the end to the beginning.
* Each element may be used **at most once** in a subarray.

**Example**

```text
Input: nums = [5, -3, 5]
Output: 10
Explanation: Subarray [5, 5] wraps around and has sum = 10
```

---

## Key Observation

A maximum circular subarray must fall into **one of two categories**:

### Case 1: Subarray does NOT wrap

This is the classic **maximum subarray sum** problem.

➡️ Solved using **Kadane’s Algorithm**

---

### Case 2: Subarray DOES wrap

If a subarray wraps, then it means:

> We take the **entire array sum** and **exclude a contiguous subarray in the middle** that has the **minimum sum**

So:

```
max_wrap = total_sum - min_subarray_sum
```

---

### Important Edge Case

If **all elements are negative**:

* Kadane (max subarray) already gives the correct answer.
* `total_sum - min_subarray_sum` becomes `0` (invalid, since subarray must be non-empty).

➡️ In this case, **return Kadane’s result only**.

---

## Queue / Deque Insight (Advanced Perspective)

This problem can also be solved using a **monotonic deque + prefix sums**, which generalizes to harder problems like:

* Subarray sum with length constraints
* Maximum sum subarray in circular array of size ≤ `n`

### Core Idea (Deque Method)

1. Duplicate the array to simulate circularity.
2. Use **prefix sums**.
3. Maintain a **monotonic increasing deque** of prefix sums.
4. For each index `i`, maximize:

   ```
   prefix[i] - min(prefix[j])   where i - j <= n
   ```

⚠️ However, **this problem does NOT require the deque approach**, since the Kadane-based solution is simpler, faster, and cleaner.

---

## Optimal Approach Used (Kadane + Inversion Trick)

### Steps

1. Run Kadane to find:

   * `max_subarray_sum`
2. Run Kadane on **negated array** to find:

   * `min_subarray_sum`
3. Compute:

   ```
   max_circular = total_sum - min_subarray_sum
   ```
4. Return:

   ```
   max(max_subarray_sum, max_circular)
   ```

   except when all numbers are negative.

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        def kadane(arr: List[int]) -> int:
            curr = best = arr[0]
            for x in arr[1:]:
                curr = max(x, curr + x)
                best = max(best, curr)
            return best

        total_sum = sum(nums)

        max_subarray = kadane(nums)
        min_subarray = -kadane([-x for x in nums])

        # If all numbers are negative, circular sum becomes invalid
        if max_subarray < 0:
            return max_subarray

        return max(max_subarray, total_sum - min_subarray)
```

---

## Worked Out Example

### Input

```text
nums = [5, -3, 5]
```

### Step 1: Normal Kadane

```
Max subarray (non-circular) = 7   → [5, -3, 5]
```

### Step 2: Total Sum

```
total_sum = 5 + (-3) + 5 = 7
```

### Step 3: Minimum Subarray

```
Minimum subarray = -3
```

### Step 4: Circular Sum

```
max_circular = total_sum - min_subarray
             = 7 - (-3)
             = 10
```

### Final Answer

```
max(7, 10) = 10
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** |

---

## Final Takeaway

* This problem is a **classic Kadane extension**.
* Circular logic reduces cleanly to:

  ```
  max(normal_max, total_sum - normal_min)
  ```
* The deque approach is conceptually powerful but unnecessary here.

If you want, I can also:

* Walk through the **deque-based solution step by step**
* Compare **918 vs 53 vs 209** pattern-wise
* Provide **interview traps and common mistakes**