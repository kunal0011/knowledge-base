---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 152: Maximum Product Subarray"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 152: Maximum Product Subarray

**LeetCode 152 – Maximum Product Subarray**, focusing on **state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 152 — Maximum Product Subarray

### Problem Statement

Given an integer array `nums`, find the **contiguous subarray** within the array that has the **largest product**, and return the product.

**Constraints**

* `1 ≤ nums.length ≤ 2 * 10^4`
* `-10 ≤ nums[i] ≤ 10`
* Result fits in 32-bit integer

---

## Key Observation (Why Simple DP Fails)

Unlike sum-based problems, **product behaves non-monotonically**:

* A **negative × negative = positive**
* A large positive product can suddenly become small (or vice versa)

Therefore:

> At every index, we must track **both the maximum and minimum product ending at that index**.

---

## DP State Definition

Let:

* `dp_max[i]` = **maximum product of a subarray ending at index `i`**
* `dp_min[i]` = **minimum product of a subarray ending at index `i`**

Both are required because:

* `dp_min[i-1] * nums[i]` could become the next maximum if `nums[i]` is negative.

---

## DP Transition

At index `i`, the subarray ending at `i` can be:

1. Start fresh with `nums[i]`
2. Extend previous max product subarray
3. Extend previous min product subarray

### Transition Equations

```
dp_max[i] = max(
    nums[i],
    nums[i] * dp_max[i-1],
    nums[i] * dp_min[i-1]
)

dp_min[i] = min(
    nums[i],
    nums[i] * dp_max[i-1],
    nums[i] * dp_min[i-1]
)
```

---

## Base Case

```
dp_max[0] = nums[0]
dp_min[0] = nums[0]
```

---

## Final Answer

```
answer = max(dp_max[i]) for all i
```

We track the global maximum while filling DP.

---

## DP Table Construction (Worked Example)

### Example Input

```text
nums = [2, 3, -2, 4]
```

---

### Step-by-Step DP Table

| i | nums[i] | dp\_max[i] | dp\_min[i] | Explanation |
| --- | --- | --- | --- | --- |
| 0 | 2 | 2 | 2 | Base case |
| 1 | 3 | max(3, 2×3, 2×3) = **6** | min(3, 6, 6) = **3** | Positive extension |
| 2 | -2 | max(-2, 6×-2, 3×-2) = **-2** | min(-2, -12, -6) = **-12** | Negative flips roles |
| 3 | 4 | max(4, -2×4, -12×4) = **4** | min(4, -8, -48) = **-48** | Reset at 4 |

---

### Global Maximum

```
max(dp_max) = 6
```

### Result

```
Output = 6
Subarray = [2, 3]
```

---

## Optimized Space Observation

Only `i-1` is needed → **O(1) space optimization possible**.

---

## Python 3 (DP with State Explanation)

```python
from typing import List

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        dp_max = nums[0]
        dp_min = nums[0]
        result = nums[0]

        for i in range(1, len(nums)):
            curr = nums[i]

            # store previous values before overwrite
            prev_max = dp_max
            prev_min = dp_min

            dp_max = max(
                curr,
                curr * prev_max,
                curr * prev_min
            )

            dp_min = min(
                curr,
                curr * prev_max,
                curr * prev_min
            )

            result = max(result, dp_max)

        return result
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(1)** (optimized) |

---

## Mental Model Summary

* Always track **best and worst product so far**
* Negative numbers **swap roles**
* Zero naturally resets subarrays
* Similar DP pattern appears in:

  * Maximum product paths
  * Stock problems with inversion effects

---

If you want, I can also provide:

* Backtracking-style intuition (why it does *not* work here)
* Visual sign-flip diagram
* Comparison with Kadane’s Algorithm
* Edge cases (`0`, single element, all negatives)

Tell me how deep you want to go.