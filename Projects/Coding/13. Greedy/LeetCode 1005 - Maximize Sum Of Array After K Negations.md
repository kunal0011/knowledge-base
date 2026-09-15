---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1005: Maximize Sum Of Array After K Negations"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1005: Maximize Sum Of Array After K Negations

Below is a **complete, structured explanation of LeetCode 1005** aligned with how greedy problems are typically reasoned in interviews and competitive programming.

---

## LeetCode 1005 — *Maximize Sum Of Array After K Negations*

### Problem Statement

You are given an integer array `nums` and an integer `k`.

In one operation, you can choose any index `i` and **negate** `nums[i]` (i.e., replace `nums[i]` with `-nums[i]`).

You must perform **exactly `k` operations**.

Return the **maximum possible sum** of the array after performing these `k` negations.

---

### Key Observations

1. **Negating a negative number increases the total sum**

   * Example: `-5 → 5` increases sum by `10`.
   * Therefore, **negating negative numbers is always beneficial**.
2. **Negating a positive number decreases the total sum**

   * Example: `5 → -5` decreases sum by `10`.
   * This should be avoided unless forced.
3. **If `k` is still left after all negatives are handled**

   * Negating the same element twice cancels out.
   * Only the **parity (odd/even)** of remaining `k` matters.
4. **If remaining `k` is odd**

   * We must negate **exactly one element**.
   * To minimize damage, negate the element with **smallest absolute value**.

---

### Greedy Strategy (Core Idea)

1. Sort the array.
2. Convert as many negative numbers to positive as possible (while `k > 0`).
3. If `k` is still odd:

   * Flip the element with the **minimum absolute value**.
4. Compute the final sum.

This works because each greedy choice locally maximizes the sum and does not block a better global outcome.

---

### Greedy Solution Tricks

* **Sorting helps**:

  * Negatives appear first.
  * Minimum absolute value is easy to identify.
* **Early termination**:

  * Stop flipping negatives once `k == 0`.
* **Parity shortcut**:

  * If remaining `k` is even → no further changes needed.
  * If remaining `k` is odd → flip smallest absolute value once.

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        nums.sort()

        i = 0
        n = len(nums)

        # Step 1: Flip negative numbers while possible
        while i < n and nums[i] < 0 and k > 0:
            nums[i] = -nums[i]
            k -= 1
            i += 1

        # Step 2: If k is odd, flip the smallest absolute value
        if k % 2 == 1:
            min_index = nums.index(min(nums))
            nums[min_index] = -nums[min_index]

        return sum(nums)
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```text
nums = [-4, -2, -3, 1, 5]
k = 4
```

---

#### Step 1: Sort the array

```
[-4, -3, -2, 1, 5]
```

---

#### Step 2: Flip negative numbers greedily

| Operation | k | Array State |
| --- | --- | --- |
| Flip -4 → 4 | 3 | [4, -3, -2, 1, 5] |
| Flip -3 → 3 | 2 | [4, 3, -2, 1, 5] |
| Flip -2 → 2 | 1 | [4, 3, 2, 1, 5] |

Now:

* No negatives left
* `k = 1` remaining

---

#### Step 3: Remaining `k` is odd → flip smallest absolute value

Smallest value in array:

```
min = 1
```

Flip:

```
1 → -1
```

Final array:

```
[4, 3, 2, -1, 5]
```

---

#### Step 4: Compute sum

```
4 + 3 + 2 - 1 + 5 = 13
```

---

### Final Answer

```
13
```

---

### Time and Space Complexity

* **Time Complexity**: `O(n log n)` (due to sorting)
* **Space Complexity**: `O(1)` (in-place modifications)

---

### Greedy Pattern Recognition

This problem fits the **“Greedy with Sorting + Parity Handling”** pattern, commonly seen when:

* Operations are limited (`k`)
* Each operation has reversible effects
* Only the parity of remaining operations matters

---

If you want, I can also:

* Show a **heap-based alternative**
* Provide a **proof of correctness**
* Compare this with similar problems like **LeetCode 948 / 861**

Just tell me how deep you want to go.