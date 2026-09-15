---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 532: K-diff Pairs in an Array"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 532: K-diff Pairs in an Array

Below is a **complete, structured explanation** of **LeetCode 532 – K-diff Pairs in an Array**, following your requested format and depth.

---

## LeetCode 532: K-diff Pairs in an Array

### Problem Statement

Given an integer array `nums` and an integer `k`, return the number of **unique k-diff pairs** in the array.

A **k-diff pair** is defined as an integer pair `(nums[i], nums[j])` such that:

* `i != j`
* `|nums[i] - nums[j]| == k`

**Notes**

* Pairs `(a, b)` and `(b, a)` are considered the **same**.
* `k` is a **non-negative integer**.

---

### Key Observations

1. **Uniqueness matters**

   * We count **unique value pairs**, not index pairs.
   * `(1, 3)` and `(3, 1)` count as **one** pair.
2. **Absolute difference**

   * Since `|a - b| = k`, sorting helps reduce this to `b - a = k`.
3. **Special case: `k == 0`**

   * We need elements that appear **at least twice**.
   * Example: `[1,1,1] → (1,1)` is one valid pair.
4. **Why sorting helps**

   * Enables **two-pointer traversal**
   * Prevents duplicate counting
   * Converts absolute difference into directional difference

---

### Two Pointer Technique (Core Idea)

After sorting the array:

* Use two pointers:

  * `left` → smaller element
  * `right` → larger element
* Maintain `right > left`

#### Pointer Movement Logic

Let `diff = nums[right] - nums[left]`

| Condition | Action |
| --- | --- |
| `diff < k` | Move `right` forward |
| `diff > k` | Move `left` forward |
| `diff == k` | Found a valid pair → count it and skip duplicates |

---

### Algorithm Steps

1. Sort `nums`
2. Initialize `left = 0`, `right = 1`
3. While `right < n`:

   * If `left == right`, move `right`
   * Compute difference
   * Adjust pointers based on comparison with `k`
   * On match, increment count and skip duplicates
4. Return the count

---

### Python 3 Implementation (with Typing)

```python
from typing import List

class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        if k < 0:
            return 0  # Absolute difference cannot be negative

        nums.sort()
        n = len(nums)
        left, right = 0, 1
        count = 0

        while right < n:
            if left == right:
                right += 1
                continue

            diff = nums[right] - nums[left]

            if diff < k:
                right += 1
            elif diff > k:
                left += 1
            else:
                count += 1

                left_val = nums[left]
                right_val = nums[right]

                # Skip duplicates
                while left < n and nums[left] == left_val:
                    left += 1
                while right < n and nums[right] == right_val:
                    right += 1

        return count
```

---

### Worked Out Example

#### Example Input

```text
nums = [3, 1, 4, 1, 5]
k = 2
```

#### Step 1: Sort

```text
nums = [1, 1, 3, 4, 5]
```

#### Step 2: Pointer Traversal

| left | right | nums[left] | nums[right] | diff | Action |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 0 | diff < k → move right |
| 0 | 2 | 1 | 3 | 2 | diff == k → count = 1 |
| 1 | 3 | 1 | 4 | 3 | diff > k → move left |
| 2 | 3 | 3 | 4 | 1 | diff < k → move right |
| 2 | 4 | 3 | 5 | 2 | diff == k → count = 2 |

#### Valid Pairs

```
(1, 3)
(3, 5)
```

#### Output

```
2
```

---

### Time and Space Complexity

* **Time Complexity**: `O(n log n)` (due to sorting)
* **Space Complexity**: `O(1)` (excluding sort space)

---

### Final Takeaway

* Sorting + Two Pointers gives a **clean, duplicate-safe** solution.
* Handles both `k > 0` and `k == 0` uniformly.
* This is a **classic two-pointer difference pattern** frequently reused in array problems.

If you want, I can also:

* Compare this with the **HashMap-based approach**
* Convert it into a **pattern template**
* Explain **why naive nested loops fail uniqueness constraints**