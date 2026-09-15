---
date: "2025-12-23"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 33: Search in Rotated Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 33: Search in Rotated Sorted Array

Below is a complete, structured explanation of **LeetCode 33 – Search in Rotated Sorted Array**, aligned with binary-search reasoning and implementation rigor.

---

## LeetCode 33: Search in Rotated Sorted Array

### Problem Statement

You are given an integer array `nums` sorted in **ascending order**, but **rotated at an unknown pivot**.  
You are also given an integer `target`.

Return the **index of `target`** if it exists in `nums`; otherwise, return `-1`.

**Constraints**

* All elements in `nums` are **distinct**
* Time complexity must be **O(log n)**

---

### Key Observation

Although the array is rotated, the following invariant always holds:

> **At least one half of the array is always sorted**.

For any middle index `mid`:

* Either the **left half `[left … mid]` is sorted**
* Or the **right half `[mid … right]` is sorted**

This allows us to **adapt binary search** instead of reverting to linear search.

---

### Binary Search Technique (Modified Binary Search)

At each step:

1. Compute `mid`
2. Check if `nums[mid] == target`
3. Determine which half is sorted:

   * **Left sorted** if `nums[left] <= nums[mid]`
   * Otherwise, **right sorted**
4. Decide whether `target` lies inside the sorted half
5. Discard the irrelevant half

This guarantees logarithmic time.

---

### Algorithm Steps

1. Initialize two pointers: `left = 0`, `right = len(nums) - 1`
2. While `left <= right`:

   * Compute `mid`
   * If `nums[mid] == target`, return `mid`
   * If left half is sorted:

     * Check if `target` lies between `nums[left]` and `nums[mid]`
   * Else right half is sorted:

     * Check if `target` lies between `nums[mid]` and `nums[right]`
3. If not found, return `-1`

---

### Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2

            if nums[mid] == target:
                return mid

            # Left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1
```

---

### Worked-Out Example

**Input**

```text
nums = [4,5,6,7,0,1,2]
target = 0
```

| left | mid | right | nums[mid] | Observation | Action |
| --- | --- | --- | --- | --- | --- |
| 0 | 3 | 6 | 7 | Left half [4,5,6,7] sorted | Target not in left → move right |
| 4 | 5 | 6 | 1 | Left half [0,1] sorted | Target in left → move left |
| 4 | 4 | 4 | 0 | Found target | Return index 4 |

**Output**

```
4
```

---

### Why This Works

* Binary search is preserved because **one side is always sorted**
* We intelligently eliminate half the search space every iteration
* Meets the required **O(log n)** time complexity

---

### Common Pitfalls

* Forgetting to check which half is sorted
* Incorrect boundary comparisons (`<=` vs `<`)
* Using linear search (violates constraints)

---

If you want, I can also:

* Convert this into a **decision tree visualization**
* Compare with **LeetCode 81 (duplicates allowed)**
* Explain why standard binary search fails without modification