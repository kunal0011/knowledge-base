---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 162: Find Peak Element"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 162: Find Peak Element

---

### Problem Statement

A peak element is an element that is strictly greater than its neighbors. Given a 0-indexed integer array `nums`, find a peak element in `O(log n)` time.

---

### Key Observation

* Compare `nums[mid]` with `nums[mid + 1]`.
* If `nums[mid] < nums[mid + 1]`, an uphill slope guarantees at least one peak in the right half: `left = mid + 1`.
* Otherwise, a peak exists at or to the left of `mid`: `right = mid`.

---

### Core Technique: Binary Search on Slopes (Gradient Ascend)

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < nums[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left
```

---

### Worked-Out Example

```python
nums = [1, 2, 1, 3, 5, 6, 4]
left = 0, right = 6 -> mid = 3: nums[3]=3 < nums[4]=5 -> left = 4
left = 4, right = 6 -> mid = 5: nums[5]=6 > nums[6]=4 -> right = 5
left = 4, right = 5 -> mid = 4: nums[4]=5 < nums[5]=6 -> left = 5
left == right (5) -> return index 5 (value 6 is a peak)
```

---

### Complexity Analysis

* **Time Complexity:** `O(log n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Binary search works on unsorted arrays whenever you can discard half the search space based on slope direction.