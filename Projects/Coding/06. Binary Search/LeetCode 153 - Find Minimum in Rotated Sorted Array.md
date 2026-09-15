---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 153: Find Minimum in Rotated Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 153: Find Minimum in Rotated Sorted Array

---

### Problem Statement

Given a sorted rotated array of unique elements, return the minimum element in `O(log n)` time.

---

### Key Observation

* Compare `nums[mid]` with `nums[right]`.
* If `nums[mid] > nums[right]`, the pivot (minimum) lies strictly in the right half: `left = mid + 1`.
* Otherwise, the minimum is at or to the left of `mid`: `right = mid`.

---

### Core Technique: Inflection Point Binary Search

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]
```

---

### Worked-Out Example

```python
nums = [4, 5, 6, 7, 0, 1, 2]
left = 0, right = 6 -> mid = 3 (nums[3] = 7 > nums[6] = 2) -> left = 4
left = 4, right = 6 -> mid = 5 (nums[5] = 1 < nums[6] = 2) -> right = 5
left = 4, right = 5 -> mid = 4 (nums[4] = 0 < nums[5] = 1) -> right = 4
left == right (4) -> return nums[4] = 0
```

---

### Complexity Analysis

* **Time Complexity:** `O(log n)`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Compare `mid` with `right` boundary to determine which half contains the rotation pivot.