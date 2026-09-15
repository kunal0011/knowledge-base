---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 4: Median of Two Sorted Arrays"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 4: Median of Two Sorted Arrays

---

### Problem Statement

Given two sorted arrays `nums1` and `nums2` of size `m` and `n`, return the median of the two sorted arrays in `O(log (m+n))` time.

---

### Key Observation

* Binary search for partition cut in the **smaller** array `nums1` (size `m <= n`).
* Partition both arrays into left and right halves such that `len(left_half) == len(right_half)` and `max(left1, left2) <= min(right1, right2)`.

---

### Core Technique: Dual Array Partition Binary Search

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        total = m + n
        half = (total + 1) // 2
        
        while left <= right:
            i = (left + right) // 2
            j = half - i
            
            left1 = nums1[i - 1] if i > 0 else float('-inf')
            right1 = nums1[i] if i < m else float('inf')
            left2 = nums2[j - 1] if j > 0 else float('-inf')
            right2 = nums2[j] if j < n else float('inf')
            
            if left1 <= right2 and left2 <= right1:
                if total % 2 != 0:
                    return float(max(left1, left2))
                return (max(left1, left2) + min(right1, right2)) / 2.0
            elif left1 > right2:
                right = i - 1
            else:
                left = i + 1
        return 0.0
```

---

### Worked-Out Example

```python
nums1 = [1, 3], nums2 = [2]
m = 2, n = 1 -> swap -> nums1 = [2], nums2 = [1, 3]
total = 3, half = 2
i = 1, j = 1 -> left1 = 2, right1 = inf, left2 = 1, right2 = 3
left1 (2) <= right2 (3) and left2 (1) <= right1 (inf) -> Valid partition!
Median = max(left1, left2) = max(2, 1) = 2.0
```

---

### Complexity Analysis

* **Time Complexity:** `O(log(min(m, n)))`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Always run binary search on the shorter array to ensure partition indices remain non-negative and log-bounded.