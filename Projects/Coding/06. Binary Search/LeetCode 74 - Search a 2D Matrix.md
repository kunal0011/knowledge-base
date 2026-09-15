---
date: "2026-08-29"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 74: Search a 2D Matrix"
tags:
  - leetcode
  - coding
  - binary-search
---

# LeetCode 74: Search a 2D Matrix

---

### Problem Statement

You are given an `m x n` integer matrix with sorted rows and first element of each row greater than last of previous. Return `true` if `target` is in matrix.

---

### Key Observation

* The entire matrix is strictly sorted in row-major order.
* We can treat the `m x n` matrix as a flat 1D array of size `m * n` without extra space: `row = mid // n`, `col = mid % n`.

---

### Core Technique: Virtual 1D Coordinate Mapping Binary Search

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m * n - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            r, c = mid // n, mid % n
            val = matrix[r][c]
            
            if val == target:
                return True
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
```

---

### Worked-Out Example

```python
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
m = 3, n = 4 -> total elements = 12
mid = 5 -> row = 5 // 4 = 1, col = 5 % 4 = 1 -> matrix[1][1] = 11 > 3 -> right = 4
mid = 2 -> row = 2 // 4 = 0, col = 2 % 4 = 2 -> matrix[0][2] = 5 > 3 -> right = 1
mid = 1 -> row = 0, col = 1 -> matrix[0][1] = 3 == target -> return True
```

---

### Complexity Analysis

* **Time Complexity:** `O(log(m * n))`
* **Space Complexity:** `O(1)`

---

### Takeaway Pattern

Map 1D index to 2D matrix using `row = idx // cols` and `col = idx % cols` for direct binary search.