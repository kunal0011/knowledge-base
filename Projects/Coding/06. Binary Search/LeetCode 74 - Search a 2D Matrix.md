---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 74: Search a 2D Matrix"
tags:
  - leetcode
  - coding
  - binary-search
  - matrix
  - coordinate-mapping
  - amazon
  - google
---

# LeetCode 74: Search a 2D Matrix

**Target Companies:** Amazon, Microsoft, Google, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Virtual 1D Coordinate Mapping Binary Search  

---

### Problem Statement

You are given an $m \times n$ integer matrix `matrix` with the following two properties:
1. Each row is sorted in non-decreasing order.
2. The first integer of each row is greater than the last integer of the previous row.

Given an integer `target`, return `true` if `target` is in `matrix` or `false` otherwise.

You must write a solution in $O(\log(m \cdot n))$ time complexity.

---

### Input & Output Formats & Constraints

- **Input:** `matrix: List[List[int]]`, `target: int`
- **Output:** `bool` (`True` if found, `False` otherwise)
- **Constraints:**
  - $m == \text{matrix.length}$
  - $n == \text{matrix}[i].\text{length}$
  - $1 \le m, n \le 100$
  - $-10^4 \le \text{matrix}[i][j], \text{target} \le 10^4$

---

### Key Idea & Intuition

Because each row is sorted and the first element of row $i + 1$ is strictly greater than the last element of row $i$, the entire $m \times n$ matrix is strictly sorted in row-major order.

Instead of performing two separate binary searches (one to find the row, another to find the column), we treat the matrix as a single, contiguous **virtual 1D array** of size $m \cdot n$:
- Range of virtual indices: $[0, m \cdot n - 1]$.
- Any 1D index `mid` maps deterministically to 2D coordinates via integer division and modulo:
  $$\text{row} = \text{mid} // n$$
  $$\text{col} = \text{mid} \% n$$
- We execute standard binary search directly over $[0, m \cdot n - 1]$, achieving true $O(\log(m \cdot n))$ time with zero extra memory allocation!

---

### Solution Approach (Step-by-Step)

1. If `matrix` is empty or `matrix[0]` is empty, return `False`.
2. Let $m = \text{len}(matrix)$ and $n = \text{len}(matrix[0])$.
3. Set `left = 0`, `right = m * n - 1`.
4. While `left <= right`:
   - Compute `mid = left + (right - left) // 2`.
   - Calculate matrix coordinates: `r = mid // n`, `c = mid % n`.
   - If `matrix[r][c] == target`: return `True`.
   - Else if `matrix[r][c] < target`: `left = mid + 1`.
   - Else: `right = mid - 1`.
5. Return `False`.

---

### Visual Algorithm Walkthrough

```
Matrix (3 x 4):
  Row 0: [ 1,   3,   5,   7 ]
  Row 1: [ 10, 11,  16,  20 ]
  Row 2: [ 23, 30,  34,  60 ]

Target = 3
Virtual 1D Array (Indices 0 to 11):
  [ 1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60 ]
    0  1  2  3   4   5   6   7   8   9  10  11

Iteration 1:
  left = 0, right = 11 -> mid = (0 + 11) // 2 = 5
  row = 5 // 4 = 1, col = 5 % 4 = 1 -> matrix[1][1] = 11
  11 > 3 -> right = mid - 1 = 4

Iteration 2:
  left = 0, right = 4 -> mid = (0 + 4) // 2 = 2
  row = 2 // 4 = 0, col = 2 % 4 = 2 -> matrix[0][2] = 5
  5 > 3 -> right = mid - 1 = 1

Iteration 3:
  left = 0, right = 1 -> mid = (0 + 1) // 2 = 0
  row = 0 // 4 = 0, col = 0 % 4 = 0 -> matrix[0][0] = 1
  1 < 3 -> left = mid + 1 = 1

Iteration 4:
  left = 1, right = 1 -> mid = 1
  row = 1 // 4 = 0, col = 1 % 4 = 1 -> matrix[0][1] = 3 == target!
  MATCH FOUND! Return True.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Target Exists in Interior
- **Input:** `matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]`, `target = 3`
- **Output:** `true`

#### Example 2: Target Does Not Exist
- **Input:** `matrix = [[1, 3, 5, 7], [10, 11, 16, 20], [23, 30, 34, 60]]`, `target = 13`
- **Output:** `false`

#### Example 3: $1 \times 1$ Matrix
- **Input:** `matrix = [[1]]`, `target = 1`
- **Output:** `true`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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
            r = mid // n
            c = mid % n
            val = matrix[r][c]
            
            if val == target:
                return True
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    bool searchMatrix(std::vector<std::vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;

        int m = matrix.size(), n = matrix[0].size();
        int left = 0, right = m * n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int r = mid / n;
            int c = mid % n;
            int val = matrix[r][c];

            if (val == target) {
                return true;
            } else if (val < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }

        int m = matrix.length, n = matrix[0].length;
        int left = 0, right = m * n - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int r = mid / n;
            int c = mid % n;
            int val = matrix[r][c];

            if (val == target) {
                return true;
            } else if (val < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log(M \cdot N))$ — Direct binary search across the virtual 1D array of length $M \cdot N$.
- **Space Complexity:** $O(1)$ auxiliary space — Coordinates are computed on the fly using modulo and division.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Virtual 1D-to-2D Mapping: `row = idx / cols`, `col = idx % cols`.
- **Trap:** Confusing with LeetCode 240 (Search a 2D Matrix II): In LeetCode 240, rows and columns are sorted independently, but row $i + 1$ does not necessarily start greater than row $i$'s end. LeetCode 240 requires the $O(M + N)$ top-right corner pointer approach, whereas LeetCode 74 allows true $O(\log(M \cdot N))$ binary search!