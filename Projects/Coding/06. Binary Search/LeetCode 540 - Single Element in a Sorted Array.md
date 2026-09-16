---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 540: Single Element in a Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
  - amazon
  - google
---

# LeetCode 540: Single Element in a Sorted Array

**Target Companies:** Amazon, Google, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** Binary Search on Parity Invariant

---

### Problem Statement

You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.

Return the single element that appears only once.

Your solution must run in **$O(\log n)$ time** and **$O(1)$ space**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` sorted in non-decreasing order.
- **Output:** `int` representing the unique element.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $0 \le \text{nums}[i] \le 10^5$
  - Array length is always odd ($2k + 1$).

---

### Key Idea & Intuition

- **The Index Parity Invariant:**
  - Before the single element:
    - Pairs start at **even** indices and end at **odd** indices: `nums[2k] == nums[2k + 1]`.
  - After the single element:
    - The single element disrupts the alignment!
    - Pairs now start at **odd** indices and end at **even** indices: `nums[2k - 1] == nums[2k]`.
- **Binary Search Decision:**
  - Check midpoint `mid`:
    - Ensure `mid` is even: if `mid % 2 == 1`, decrement `mid -= 1`.
    - If `nums[mid] == nums[mid + 1]`:
      - The pair is intact! The single element must lie to the **right**: `left = mid + 2`.
    - Else:
      - The pattern is broken! The single element is either at `mid` or to the **left**: `right = mid`.
  - When `left == right`, `nums[left]` is the unique single element.

---

### Solution Approach (Step-by-Step)

1. Initialize `left = 0`, `right = len(nums) - 1`.
2. While `left < right`:
   - `mid = left + (right - left) // 2`.
   - If `mid % 2 == 1`: `mid -= 1` (align to even index).
   - If `nums[mid] == nums[mid + 1]`:
     - `left = mid + 2`
   - Else:
     - `right = mid`
3. Return `nums[left]`.

---

### Visual Algorithm Walkthrough

```
Indices:    0  1  2  3  4  5  6  7  8
Array:     [1, 1, 2, 3, 3, 4, 4, 8, 8]
                   ^
            Single Element (2)

Pairs before single element (2):
idx 0 (even), idx 1 (odd): 1 == 1  -> MATCH (Pattern OK)

At idx 2 (even): 2 != nums[3] (3)   -> BROKEN! Search left half!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Single Element in the Middle
- **Input:** `nums = [1,1,2,3,3,4,4,8,8]`
- **Trace:**
  - `left = 0, right = 8`.
  - `mid = 4` (even). `nums[4] = 3, nums[5] = 4`. `3 != 4`.
  - Pattern broken -> `right = 4`.
  - `left = 0, right = 4`.
  - `mid = 2` (even). `nums[2] = 2, nums[3] = 3`. `2 != 3`.
  - Pattern broken -> `right = 2`.
  - `left = 0, right = 2`.
  - `mid = 0` (even). `nums[0] = 1, nums[1] = 1`. `1 == 1`.
  - Pattern intact -> `left = 0 + 2 = 2`.
  - `left == right == 2`. Loop terminates.
- **Output:** `nums[2] = 2`

#### Example 2: Single Element at End
- **Input:** `nums = [3,3,7,7,10,11,11]`
- **Output:** `10`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        
        while left < right:
            mid = (left + right) // 2
            if mid % 2 == 1:
                mid -= 1  # Force mid to be an even index
                
            if nums[mid] == nums[mid + 1]:
                left = mid + 2
            else:
                right = mid
                
        return nums[left]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int singleNonDuplicate(std::vector<int>& nums) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (mid % 2 == 1) {
                mid--;
            }
            if (nums[mid] == nums[mid + 1]) {
                left = mid + 2;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int left = 0, right = nums.length - 1;
        
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (mid % 2 == 1) {
                mid--;
            }
            if (nums[mid] == nums[mid + 1]) {
                left = mid + 2;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — Standard binary search halving search space on each comparison.
- **Space Complexity:** $O(1)$ — Only constant pointers used.

---

### Takeaway Pattern & Interview Traps

1. **Parity Alignment Trick:**
   - In sorted duplicated arrays, the single element acts as a phase inverter. Before the single element, the first of each pair is at an **even** index; after the single element, the first of each pair shifts to an **odd** index.
   - Forcing `mid` to be even (`if (mid % 2 == 1) mid--;` or using XOR `mid ^ 1`) simplifies the logic: if `nums[mid] == nums[mid + 1]`, the disruption must be to the right (`left = mid + 2`).
2. **XOR Index Bitmask Shortcut:**
   - Notice that for even indices $k$, $k \oplus 1 = k + 1$, and for odd indices $k$, $k \oplus 1 = k - 1$. Comparing `nums[mid] == nums[mid ^ 1]` works universally for both even and odd `mid` without manually branching!

