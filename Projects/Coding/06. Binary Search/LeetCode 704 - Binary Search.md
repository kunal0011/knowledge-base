---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 704: Binary Search"
tags:
  - leetcode
  - coding
  - binary-search
  - fundamentals
  - amazon
  - google
---

# LeetCode 704: Binary Search

**Target Companies:** Google, Amazon, Microsoft, Apple, Meta  
**Difficulty:** Easy  
**Topic:** Foundational Closed-Interval Binary Search Template  

---

### Problem Statement

Given an array of integers `nums` which is sorted in ascending order, and an integer `target`, write a function to search `target` in `nums`. If `target` exists, then return its index. Otherwise, return `-1`.

You must write an algorithm with $O(\log n)$ runtime complexity.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `int` (index of target or `-1`)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-10^4 < \text{nums}[i], \text{target} < 10^4$
  - All the integers in `nums` are **unique**.
  - `nums` is sorted in ascending order.

---

### Key Idea & Intuition

Binary search operates on the **decrease-and-conquer** principle over a monotonic search space $[L, R]$:
1. Because `nums` is sorted in ascending order, inspecting the midpoint `mid` reveals information about all elements to its left and right:
   - If `nums[mid] == target`: Target found, return `mid`.
   - If `nums[mid] < target`: Since every element to the left of `mid` is $\le nums[mid] < target$, no element in $[L, mid]$ can possibly be equal to `target`. Discard the left half: `left = mid + 1`.
   - If `nums[mid] > target`: Symmetric reasoning proves no element in $[mid, R]$ can equal `target`. Discard the right half: `right = mid - 1`.
2. **The Classic Integer Overflow Bug:**
   In languages like C++ and Java, writing `mid = (left + right) / 2` causes a critical 32-bit signed integer overflow when $left + right \ge 2^{31}$.
   Always compute midpoint as:
   $$\text{mid} = \text{left} + \frac{\text{right} - \text{left}}{2}$$

---

### Solution Approach (Step-by-Step)

1. Initialize `left = 0`, `right = len(nums) - 1`.
2. While `left <= right`:
   - `mid = left + (right - left) // 2`.
   - If `nums[mid] == target`: return `mid`.
   - Else if `nums[mid] < target`: `left = mid + 1`.
   - Else: `right = mid - 1`.
3. Return `-1`.

---

### Visual Algorithm Walkthrough

```
nums = [-1, 0, 3, 5, 9, 12], target = 9
Index:   0  1  2  3  4   5

Step 1:
  left = 0 (-1), right = 5 (12)
  mid = 0 + (5 - 0) // 2 = 2 (nums[2] = 3)
  nums[mid] = 3 < 9 -> Discard left half [0..2]
  left = mid + 1 = 3

Step 2:
  left = 3 (5), right = 5 (12)
  mid = 3 + (5 - 3) // 2 = 4 (nums[4] = 9)
  nums[mid] = 9 == target!
  MATCH FOUND!

Return index 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Target Present
- **Input:** `nums = [-1, 0, 3, 5, 9, 12]`, `target = 9`
- **Output:** `4`

#### Example 2: Target Absent
- **Input:** `nums = [-1, 0, 3, 5, 9, 12]`, `target = 2`
- **Trace:**
  - $mid = 2$ ($3 > 2$) $\implies right = 1$
  - $mid = 0$ ($-1 < 2$) $\implies left = 1$
  - $mid = 1$ ($0 < 2$) $\implies left = 2$
  - $left > right$ ($2 > 1$) $\implies$ loop exits.
- **Output:** `-1`

#### Example 3: Single Element Match
- **Input:** `nums = [5]`, `target = 5`
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int search(std::vector<int>& nums, int target) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2; // Prevents overflow
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2; // Prevents overflow
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — At each iteration, the search interval size is divided by 2. The maximum number of comparisons is $\lfloor\log_2 N\rfloor + 1$.
- **Space Complexity:** $O(1)$ auxiliary space — Modifies only scalar index variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Closed-Interval $[L, R]$ Binary Search Template (`while (left <= right)`).
- **Trap:** The integer overflow trap: Writing `(left + right) / 2` instead of `left + (right - left) / 2`.