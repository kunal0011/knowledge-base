---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 153: Find Minimum in Rotated Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
  - rotated-sorted-array
  - amazon
  - google
---

# LeetCode 153: Find Minimum in Rotated Sorted Array

**Target Companies:** Amazon (Top Classic), Microsoft, Google, Meta, Apple  
**Difficulty:** Medium  
**Topic:** Inflection Point Binary Search / Cliff Discontinuity Detection  

---

### Problem Statement

Suppose an array of length $n$ sorted in ascending order is rotated between $1$ and $n$ times. For example, the array `nums = [0,1,2,4,5,6,7]` might become:
- `[4,5,6,7,0,1,2]` if it was rotated 4 times.
- `[0,1,2,4,5,6,7]` if it was rotated 7 times.

Notice that rotating an array `[a[0], a[1], a[2], ..., a[n-1]]` 1 time results in the array `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]`.

Given the sorted rotated array `nums` of **unique** elements, return the **minimum element** of this array.

You must write an algorithm that runs in $O(\log n)$ time.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` (rotated sorted array of unique integers)
- **Output:** `int` (minimum element)
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le n \le 5000$
  - $-5000 \le \text{nums}[i] \le 5000$
  - All integers of `nums` are **unique**.
  - `nums` is sorted and rotated between $1$ and $n$ times.

---

### Key Idea & Intuition

A rotated sorted array consists of two monotonically increasing sequences separated by a single "cliff" or inflection point:
$$\text{Part 1: } [nums[0] \dots nums[\text{pivot}-1]] \quad > \quad \text{Part 2: } [nums[\text{pivot}] \dots nums[n-1]]$$
The minimum element of the entire array is located precisely at this `pivot`!

#### Why Compare Against `nums[right]` (and NOT `nums[left]`)?
- If `nums[mid] > nums[right]`:
  - The middle element belongs to Part 1 (the elevated upper segment). The cliff and minimum element must lie strictly to the right of `mid`:
  - Update: `left = mid + 1`.
- If `nums[mid] <= nums[right]`:
  - The middle element belongs to Part 2 (the lower segment). The minimum element could be `nums[mid]` itself, or lie to the left of `mid`:
  - Update: `right = mid` (we preserve `mid` because it could be the minimum!).
- When `left == right`, the search space has converged to the single minimum element.

---

### Solution Approach (Step-by-Step)

1. Set `left = 0`, `right = len(nums) - 1`.
2. While `left < right`:
   - `mid = left + (right - left) // 2`.
   - If `nums[mid] > nums[right]`:
     - `left = mid + 1`
   - Else:
     - `right = mid`
3. Return `nums[left]`.

---

### Visual Algorithm Walkthrough

```
nums = [4, 5, 6, 7, 0, 1, 2]
Index:  0  1  2  3  4  5  6

Initial:
  left = 0 (4), right = 6 (2)
  mid = 3 (7)
  nums[mid] = 7 > nums[right] = 2 -> Cliff is in right half!
  left = mid + 1 = 4

Iteration 2:
  left = 4 (0), right = 6 (2)
  mid = 5 (1)
  nums[mid] = 1 <= nums[right] = 2 -> Right half is sorted!
  right = mid = 5

Iteration 3:
  left = 4 (0), right = 5 (1)
  mid = 4 (0)
  nums[mid] = 0 <= nums[right] = 1 -> Right half is sorted!
  right = mid = 4

Loop terminates (left == right == 4).
Minimum element is nums[4] = 0.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Rotated in Middle
- **Input:** `nums = [3, 4, 5, 1, 2]`
- **Output:** `1`

#### Example 2: Rotated Multiple Times
- **Input:** `nums = [4, 5, 6, 7, 0, 1, 2]`
- **Output:** `0`

#### Example 3: Completely Unrotated (Sorted)
- **Input:** `nums = [11, 13, 15, 17]`
- **Trace:**
  - `nums[mid] = 13 <= nums[right] = 17 \implies right = 1`
  - `nums[mid] = 11 <= nums[right] = 13 \implies right = 0`
  - Returns `nums[0] = 11`.
- **Output:** `11`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int findMin(std::vector<int>& nums) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
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
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
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

- **Time Complexity:** $O(\log N)$ — The search interval is halved at every iteration, terminating in at most $\log_2 N$ steps.
- **Space Complexity:** $O(1)$ auxiliary space — Modifies and stores only index scalar variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Inflection Point Binary Search via Monotonic Right Boundary Anchor.
- **Trap:** Using `right = mid - 1` when `nums[mid] <= nums[right]`: if `mid` is the minimum itself, setting `right = mid - 1` would exclude the answer! Keep `right = mid`.