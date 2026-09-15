---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 162: Find Peak Element"
tags:
  - leetcode
  - coding
  - binary-search
  - slope-climbing
  - amazon
  - google
---

# LeetCode 162: Find Peak Element

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Gradient Ascent Binary Search on Unsorted Discrete Functions  

---

### Problem Statement

A peak element is an element that is strictly greater than its neighbors.

Given a **0-indexed** integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any of the peaks**.

You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in $O(\log n)$ time.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (index of any peak element)
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$
  - `nums[i] != nums[i + 1]` for all valid $i$.

---

### Key Idea & Intuition

How can binary search apply to an **unsorted** array?
Binary search does not strictly require a sorted array; it requires a **predicate that allows discarding half the search space**.

Here, the boundary conditions provide the invariant: $nums[-1] = -\infty$ and $nums[n] = -\infty$.

#### The Slope Walking Principle:
Inspect the relationship between `nums[mid]` and its right neighbor `nums[mid + 1]`:
1. **Case 1: `nums[mid] < nums[mid + 1]` (Uphill Slope to the Right)**
   - The sequence is rising as we move right.
   - Even if the sequence keeps rising monotonically all the way to index $n - 1$, since $nums[n] = -\infty$, the last element $nums[n - 1]$ would be greater than its neighbors and thus a valid peak!
   - If the sequence ceases to rise and drops anywhere between $mid + 1$ and $n - 1$, the first element where it drops is a peak.
   - Therefore, a peak is **guaranteed to exist in the right half**: `left = mid + 1`.
2. **Case 2: `nums[mid] > nums[mid + 1]` (Downhill Slope to the Right)**
   - The slope is decreasing to the right, meaning `nums[mid]` is greater than its right neighbor.
   - By symmetric reasoning against $nums[-1] = -\infty$, a peak is **guaranteed to exist at `mid` or to the left of `mid`**: `right = mid`.

When `left == right`, the search space has converged to an index that is strictly greater than both its left and right neighbors.

---

### Solution Approach (Step-by-Step)

1. Set `left = 0`, `right = len(nums) - 1`.
2. While `left < right`:
   - `mid = left + (right - left) // 2`.
   - If `nums[mid] < nums[mid + 1]`:
     - Move right: `left = mid + 1`.
   - Else:
     - Move left (preserving `mid`): `right = mid`.
3. Return `left`.

---

### Visual Algorithm Walkthrough

```
nums = [1, 2, 1, 3, 5, 6, 4]
Indices: 0  1  2  3  4  5  6

Initial:
  left = 0 (1), right = 6 (4)
  mid = 3: nums[3] = 3, nums[4] = 5
  3 < 5 -> Uphill slope! Peak guaranteed in right half.
  left = mid + 1 = 4

Iteration 2:
  left = 4 (5), right = 6 (4)
  mid = 5: nums[5] = 6, nums[6] = 4
  6 > 4 -> Downhill slope! Peak at or to left of 5.
  right = mid = 5

Iteration 3:
  left = 4 (5), right = 5 (6)
  mid = 4: nums[4] = 5, nums[5] = 6
  5 < 6 -> Uphill slope!
  left = mid + 1 = 5

Loop terminates (left == right == 5).
Peak found at index 5 (value 6 > 5 and 6 > 4).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Single Peak in Middle
- **Input:** `nums = [1, 2, 3, 1]`
- **Trace:**
  - $mid = 1$: $nums[1] = 2 < nums[2] = 3 \implies left = 2$
  - $mid = 2$: $nums[2] = 3 > nums[3] = 1 \implies right = 2$
  - Returns index `2` (value 3).
- **Output:** `2`

#### Example 2: Monotonically Increasing Array
- **Input:** `nums = [1, 2, 3, 4, 5]`
- **Trace:** Uphill at every step $\implies$ converges to index `4` (value 5 > 4 and 5 > $-\infty$).
- **Output:** `4`

#### Example 3: Monotonically Decreasing Array
- **Input:** `nums = [5, 4, 3, 2, 1]`
- **Trace:** Downhill at every step $\implies$ converges to index `0` (value 5 > $-\infty$ and 5 > 4).
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
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

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int findPeakElement(std::vector<int>& nums) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findPeakElement(int[] nums) {
        int left = 0, right = nums.length - 1;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — The search space is halved at every comparison.
- **Space Complexity:** $O(1)$ auxiliary space — Uses only index pointers.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Discrete Gradient Ascent via Slope-Following Binary Search.
- **Trap:** Array boundary when checking neighbor: Notice that because `left < right`, `mid = (left + right) / 2` satisfies `mid < right`. Therefore, `mid + 1` is strictly within array bounds ($mid + 1 \le right < n$), completely preventing any index out of bounds exception!