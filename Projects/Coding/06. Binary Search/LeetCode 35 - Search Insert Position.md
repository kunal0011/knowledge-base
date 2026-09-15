---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 35: Search Insert Position"
tags:
  - leetcode
  - coding
  - binary-search
  - lower-bound
  - google
  - amazon
---

# LeetCode 35: Search Insert Position

**Target Companies:** Google, Amazon, Apple, Microsoft, Bloomberg  
**Difficulty:** Easy  
**Topic:** Lower Bound Binary Search Invariant  

---

### Problem Statement

Given a sorted array of distinct integers `nums` and a target value `target`, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You must write an algorithm with $O(\log n)$ runtime complexity.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `int` (index of target or insertion slot)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-10^4 \le \text{nums}[i], \text{target} \le 10^4$
  - `nums` contains **distinct** values sorted in **ascending** order.

---

### Key Idea & Intuition

This is the canonical implementation of the **Lower Bound** problem:
$$\text{Find the smallest index } i \text{ such that } nums[i] \ge \text{target}$$

#### Loop Invariant Analysis:
- Throughout the binary search, we maintain:
  - Every element at index $< left$ is strictly $< \text{target}$.
  - Every element at index $> right$ is strictly $> \text{target}$.
- When the search terminates (`left > right`), the search interval has shrunk to zero size.
- At termination:
  - `right` points to the largest element $< \text{target}$.
  - `left` points to the smallest element $\ge \text{target}$, which is precisely the valid index where `target` should be placed!
- If `target` is greater than all elements, `left` naturally ends at `len(nums)`.
- If `target` is smaller than all elements, `left` naturally ends at `0`.

---

### Solution Approach (Step-by-Step)

1. Set `left = 0`, `right = len(nums) - 1`.
2. While `left <= right`:
   - Compute `mid = left + (right - left) // 2`.
   - If `nums[mid] == target`: return `mid`.
   - Else if `nums[mid] < target`: `left = mid + 1`.
   - Else: `right = mid - 1`.
3. If loop finishes without returning, return `left`.

---

### Visual Algorithm Walkthrough

```
nums = [1, 3, 5, 6], target = 2

Initial:
  [1,   3,   5,   6]
   ^              ^
  left           right (mid = (0 + 3) // 2 = 1, nums[1] = 3)

Iteration 1:
  nums[mid] = 3 > 2 (target)
  right = mid - 1 = 0

  [1,   3,   5,   6]
   ^
 left, right (mid = (0 + 0) // 2 = 0, nums[0] = 1)

Iteration 2:
  nums[mid] = 1 < 2 (target)
  left = mid + 1 = 1

  [1,   3,   5,   6]
   ^    ^
 right left

Termination:
  left (1) > right (0). Loop ends.
  left is 1. Target 2 should be inserted between 1 and 3 (index 1).
Return left = 1.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Target Exists in Array
- **Input:** `nums = [1, 3, 5, 6]`, `target = 5`
- **Output:** `2`

#### Example 2: Insert into Interior Gap
- **Input:** `nums = [1, 3, 5, 6]`, `target = 2`
- **Output:** `1`

#### Example 3: Insert at the End
- **Input:** `nums = [1, 3, 5, 6]`, `target = 7`
- **Output:** `4`

#### Example 4: Insert at the Beginning
- **Input:** `nums = [1, 3, 5, 6]`, `target = 0`
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
        return left
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int searchInsert(std::vector<int>& nums, int target) {
        int left = 0, right = static_cast<int>(nums.size()) - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return left;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return left;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — Halves the search space at each comparison.
- **Space Complexity:** $O(1)$ auxiliary space — Modifies and stores only pointer variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Lower Bound Search ($nums[i] \ge target$).
- **Trap:** Confusing whether to return `left` or `right` upon loop termination. Remember: when `left <= right` terminates with `left > right`, `left` has stepped onto the first element greater than or equal to `target`, making `left` the definitive insertion index.