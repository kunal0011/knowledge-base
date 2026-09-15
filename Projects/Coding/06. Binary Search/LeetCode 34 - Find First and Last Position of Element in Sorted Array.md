---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 34: Find First and Last Position of Element in Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
  - lower-bound
  - upper-bound
  - google
  - amazon
---

# LeetCode 34: Find First and Last Position of Element in Sorted Array

**Target Companies:** Google, Amazon, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Boundary-Biased Binary Search (Lower & Upper Bounds)  

---

### Problem Statement

Given an array of integers `nums` sorted in non-decreasing order, find the starting and ending position of a given `target` value.

If `target` is not found in the array, return `[-1, -1]`.

You must write an algorithm with $O(\log n)$ runtime complexity.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `List[int]` (two elements `[first_idx, last_idx]`)
- **Constraints:**
  - $0 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$
  - `nums` is a non-decreasing array.
  - $-10^9 \le \text{target} \le 10^9$

---

### Key Idea & Intuition

Standard binary search stops as soon as `nums[mid] == target`. However, when duplicates exist, that occurrence could be anywhere in the target cluster.

To locate the exact boundaries in $O(\log n)$ time, we run **two distinct binary searches**:
1. **Find First (Left) Boundary:**
   - When `nums[mid] == target`, record `mid` as a candidate, but keep searching **to the left**: `right = mid - 1`.
   - When `nums[mid] < target`, move right: `left = mid + 1`.
   - When `nums[mid] > target`, move left: `right = mid - 1`.
2. **Find Last (Right) Boundary:**
   - If the left boundary did not exist, the target is not in `nums`, so return `[-1, -1]` immediately.
   - When `nums[mid] == target`, record `mid` as a candidate, but keep searching **to the right**: `left = mid + 1`.
   - When `nums[mid] < target`, move right: `left = mid + 1`.
   - When `nums[mid] > target`, move left: `right = mid - 1`.

---

### Solution Approach (Step-by-Step)

1. Implement helper `find_bound(is_first: bool) -> int`:
   - Set `left = 0`, `right = len(nums) - 1`, `bound = -1`.
   - While `left <= right`:
     - `mid = left + (right - left) // 2`.
     - If `nums[mid] == target`:
       - `bound = mid`
       - If `is_first`: `right = mid - 1` (continue left)
       - Else: `left = mid + 1` (continue right)
     - Else if `nums[mid] < target`: `left = mid + 1`.
     - Else: `right = mid - 1`.
   - Return `bound`.
2. Call `first = find_bound(True)`.
3. If `first == -1`, return `[-1, -1]`.
4. Call `last = find_bound(False)`.
5. Return `[first, last]`.

---

### Visual Algorithm Walkthrough

```
nums = [5, 7, 7, 8, 8, 10], target = 8

--- Left Boundary Search ---
left = 0, right = 5
mid = 2, nums[2] = 7 < 8 -> left = mid + 1 = 3

left = 3, right = 5
mid = 4, nums[4] = 8 == target!
  Candidate first = 4.
  Search left: right = mid - 1 = 3.

left = 3, right = 3
mid = 3, nums[3] = 8 == target!
  Candidate first = 3.
  Search left: right = mid - 1 = 2.

left (3) > right (2) -> STOP. First occurrence = 3.

--- Right Boundary Search ---
left = 3, right = 5
mid = 4, nums[4] = 8 == target!
  Candidate last = 4.
  Search right: left = mid + 1 = 5.

left = 5, right = 5
mid = 5, nums[5] = 10 > 8 -> right = mid - 1 = 4.

left (5) > right (4) -> STOP. Last occurrence = 4.

Result: [3, 4]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Target with Multiple Duplicates
- **Input:** `nums = [5, 7, 7, 8, 8, 10]`, `target = 8`
- **Output:** `[3, 4]`

#### Example 2: Target Not in Array
- **Input:** `nums = [5, 7, 7, 8, 8, 10]`, `target = 6`
- **Output:** `[-1, -1]`

#### Example 3: Empty Array
- **Input:** `nums = []`, `target = 0`
- **Output:** `[-1, -1]`

#### Example 4: Single Element Match
- **Input:** `nums = [1]`, `target = 1`
- **Output:** `[0, 0]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def find_bound(is_first: bool) -> int:
            left, right = 0, len(nums) - 1
            bound = -1
            
            while left <= right:
                mid = left + (right - left) // 2
                if nums[mid] == target:
                    bound = mid
                    if is_first:
                        right = mid - 1  # Bias left
                    else:
                        left = mid + 1   # Bias right
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            return bound
            
        first = find_bound(True)
        if first == -1:
            return [-1, -1]
        last = find_bound(False)
        return [first, last]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> searchRange(std::vector<int>& nums, int target) {
        auto findBound = [&](bool isFirst) -> int {
            int left = 0, right = static_cast<int>(nums.size()) - 1;
            int bound = -1;

            while (left <= right) {
                int mid = left + (right - left) / 2;
                if (nums[mid] == target) {
                    bound = mid;
                    if (isFirst) {
                        right = mid - 1; // Bias left
                    } else {
                        left = mid + 1;  // Bias right
                    }
                } else if (nums[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return bound;
        };

        int first = findBound(true);
        if (first == -1) return {-1, -1};
        int last = findBound(false);
        return {first, last};
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int first = findBound(nums, target, true);
        if (first == -1) {
            return new int[]{-1, -1};
        }
        int last = findBound(nums, target, false);
        return new int[]{first, last};
    }

    private int findBound(int[] nums, int target, boolean isFirst) {
        int left = 0, right = nums.length - 1;
        int bound = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                bound = mid;
                if (isFirst) {
                    right = mid - 1; // Continue searching left
                } else {
                    left = mid + 1;  // Continue searching right
                }
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return bound;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — Exactly two binary searches, each halving the search space at every step. Total time is $2 \cdot O(\log N) = O(\log N)$.
- **Space Complexity:** $O(1)$ — Only scalar index pointers.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Binary Search with Directional Bias (Lower Bound / Upper Bound).
- **Trap:** Forgetting integer overflow when computing `mid = (left + right) / 2` in C++ and Java. Always write `mid = left + (right - left) / 2`.