---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 33: Search in Rotated Sorted Array"
tags:
  - leetcode
  - coding
  - binary-search
  - amazon
  - google
---

# LeetCode 33: Search in Rotated Sorted Array

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Modified Binary Search on Rotated Invariant

---

### Problem Statement

There is an integer array `nums` sorted in ascending order (with **distinct** values).

Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index $k$ ($1 \le k < \text{nums.length}$) such that the resulting array is `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`.

Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not in `nums`.

You must write an algorithm with **$O(\log n)$ runtime complexity**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `int` (index of target or -1)
- **Constraints:**
  - $1 \le \text{nums.length} \le 5000$
  - $-10^4 \le \text{nums}[i], \text{target} \le 10^4$
  - All values of `nums` are **unique**.

---

### Key Idea & Intuition

- **The Halving Invariant:**
  - In any rotated sorted array, splitting at midpoint `mid` guarantees that **at least one half is strictly normally sorted**!
- **Testing the Sorted Half:**
  - Compare `nums[left]` and `nums[mid]`:
    - Case 1: `nums[left] <= nums[mid]` $\implies$ **Left half is sorted**.
      - Check if `target` lies within `[nums[left], nums[mid])`:
        - If yes: search left (`right = mid - 1`).
        - Otherwise: search right (`left = mid + 1`).
    - Case 2: `nums[left] > nums[mid]` $\implies$ **Right half is sorted**.
      - Check if `target` lies within `(nums[mid], nums[right]]`:
        - If yes: search right (`left = mid + 1`).
        - Otherwise: search left (`right = mid - 1`).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
                
            # Check if left half is sorted
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # Otherwise, right half is sorted
            else:
                if nums[mid] < target <= nums[right]:
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
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
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
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;

            if (nums[left] <= nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log N)$ — Classical binary search dividing the array in half at every comparison.
- **Space Complexity:** $O(1)$ — Only pointer variables.
