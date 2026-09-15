---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 4: Median of Two Sorted Arrays"
tags:
  - leetcode
  - coding
  - binary-search
  - amazon
  - google
---

# LeetCode 4: Median of Two Sorted Arrays

**Target Companies:** Google (Signature Classic Hard), Amazon, Meta, Apple  
**Difficulty:** Hard  
**Topic:** Binary Search on Partition Boundary

---

### Problem Statement

Given two sorted arrays `nums1` and `nums2` of size $m$ and $n$ respectively, return the **median** of the two sorted arrays.

The overall run time complexity should be **$O(\log (m+n))$**.

---

### Input & Output Formats & Constraints

- **Input:** `nums1: List[int]`, `nums2: List[int]`
- **Output:** `float` (median value)
- **Constraints:**
  - $0 \le m \le 1000, 0 \le n \le 1000, 1 \le m + n \le 2000$
  - $-10^6 \le \text{nums1}[i], \text{nums2}[i] \le 10^6$

---

### Key Idea & Intuition

- **Binary Search on the Shorter Array:**
  - Ensure $m \le n$ (swap if needed) so binary search runs in $O(\log(\min(m, n)))$.
  - We want to partition both arrays such that the left half has $\lfloor (m + n + 1) / 2 \rfloor$ elements, and every element in the left half $\le$ every element in the right half:
    - $\text{maxLeft1} \le \text{minRight2}$ and $\text{maxLeft2} \le \text{minRight1}$.
  - If $\text{maxLeft1} > \text{minRight2}$, partition 1 is too far right $\implies$ search left (`high = i - 1`).
  - If $\text{maxLeft2} > \text{minRight1}$, partition 1 is too far left $\implies$ search right (`low = i + 1`).

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
            
        m, n = len(nums1), len(nums2)
        low, high = 0, m
        
        while low <= high:
            i = (low + high) // 2
            j = (m + n + 1) // 2 - i
            
            max_left1 = float('-inf') if i == 0 else nums1[i - 1]
            min_right1 = float('inf') if i == m else nums1[i]
            
            max_left2 = float('-inf') if j == 0 else nums2[j - 1]
            min_right2 = float('inf') if j == n else nums2[j]
            
            if max_left1 <= min_right2 and max_left2 <= min_right1:
                if (m + n) % 2 == 1:
                    return max(max_left1, max_left2)
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2.0
            elif max_left1 > min_right2:
                high = i - 1
            else:
                low = i + 1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
        if (nums1.size() > nums2.size()) return findMedianSortedArrays(nums2, nums1);

        int m = nums1.size(), n = nums2.size();
        int low = 0, high = m;

        while (low <= high) {
            int i = low + (high - low) / 2;
            int j = (m + n + 1) / 2 - i;

            int maxLeft1 = (i == 0) ? INT_MIN : nums1[i - 1];
            int minRight1 = (i == m) ? INT_MAX : nums1[i];

            int maxLeft2 = (j == 0) ? INT_MIN : nums2[j - 1];
            int minRight2 = (j == n) ? INT_MAX : nums2[j];

            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((m + n) % 2 == 1) {
                    return std::max(maxLeft1, maxLeft2);
                }
                return (std::max(maxLeft1, maxLeft2) + std::min(minRight1, minRight2)) / 2.0;
            } else if (maxLeft1 > minRight2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        return 0.0;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) return findMedianSortedArrays(nums2, nums1);

        int m = nums1.length, n = nums2.length;
        int low = 0, high = m;

        while (low <= high) {
            int i = (low + high) / 2;
            int j = (m + n + 1) / 2 - i;

            int maxLeft1 = (i == 0) ? Integer.MIN_VALUE : nums1[i - 1];
            int minRight1 = (i == m) ? Integer.MAX_VALUE : nums1[i];

            int maxLeft2 = (j == 0) ? Integer.MIN_VALUE : nums2[j - 1];
            int minRight2 = (j == n) ? Integer.MAX_VALUE : nums2[j];

            if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
                if ((m + n) % 2 == 1) {
                    return Math.max(maxLeft1, maxLeft2);
                }
                return (Math.max(maxLeft1, maxLeft2) + Math.min(minRight1, minRight2)) / 2.0;
            } else if (maxLeft1 > minRight2) {
                high = i - 1;
            } else {
                low = i + 1;
            }
        }
        return 0.0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log(\min(m, n)))$ — Binary search on partition of shorter array.
- **Space Complexity:** $O(1)$ auxiliary space.
