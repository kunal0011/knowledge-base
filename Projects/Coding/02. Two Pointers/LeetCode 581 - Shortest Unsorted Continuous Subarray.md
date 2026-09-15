---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - two-pointers
  - amazon
  - google
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

**Target Companies:** Amazon, Google, Meta  
**Difficulty:** Medium  
**Topic:** Monotonic Extremes Sweep in O(N) Time and O(1) Space

---

### Problem Statement

Given an integer array `nums`, you need to find one **continuous subarray** that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order.

Return the **shortest length** of such a subarray.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        end = -1
        max_so_far = nums[0]
        
        for i in range(1, n):
            if nums[i] < max_so_far:
                end = i
            else:
                max_so_far = nums[i]
                
        start = 0
        min_so_far = nums[-1]
        for i in range(n - 2, -1, -1):
            if nums[i] > min_so_far:
                start = i
            else:
                min_so_far = nums[i]
                
        return end - start + 1 if end != -1 else 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findUnsortedSubarray(std::vector<int>& nums) {
        int n = nums.size();
        int end = -1, start = 0;
        int maxSoFar = nums[0], minSoFar = nums[n - 1];

        for (int i = 1; i < n; ++i) {
            if (nums[i] < maxSoFar) end = i;
            else maxSoFar = nums[i];
        }

        for (int i = n - 2; i >= 0; --i) {
            if (nums[i] > minSoFar) start = i;
            else minSoFar = nums[i];
        }

        return end == -1 ? 0 : end - start + 1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int end = -1, start = 0;
        int maxSoFar = nums[0], minSoFar = nums[n - 1];

        for (int i = 1; i < n; i++) {
            if (nums[i] < maxSoFar) end = i;
            else maxSoFar = nums[i];
        }

        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] > minSoFar) start = i;
            else minSoFar = nums[i];
        }

        return end == -1 ? 0 : end - start + 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Two single passes without sorting.
- **Space Complexity:** $O(1)$ auxiliary space.
