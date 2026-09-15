---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 611: Valid Triangle Number"
tags:
  - leetcode
  - coding
  - two-pointers
  - amazon
  - google
---

# LeetCode 611: Valid Triangle Number

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Sorted Two Pointers / Triangle Inequality Invariant

---

### Problem Statement

Given an integer array `nums`, return the number of triplets chosen from the array that can make triangles if we take them as side lengths of a triangle.

Three sides $a, b, c$ form a triangle if $a + b > c$, $a + c > b$, and $b + c > a$. When sorted ($a \le b \le c$), this reduces strictly to $a + b > c$.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        count = 0
        n = len(nums)
        
        for k in range(n - 1, 1, -1):
            left, right = 0, k - 1
            while left < right:
                if nums[left] + nums[right] > nums[k]:
                    # All pairs from left to right-1 also satisfy condition with right
                    count += (right - left)
                    right -= 1
                else:
                    left += 1
                    
        return count
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int triangleNumber(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        int count = 0, n = nums.size();

        for (int k = n - 1; k >= 2; --k) {
            int left = 0, right = k - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[k]) {
                    count += (right - left);
                    right--;
                } else {
                    left++;
                }
            }
        }
        return count;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int triangleNumber(int[] nums) {
        Arrays.sort(nums);
        int count = 0, n = nums.length;

        for (int k = n - 1; k >= 2; k--) {
            int left = 0, right = k - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[k]) {
                    count += (right - left);
                    right--;
                } else {
                    left++;
                }
            }
        }
        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N^2)$ — Sorting takes $O(N \log N)$, outer loop fixed $k$ takes $N$, inner two-pointer search takes $O(N)$.
- **Space Complexity:** $O(1)$ auxiliary space.
