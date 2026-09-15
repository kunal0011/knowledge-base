---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 1679: Max Number of K-Sum Pairs"
tags:
  - leetcode
  - coding
  - two-pointers
  - amazon
  - google
---

# LeetCode 1679: Max Number of K-Sum Pairs

**Target Companies:** Amazon, Google  
**Difficulty:** Medium  
**Topic:** Two Pointers on Sorted Array / Hash Map

---

### Problem Statement

You are given an integer array `nums` and an integer `k`.

In one operation, you can pick two numbers from the array whose sum equals `k` and remove them from the array.

Return the **maximum number of operations** you can perform on the array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^9$
  - $1 \le k \le 10^9$

---

### Key Idea & Intuition

- **Two Pointers Invariant:**
  - Sort `nums` in ascending order.
  - Set `left = 0`, `right = len(nums) - 1`.
  - If `nums[left] + nums[right] == k`: Pair found! Increment `operations += 1`, `left += 1`, `right -= 1`.
  - If `nums[left] + nums[right] < k`: Sum is too small $\implies$ increment `left += 1`.
  - If `nums[left] + nums[right] > k`: Sum is too large $\implies$ decrement `right -= 1`.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        left, right = 0, len(nums) - 1
        ops = 0
        
        while left < right:
            total = nums[left] + nums[right]
            if total == k:
                ops += 1
                left += 1
                right -= 1
            elif total < k:
                left += 1
            else:
                right -= 1
                
        return ops
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxOperations(std::vector<int>& nums, int k) {
        std::sort(nums.begin(), nums.end());
        int left = 0, right = nums.size() - 1, ops = 0;

        while (left < right) {
            int total = nums[left] + nums[right];
            if (total == k) {
                ops++;
                left++;
                right--;
            } else if (total < k) {
                left++;
            } else {
                right--;
            }
        }
        return ops;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int maxOperations(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0, right = nums.length - 1, ops = 0;

        while (left < right) {
            int total = nums[left] + nums[right];
            if (total == k) {
                ops++;
                left++;
                right--;
            } else if (total < k) {
                left++;
            } else {
                right--;
            }
        }
        return ops;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \log N)$ due to sorting.
- **Space Complexity:** $O(1)$ auxiliary space.
