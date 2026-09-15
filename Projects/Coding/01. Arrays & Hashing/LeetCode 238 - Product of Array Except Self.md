---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 238: Product of Array Except Self"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
---

# LeetCode 238: Product of Array Except Self

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft  
**Difficulty:** Medium  
**Topic:** Prefix & Suffix Accumulation in O(1) Auxiliary Space

---

### Problem Statement

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is guaranteed to fit in a **32-bit** integer.

You must write an algorithm that runs in **$O(n)$ time** and **without using the division operation**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[int]`
- **Constraints:**
  - $2 \le \text{nums.length} \le 10^5$
  - $-30 \le \text{nums}[i] \le 30$
  - Output fits within 32-bit signed integer.

---

### Key Idea & Intuition

- **Decomposition:**
  $$\text{ans}[i] = (\text{product of all elements before } i) \times (\text{product of all elements after } i)$$
- **Constant Auxiliary Space Trick:**
  - We use the output array `res` to hold the running prefix products during a left pass.
  - Then we sweep from right to left using a single scalar variable `suffix = 1` to multiply each position by the suffix product.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        
        # Left pass: prefix products
        prefix = 1
        for i in range(n):
            res[i] = prefix
            prefix *= nums[i]
            
        # Right pass: suffix products
        suffix = 1
        for i in range(n - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]
            
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> productExceptSelf(std::vector<int>& nums) {
        int n = nums.size();
        std::vector<int> res(n, 1);

        int prefix = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = prefix;
            prefix *= nums[i];
        }

        int suffix = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] *= suffix;
            suffix *= nums[i];
        }

        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];

        int prefix = 1;
        for (int i = 0; i < n; i++) {
            res[i] = prefix;
            prefix *= nums[i];
        }

        int suffix = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= suffix;
            suffix *= nums[i];
        }

        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Two linear passes over the array.
- **Space Complexity:** $O(1)$ auxiliary space (excluding the output array).
