---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 658: Find K Closest Elements"
tags:
  - leetcode
  - coding
  - two-pointers
  - binary-search
  - google
  - amazon
---

# LeetCode 658: Find K Closest Elements

**Target Companies:** Google, Amazon, Meta  
**Difficulty:** Medium  
**Topic:** Binary Search on Left Boundary / Two Pointers

---

### Problem Statement

Given a sorted integer array `arr`, two integers `k` and `x`, return the `k` closest integers to `x` in the array. The result should also be sorted in ascending order.

An integer `a` is closer to `x` than an integer `b` if:
- `|a - x| < |b - x|`, or
- `|a - x| == |b - x|` and `a < b`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - k
        
        while left < right:
            mid = (left + right) // 2
            # Compare distance between arr[mid] and arr[mid + k] to x
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            else:
                right = mid
                
        return arr[left : left + k]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> findClosestElements(std::vector<int>& arr, int k, int x) {
        int left = 0, right = arr.size() - k;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return std::vector<int>(arr.begin() + left, arr.begin() + left + k);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<Integer> findClosestElements(int[] arr, int k, int x) {
        int left = 0, right = arr.length - k;

        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x - arr[mid] > arr[mid + k] - x) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        List<Integer> result = new ArrayList<>();
        for (int i = left; i < left + k; i++) {
            result.add(arr[i]);
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(\log(N - K) + K)$ — Binary search to locate window start + slice extraction.
- **Space Complexity:** $O(1)$ auxiliary space.
