---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 560: Subarray Sum Equals K"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - prefix-sum
  - amazon
  - google
---

# LeetCode 560: Subarray Sum Equals K

**Target Companies:** Google, Amazon, Meta (Top #1 Classic)  
**Difficulty:** Medium  
**Topic:** Prefix Sum + Hash Map Frequency

---

### Problem Statement

Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals to `k`.

A subarray is a contiguous non-empty sequence of elements within an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int`
- **Constraints:**
  - $1 \le \text{nums.length} \le 2 \times 10^4$
  - $-1000 \le \text{nums}[i] \le 1000$
  - $-10^7 \le k \le 10^7$

---

### Key Idea & Intuition

- **Why Sliding Window Fails:**
  - `nums` can contain **negative numbers**. The sum is non-monotonic as the window expands!
- **Prefix Sum Principle:**
  - Let $P[i]$ be the prefix sum up to index $i$: $\text{sum}(nums[0 \dots i])$.
  - The sum of subarray `nums[j \dots i]` is $P[i] - P[j - 1]$.
  - We want $P[i] - P[j - 1] = k \iff P[j - 1] = P[i] - k$.
  - Maintain a hash map of prefix sum frequencies: `prefix_counts[P] = count`.
  - Base case: `prefix_counts[0] = 1` represents a subarray starting from index 0.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1
        
        curr_sum = 0
        total_subarrays = 0
        
        for num in nums:
            curr_sum += num
            total_subarrays += prefix_counts[curr_sum - k]
            prefix_counts[curr_sum] += 1
            
        return total_subarrays
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    int subarraySum(std::vector<int>& nums, int k) {
        std::unordered_map<int, int> prefixCounts;
        prefixCounts[0] = 1;

        int currSum = 0, total = 0;
        for (int num : nums) {
            currSum += num;
            if (prefixCounts.count(currSum - k)) {
                total += prefixCounts[currSum - k];
            }
            prefixCounts[currSum]++;
        }
        return total;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixCounts = new HashMap<>();
        prefixCounts.put(0, 1);

        int currSum = 0;
        int total = 0;

        for (int num : nums) {
            currSum += num;
            total += prefixCounts.getOrDefault(currSum - k, 0);
            prefixCounts.put(currSum, prefixCounts.getOrDefault(currSum, 0) + 1);
        }

        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass through array with $O(1)$ average hash map lookups.
- **Space Complexity:** $O(N)$ for the prefix frequency map.
