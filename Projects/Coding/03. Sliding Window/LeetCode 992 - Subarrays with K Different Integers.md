---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 992: Subarrays with K Different Integers"
tags:
  - leetcode
  - coding
  - sliding-window
  - amazon
  - google
---

# LeetCode 992: Subarrays with K Different Integers

**Target Companies:** Amazon, Google (Signature Hard)  
**Difficulty:** Hard  
**Topic:** Sliding Window (Exact K Reduction)

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the number of **good subarrays** of `nums`.

A **good array** is an array where the number of different integers in that array is exactly `k`.
- For example, `[1,2,3,1,2]` has `3` different integers: `1`, `2`, and `3`.

A **subarray** is a contiguous part of an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` representing the count of subarrays with exactly `k` distinct integers.
- **Constraints:**
  - $1 \le 	ext{nums.length} \le 2 	imes 10^4$
  - $1 \le 	ext{nums}[i], k \le 	ext{nums.length}$

---

### Key Idea & Intuition

- **The Fundamental Challenge:**
  - Standard sliding window naturally counts subarrays satisfying monotonic conditions like "$\le K$ distinct elements" or "$\ge K$ distinct elements".
  - Finding **exactly** $K$ is non-monotonic: shrinking the window might make the distinct count valid or invalid in complex ways.
- **The "At Most" Math Trick:**
  $$	ext{Exactly}(K) = 	ext{AtMost}(K) - 	ext{AtMost}(K - 1)$$
  - Any subarray with *exactly* $K$ distinct numbers is contained in the set of subarrays with *at most* $K$ distinct numbers, minus those with *at most* $K - 1$ distinct numbers!
  - `atMost(k)` is easily computed with a standard sliding window in $O(N)$ time:
    - For each `right` pointer, shrink `left` while `len(distinct) > k`.
    - Every valid window `[left, right]` adds `right - left + 1` valid subarrays ending at `right`.

---

### Solution Approach (Step-by-Step)

1. Define a helper function `atMost(k: int) -> int`:
   - If `k <= 0`, return `0`.
   - Maintain `freq = defaultdict(int)`, `left = 0`, `count = 0`.
   - For `right` in range `len(nums)`:
     - `freq[nums[right]] += 1`.
     - While `len(freq) > k`:
       - `freq[nums[left]] -= 1`
       - If `freq[nums[left]] == 0`, delete from `freq`.
       - `left += 1`
     - Add `right - left + 1` to `count`.
   - Return `count`.
2. Return `atMost(k) - atMost(k - 1)`.

---

### Visual Algorithm Walkthrough

```
Array: [1, 2, 1, 2, 3], K = 2

Subarrays with At Most 2 distinct integers:
R=0 (1): [1]                                      -> +1 (len 1)
R=1 (2): [1,2], [2]                              -> +2 (len 2)
R=2 (1): [1,2,1], [2,1], [1]                     -> +3 (len 3)
R=3 (2): [1,2,1,2], [2,1,2], [1,2], [2]          -> +4 (len 4)
R=4 (3): Window shrinks to [2,3], [3]            -> +2 (len 2)
Total AtMost(2) = 1 + 2 + 3 + 4 + 2 = 12

Subarrays with At Most 1 distinct integer:
Total AtMost(1) = 5 (all individual elements [1], [2], [1], [2], [3])

Result Exactly(2) = 12 - 5 = 7 subarrays!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Array
- **Input:** `nums = [1,2,1,2,3], k = 2`
- **Calculation:**
  - `atMost(2)` = 12
  - `atMost(1)` = 5
  - `atMost(2) - atMost(1)` = 7
- **Output:** `7` (Subarrays: `[1,2]`, `[2,1]`, `[1,2]`, `[2,3]`, `[1,2,1]`, `[2,1,2]`, `[1,2,1,2]`)

#### Example 2: Single Element Repeated
- **Input:** `nums = [1,2,1,3,4], k = 3`
- **Output:** `3` (Subarrays: `[1,2,1,3]`, `[2,1,3]`, `[1,3,4]`)

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(target_k: int) -> int:
            if target_k <= 0:
                return 0
            freq = defaultdict(int)
            left = 0
            ans = 0
            
            for right in range(len(nums)):
                freq[nums[right]] += 1
                
                while len(freq) > target_k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        del freq[nums[left]]
                    left += 1
                    
                ans += (right - left + 1)
            return ans
            
        return atMost(k) - atMost(k - 1)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
private:
    int atMost(const std::vector<int>& nums, int k) {
        if (k <= 0) return 0;
        std::unordered_map<int, int> freq;
        int left = 0, count = 0;
        
        for (int right = 0; right < nums.size(); ++right) {
            freq[nums[right]]++;
            
            while (freq.size() > k) {
                if (--freq[nums[left]] == 0) {
                    freq.erase(nums[left]);
                }
                left++;
            }
            count += (right - left + 1);
        }
        return count;
    }

public:
    int subarraysWithKDistinct(std::vector<int>& nums, int k) {
        return atMost(nums, k) - atMost(nums, k - 1);
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    private int atMost(int[] nums, int k) {
        if (k <= 0) return 0;
        Map<Integer, Integer> freq = new HashMap<>();
        int left = 0;
        int count = 0;
        
        for (int right = 0; right < nums.length; right++) {
            freq.put(nums[right], freq.getOrDefault(nums[right], 0) + 1);
            
            while (freq.size() > k) {
                int leftVal = nums[left];
                freq.put(leftVal, freq.get(leftVal) - 1);
                if (freq.get(leftVal) == 0) {
                    freq.remove(leftVal);
                }
                left++;
            }
            count += (right - left + 1);
        }
        return count;
    }

    public int subarraysWithKDistinct(int[] nums, int k) {
        return atMost(nums, k) - atMost(nums, k - 1);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — We execute two linear sliding window passes (`atMost(k)` and `atMost(k - 1)`). Each pointer `left` and `right` advances at most $N$ times.
- **Space Complexity:** $O(K)$ — The hash map stores at most $K + 1$ unique keys at any time.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Exact $K$ problem transformed into difference of two monotonic conditions: $	ext{Exactly}(K) = 	ext{AtMost}(K) - 	ext{AtMost}(K - 1)$.
- **Alternative Problems using this Pattern:** LC 1248 (Count Number of Nice Subarrays), LC 930 (Binary Subarrays With Sum), LC 340 (Longest Substring with At Most K Distinct Characters).
