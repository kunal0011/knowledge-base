---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 930: Binary Subarrays With Sum"
tags:
  - leetcode
  - coding
  - sliding-window
  - prefix-sum
  - hash-table
  - two-pointers
  - amazon
  - google
---

# LeetCode 930: Binary Subarrays With Sum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Sliding Window / Prefix Sum / Hash Table / Two Pointers  

---

### Problem Statement

Given a binary array `nums` and an integer `goal`, return the number of non-empty **subarrays** with a sum equal to `goal`.

A **subarray** is a contiguous part of the array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — array of integers containing only $0$s and $1$s ($1 \le \text{nums.length} \le 3 \times 10^4$).
  - `goal`: `int` ($0 \le goal \le \text{nums.length}$).
- **Output:**
  - `int` — count of contiguous subarrays whose elements sum to `goal`.
- **Constraints:**
  - $1 \le \text{nums.length} \le 3 \times 10^4$
  - `nums[i]` is either $0$ or $1$.
  - $0 \le goal \le \text{nums.length}$

---

### Key Idea & Intuition

#### Approach 1: Prefix Sum with Frequency Array / Hash Map ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space)
Let $P[i] = \sum_{j=0}^{i-1} \text{nums}[j]$ be the prefix sum.
The sum of subarray $\text{nums}[j \dots i - 1]$ is $P[i] - P[j]$.
We want $P[i] - P[j] = goal \iff P[j] = P[i] - goal$.
As we scan through `nums`, we query how many times $P[i] - goal$ has appeared previously in our prefix frequency map, and then record the current prefix sum.

#### Approach 2: Sliding Window $\text{atMost}(K) - \text{atMost}(K - 1)$ ($\mathcal{O}(n)$ time, $\mathcal{O}(1)$ space)
Standard sliding window cannot directly count subarrays with **exact** sum $goal$ because adding zeros to the window doesn't change the sum (non-strictly monotonic).
However, counting subarrays with **sum at most $K$** is strictly monotonic:
- Window $[l, r]$ maintains sum $\le K$.
- For each $r$, add `nums[r]` to `window_sum`.
- While `window_sum > K`, subtract `nums[l]` and increment `l`.
- The number of valid subarrays ending at $r$ is $r - l + 1$.

By the principle of inclusion-exclusion:
$$\text{exact}(goal) = \text{atMost}(goal) - \text{atMost}(goal - 1)$$
(When $goal < 0$, $\text{atMost}(goal) = 0$).

This reduces space complexity from $\mathcal{O}(n)$ to optimal $\mathcal{O}(1)$!

---

### Solution Approach (Step-by-Step: $\mathcal{O}(1)$ Space Sliding Window)

1. Define helper function `at_most(k)`:
   - If $k < 0$, return $0$.
   - Initialize `left = 0`, `curr_sum = 0`, and `count = 0`.
   - Loop `right` from $0$ to $n - 1$:
     - `curr_sum += nums[right]`
     - While `curr_sum > k`:
       - `curr_sum -= nums[left]`
       - `left += 1`
     - `count += right - left + 1`
   - Return `count`.
2. Return `at_most(goal) - at_most(goal - 1)`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 0, 1, 0, 1]`, `goal = 2`:

```
Array: [1, 0, 1, 0, 1]

Evaluate atMost(2):
r=0: [1] -> sum=1 <= 2. New: 1
r=1: [1, 0] -> sum=1 <= 2. New: 2 ([1,0], [0])
r=2: [1, 0, 1] -> sum=2 <= 2. New: 3 ([1,0,1], [0,1], [1])
r=3: [1, 0, 1, 0] -> sum=2 <= 2. New: 4
r=4: [1, 0, 1, 0, 1] -> sum=3 > 2 -> evict nums[0]=1. left=1.
     Window [1..4]: [0, 1, 0, 1], sum=2 <= 2. New: 4 - 1 + 1 = 4.
Total atMost(2) = 1 + 2 + 3 + 4 + 4 = 14.

Evaluate atMost(1):
r=0: [1] -> sum=1. New: 1
r=1: [1, 0] -> sum=1. New: 2
r=2: [1, 0, 1] -> sum=2 > 1 -> evict nums[0]=1. Window [0, 1], sum=1. New: 2
r=3: [0, 1, 0] -> sum=1. New: 3
r=4: [0, 1, 0, 1] -> sum=2 > 1 -> evict nums[1]=0, nums[2]=1. Window [0, 1], sum=1. New: 2
Total atMost(1) = 1 + 2 + 2 + 3 + 2 = 10.

Exact Sum = atMost(2) - atMost(1) = 14 - 10 = 4.
The 4 subarrays are:
  [1, 0, 1] (idx 0..2)
  [1, 0, 1, 0] (idx 0..3)
  [0, 1, 0, 1] (idx 1..4)
  [1, 0, 1] (idx 2..4)
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 0, 1, 0, 1]`, `goal = 2`
- **Output:** `4`

#### Example 2:
- **Input:** `nums = [0, 0, 0, 0, 0]`, `goal = 0`
- **Tracing:** All subarrays have sum 0. Total = $\frac{5 \times 6}{2} = 15$.
- **Output:** `15`

#### Example 3:
- **Input:** `nums = [1, 1, 1]`, `goal = 3`
- **Tracing:** Only the full array sums to 3.
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3 (Optimal $\mathcal{O}(1)$ Space Sliding Window)
```python
from typing import List

class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        def at_most(k: int) -> int:
            if k < 0:
                return 0
            
            left = 0
            curr_sum = 0
            count = 0
            
            for right, val in enumerate(nums):
                curr_sum += val
                while curr_sum > k:
                    curr_sum -= nums[left]
                    left += 1
                count += right - left + 1
                
            return count
            
        return at_most(goal) - at_most(goal - 1)
```

#### C++17 (Optimal $\mathcal{O}(1)$ Space Sliding Window)
```cpp
#include <vector>

class Solution {
public:
    int numSubarraysWithSum(const std::vector<int>& nums, int goal) {
        auto at_most = [&](int k) -> int {
            if (k < 0) return 0;
            
            int left = 0;
            int curr_sum = 0;
            int count = 0;
            int n = static_cast<int>(nums.size());
            
            for (int right = 0; right < n; ++right) {
                curr_sum += nums[right];
                while (curr_sum > k) {
                    curr_sum -= nums[left];
                    ++left;
                }
                count += (right - left + 1);
            }
            
            return count;
        };
        
        return at_most(goal) - at_most(goal - 1);
    }
};
```

#### Java 17 (Optimal $\mathcal{O}(1)$ Space Sliding Window)
```java
class Solution {
    public int numSubarraysWithSum(int[] nums, int goal) {
        return atMost(nums, goal) - atMost(nums, goal - 1);
    }
    
    private int atMost(int[] nums, int k) {
        if (k < 0) return 0;
        
        int left = 0;
        int currSum = 0;
        int count = 0;
        
        for (int right = 0; right < nums.length; right++) {
            currSum += nums[right];
            while (currSum > k) {
                currSum -= nums[left];
                left++;
            }
            count += (right - left + 1);
        }
        
        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - `at_most(goal)` and `at_most(goal - 1)` each perform a single pass over `nums`.
  - Both `left` and `right` advance monotonically at most $n$ times.
  - Total time: $2 \times \mathcal{O}(n) = \mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Unlike prefix-sum hash map approaches that require $\mathcal{O}(n)$ memory, this sliding window approach uses only a few primitive integer registers.

---

### Takeaway Pattern & Interview Traps

- **The $\text{Exact}(K) = \text{AtMost}(K) - \text{AtMost}(K - 1)$ Reduction:**
  This is the universal template for transforming exact-match non-strictly monotonic counting problems into simple, monotonic sliding windows.
  - Used in: LeetCode 930 (Binary Subarrays With Sum), LeetCode 1248 (Count Number of Nice Subarrays), LeetCode 992 (Subarrays with K Different Integers).
- **Edge Case $K < 0$:** Always guard `if k < 0: return 0` at the start of `at_most`, because when `goal == 0`, `goal - 1 == -1`, and no subarray can have a sum $\le -1$ when all numbers are non-negative.