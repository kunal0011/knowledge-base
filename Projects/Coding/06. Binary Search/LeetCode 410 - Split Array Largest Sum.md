---
date: "2026-09-15"
type: leetcode-solution
category: "Binary Search"
folder: "06. Binary Search"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - binary-search
  - binary-search-on-answer
  - greedy
  - google
  - amazon
---

# LeetCode 410: Split Array Largest Sum

**Target Companies:** Google (Signature Classic Hard), Amazon, Meta, Microsoft  
**Difficulty:** Hard  
**Topic:** Binary Search on Answer / Greedy Monotonic Feasibility Verification  

---

### Problem Statement

Given an integer array `nums` and an integer $k$, split `nums` into $k$ non-empty subarrays such that the largest sum of any subarray is **minimized**.

Return the **minimized largest sum** of the split.

A **subarray** is a contiguous part of the array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (minimum possible value of the maximum subarray sum)
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $0 \le \text{nums}[i] \le 10^6$
  - $1 \le k \le \min(50, \text{nums.length})$

---

### Key Idea & Intuition

Rather than testing all possible $\binom{N - 1}{k - 1}$ partition boundaries (which is exponential), we observe that the answer lies in a bounded, monotonic integer domain:
- **Lower Bound (`left`):** $\max(\text{nums})$ — No subarray can have a sum smaller than the single largest element.
- **Upper Bound (`right`):** $\sum \text{nums}$ — Occurs when $k = 1$ and the entire array forms one single subarray.

#### Monotonic Predicate (`can_split(mid)`):
For any candidate maximum sum threshold $S$:
- Can we divide `nums` into $\le k$ contiguous subarrays such that each subarray has sum $\le S$?
- **Greedy Verification:** Traverse `nums` from left to right. Accumulate elements into the current subarray. When adding the next element exceeds $S$, we **must** start a new subarray.
- If the total count of subarrays required exceeds $k$, $S$ is too small (infeasible) $\implies$ increase $S$ (`left = mid + 1`).
- If the total count is $\le k$, $S$ is feasible $\implies$ try a smaller threshold to minimize (`right = mid - 1`).

---

### Solution Approach (Step-by-Step)

1. Set `left = max(nums)`, `right = sum(nums)`.
2. Define `can_split(target_sum) -> bool`:
   - Initialize `count = 1`, `current_sum = 0`.
   - For `x` in `nums`:
     - If `current_sum + x > target_sum`:
       - `count += 1`
       - `current_sum = x`
     - Else:
       - `current_sum += x`
   - Return `count <= k`.
3. While `left <= right`:
   - `mid = left + (right - left) // 2`.
   - If `can_split(mid)`:
     - `ans = mid`
     - `right = mid - 1` (search for smaller feasible max sum)
   - Else:
     - `left = mid + 1`
4. Return `ans`.

---

### Visual Algorithm Walkthrough

```
nums = [7, 2, 5, 10, 8], k = 2

Search domain: [max(nums), sum(nums)] = [10, 32]

1. mid = (10 + 32) // 2 = 21:
   Subarrays packed with sum <= 21:
   [7, 2, 5] (sum = 14)
   [10, 8]   (sum = 18)
   Total subarrays = 2 <= 2 (FEASIBLE!)
   ans = 21, right = mid - 1 = 20

2. mid = (10 + 20) // 2 = 15:
   Subarrays packed with sum <= 15:
   [7, 2, 5] (sum = 14)
   [10]      (sum = 10)
   [8]       (sum = 8)
   Total subarrays = 3 > 2 (INFEASIBLE: requires 3 parts!)
   left = mid + 1 = 16

3. mid = (16 + 20) // 2 = 18:
   Subarrays packed with sum <= 18:
   [7, 2, 5] (sum = 14)
   [10, 8]   (sum = 18)
   Total subarrays = 2 <= 2 (FEASIBLE!)
   ans = 18, right = mid - 1 = 17

4. mid = (16 + 17) // 2 = 16:
   Subarrays: [7, 2, 5] (14), [10] (10), [8] (8) -> 3 > 2 (INFEASIBLE)
   left = 17

5. mid = 17:
   Subarrays: [7, 2, 5] (14), [10] (10), [8] (8) -> 3 > 2 (INFEASIBLE)
   left = 18

Terminated (left > right). Optimal minimized largest sum = 18.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Partition ($k = 2$)
- **Input:** `nums = [7, 2, 5, 10, 8]`, `k = 2`
- **Output:** `18` (Splits: `[7, 2, 5]` and `[10, 8]`)

#### Example 2: More Partitions ($k = 3$)
- **Input:** `nums = [1, 2, 3, 4, 5]`, `k = 2`
- **Output:** `9` (Splits: `[1, 2, 3]` sum 6, `[4, 5]` sum 9)

#### Example 3: $k == \text{nums.length}$
- **Input:** `nums = [1, 4, 4]`, `k = 3`
- **Output:** `4` (Each element is its own subarray)

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def can_split(max_sum: int) -> bool:
            subarrays = 1
            current_sum = 0
            for num in nums:
                if current_sum + num > max_sum:
                    subarrays += 1
                    current_sum = num
                else:
                    current_sum += num
            return subarrays <= k
            
        left, right = max(nums), sum(nums)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2
            if can_split(mid):
                ans = mid
                right = mid - 1  # Seek smaller feasible threshold
            else:
                left = mid + 1
                
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int splitArray(std::vector<int>& nums, int k) {
        long long left = *std::max_element(nums.begin(), nums.end());
        long long right = std::accumulate(nums.begin(), nums.end(), 0LL);
        long long ans = right;

        auto canSplit = [&](long long maxSum) -> bool {
            int subarrays = 1;
            long long currentSum = 0;
            for (int num : nums) {
                if (currentSum + num > maxSum) {
                    subarrays++;
                    currentSum = num;
                } else {
                    currentSum += num;
                }
            }
            return subarrays <= k;
        };

        while (left <= right) {
            long long mid = left + (right - left) / 2;
            if (canSplit(mid)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return static_cast<int>(ans);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int splitArray(int[] nums, int k) {
        long left = 0;
        long right = 0;

        for (int num : nums) {
            left = Math.max(left, num);
            right += num;
        }

        long ans = right;

        while (left <= right) {
            long mid = left + (right - left) / 2;
            if (canSplit(nums, k, mid)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return (int) ans;
    }

    private boolean canSplit(int[] nums, int k, long maxSum) {
        int subarrays = 1;
        long currentSum = 0;

        for (int num : nums) {
            if (currentSum + num > maxSum) {
                subarrays++;
                currentSum = num;
            } else {
                currentSum += num;
            }
        }

        return subarrays <= k;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N \cdot \log(\sum \text{nums} - \max(\text{nums})))$ — The binary search performs $\log_2(\sum \text{nums})$ iterations, and each feasibility check traverses all $N$ numbers in $O(N)$ time.
- **Space Complexity:** $O(1)$ auxiliary space — Modifies only scalar threshold variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Binary Search on Answer with Greedy Monotonic Verification (The "Painter's Partition / Book Allocation" Archetype).
- **Trap:** Forgetting that `left` must start at $\max(\text{nums})$: if you start `left = 0`, `mid` can be smaller than an individual number in `nums`, causing `can_split` to fail or create single-element subarrays exceeding `max_sum`.