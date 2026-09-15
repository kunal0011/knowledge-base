---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - greedy
  - binary-search
  - binary-search-on-answer
  - array
  - amazon
  - google
---

# LeetCode 410: Split Array Largest Sum

**Target Companies:** Google (Signature Classic), Amazon, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Hard  
**Topic:** Greedy / Binary Search on Answer / Array  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, split `nums` into `k` non-empty subarrays such that the largest sum of any subarray is **minimized**.

Return the **minimized largest sum** of the split.

A **subarray** is a contiguous part of the array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 1000$).
  - `k`: `int` ($1 \le k \le \min(50, \text{nums.length})$).
- **Output:**
  - `int` — the minimum possible value of the maximum subarray sum among $k$ partitions.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $0 \le \text{nums}[i] \le 10^6$
  - $1 \le k \le \min(50, \text{nums.length})$

---

### Key Idea & Intuition

Rather than trying to partition dynamically or through DP ($\mathcal{O}(k \cdot n^2)$), we invert the question:
> Instead of finding the partition that minimizes the maximum subarray sum, **can we check if a given maximum subarray sum $S$ is achievable using at most $k$ subarrays?**

#### Monotonic Feasibility Property:
Let $f(S)$ be a predicate function returning `True` if `nums` can be split into $\le k$ subarrays such that no subarray exceeds sum $S$.
- If $f(S)$ is `True`, then for any $S' > S$, $f(S')$ is also `True` (larger allowed sum only makes it easier to pack elements into fewer subarrays).
- If $f(S)$ is `False`, then for any $S' < S$, $f(S')$ is also `False`.

Because $f(S)$ is strictly monotonic, we can **Binary Search on the Answer $S$**!

#### Search Range $[low, high]$:
- **Lower bound ($low$):** Each element must belong to some subarray. Thus, no subarray sum can be smaller than the largest individual element:
  $$low = \max(\text{nums})$$
- **Upper bound ($high$):** If $k = 1$, the entire array forms a single subarray:
  $$high = \sum \text{nums}$$

#### Greedy Subarray Partitioning Function:
For a fixed candidate maximum sum `mid`:
- Traverse `nums`, accumulating `curr_sum`.
- If `curr_sum + nums[i] > mid`:
  - We must start a new subarray: `subarrays_needed += 1`, `curr_sum = nums[i]`.
- Else:
  - `curr_sum += nums[i]`
- If `subarrays_needed <= k`, then `mid` is achievable $\rightarrow$ try smaller sums (`high = mid`).
- Else, `mid` is too small $\rightarrow$ try larger sums (`low = mid + 1`).

---

### Solution Approach (Step-by-Step)

1. Set `low = max(nums)` and `high = sum(nums)`.
2. While `low < high`:
   - `mid = low + (high - low) // 2`
   - Test feasibility with greedy check:
     - Initialize `subarrays = 1`, `curr_sum = 0`.
     - For `x` in `nums`:
       - If `curr_sum + x > mid`:
         - `subarrays += 1`
         - `curr_sum = x`
       - Else:
         - `curr_sum += x`
     - If `subarrays <= k`:
       - `high = mid` (feasible, try smaller)
     - Else:
       - `low = mid + 1` (infeasible, need larger capacity)
3. Return `low`.

---

### Visual Algorithm Walkthrough

For `nums = [7, 2, 5, 10, 8]`, `k = 2`:
- $low = \max = 10$
- $high = \sum = 32$

```
Iteration 1:
  low = 10, high = 32 -> mid = 21
  Greedy test mid = 21:
    [7, 2, 5] -> sum 14 <= 21
    Add 10 -> 14 + 10 = 24 > 21. Subarray 1: [7, 2, 5] (sum 14).
    New subarray: [10, 8] -> sum 18 <= 21. Subarray 2: [10, 8] (sum 18).
    Total subarrays = 2 <= 2 (Feasible!)
  high = 21

Iteration 2:
  low = 10, high = 21 -> mid = 15
  Greedy test mid = 15:
    [7, 2, 5] -> sum 14 <= 15.
    Add 10 -> 14 + 10 = 24 > 15. Subarray 1: [7, 2, 5] (sum 14).
    New subarray: [10].
    Add 8 -> 10 + 8 = 18 > 15. Subarray 2: [10] (sum 10).
    New subarray 3: [8] (sum 8).
    Total subarrays = 3 > 2 (Infeasible!)
  low = 15 + 1 = 16

Iteration 3:
  low = 16, high = 21 -> mid = 18
  Greedy test mid = 18:
    [7, 2, 5] -> sum 14
    [10, 8] -> sum 18
    Total subarrays = 2 <= 2 (Feasible!)
  high = 18

... Converges to low = 18.
Answer = 18 (Subarrays: [7, 2, 5] and [10, 8]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [7, 2, 5, 10, 8]`, `k = 2`
- **Output:** `18`

#### Example 2:
- **Input:** `nums = [1, 2, 3, 4, 5]`, `k = 2`
- **Tracing:** Best split: `[1, 2, 3]` (sum 6) and `[4, 5]` (sum 9) -> max is 9.
- **Output:** `9`

#### Example 3 ($k = n$):
- **Input:** `nums = [1, 4, 4]`, `k = 3`
- **Tracing:** Each number gets its own subarray. Maximum is $\max(nums) = 4$.
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        low = max(nums)
        high = sum(nums)
        
        def feasible(max_sum: int) -> bool:
            subarrays = 1
            curr_sum = 0
            for x in nums:
                if curr_sum + x > max_sum:
                    subarrays += 1
                    curr_sum = x
                else:
                    curr_sum += x
            return subarrays <= k
            
        while low < high:
            mid = low + (high - low) // 2
            if feasible(mid):
                high = mid
            else:
                low = mid + 1
                
        return low
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int splitArray(const std::vector<int>& nums, int k) {
        long long low = *std::max_element(nums.begin(), nums.end());
        long long high = std::accumulate(nums.begin(), nums.end(), 0LL);
        
        auto feasible = [&](long long max_sum) -> bool {
            int subarrays = 1;
            long long curr_sum = 0;
            for (int x : nums) {
                if (curr_sum + x > max_sum) {
                    subarrays++;
                    curr_sum = x;
                } else {
                    curr_sum += x;
                }
            }
            return subarrays <= k;
        };
        
        while (low < high) {
            long long mid = low + (high - low) / 2;
            if (feasible(mid)) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        return static_cast<int>(low);
    }
};
```

#### Java 17
```java
class Solution {
    public int splitArray(int[] nums, int k) {
        long low = 0;
        long high = 0;
        
        for (int num : nums) {
            low = Math.max(low, num);
            high += num;
        }
        
        while (low < high) {
            long mid = low + (high - low) / 2;
            if (feasible(nums, k, mid)) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        
        return (int) low;
    }
    
    private boolean feasible(int[] nums, int k, long maxSum) {
        int subarrays = 1;
        long currSum = 0;
        
        for (int x : nums) {
            if (currSum + x > maxSum) {
                subarrays++;
                currSum = x;
            } else {
                currSum += x;
            }
        }
        
        return subarrays <= k;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log(\sum \text{nums} - \max \text{nums}))$
  - The binary search range is at most $\sum \text{nums} \le 1000 \times 10^6 = 10^9$.
  - $\log_2(10^9) \approx 30$ iterations.
  - In each iteration, we greedily scan $n \le 1000$ elements.
  - Total operations $\approx 30 \times 1000 = 3 \times 10^4$ operations, running in $\approx 1\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Binary search uses only primitive integer registers.

---

### Takeaway Pattern & Interview Traps

- **"Minimax" Optimization Pattern:** Whenever a problem asks to *"minimize the maximum"* or *"maximize the minimum"* of contiguous segments, immediately evaluate **Binary Search on the Answer with a Greedy Validator**.
- **Isomorphic Problems:**
  - LeetCode 1011 (Capacity To Ship Packages Within D Days)
  - LeetCode 875 (Koko Eating Bananas)
  - LeetCode 1482 (Minimum Number of Days to Make m Bouquets)
- **Use `64-bit` for Accumulation:** When $n = 1000$ and elements are $10^6$, the sum can reach $10^9$. While fitting in signed 32-bit int, computing `mid = (low + high) / 2` with larger inputs could overflow, so `long long` in C++ and `long` in Java is safest.