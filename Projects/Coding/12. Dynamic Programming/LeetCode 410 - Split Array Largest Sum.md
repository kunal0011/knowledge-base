---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 410: Split Array Largest Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - binary-search
  - greedy
  - google
  - amazon
  - meta
---

# LeetCode 410: Split Array Largest Sum

**Target Companies:** Google, Amazon, Meta, Microsoft, ByteDance, Apple  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Binary Search on Answer / Greedy Partition / Minimax  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, split `nums` into `k` non-empty contiguous subarrays such that the **largest sum among these subarrays is minimized**.

Return the **minimized largest sum** of the split.

A **subarray** is a contiguous part of the array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums: List[int]` — Array of non-negative integers.
  - `k: int` — Number of contiguous subarrays.
- **Output:**
  - `int` — Minimum possible value of the maximum subarray sum.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $0 \le \text{nums}[i] \le 10^6$
  - $1 \le k \le \min(50, \text{nums.length})$

---

### Key Idea & Intuition

1. **Minimax Optimization:**
   - We seek $\min_{\text{partitions}} \max_{1 \le p \le k} (\text{sum of partition } p)$.
   - Two fundamental paradigms solve this problem:
     1. **Binary Search on the Answer (Greedy Feasibility) — $\mathcal{O}(N \log(\sum \text{nums}))$:** Optimal for production and interview performance.
     2. **Partition Dynamic Programming — $\mathcal{O}(N^2 \cdot k)$:** The classic theoretical formulation that demonstrates optimal substructure.

2. **Paradigm 1: Binary Search on Answer (Optimal):**
   - What are the minimum and maximum bounds for the maximum subarray sum?
     - **Lower bound (`low`):** $\max(\text{nums})$ (any partition must hold at least the largest single element).
     - **Upper bound (`high`):** $\sum \text{nums}$ (one partition holding the entire array).
   - If we guess a candidate maximum sum `mid`:
     - Can we partition `nums` into $\le k$ subarrays where each sum $\le \text{mid}$?
     - **Greedy Check:** Iterate through `nums`, accumulating running sum. Whenever adding the next element exceeds `mid`, start a new subarray.
     - If the number of required subarrays is $\le k$, then `mid` is feasible; try a tighter bound (`high = mid`).
     - If required subarrays $> k$, `mid` is too small; we need a larger limit (`low = mid + 1`).

3. **Paradigm 2: Partition Dynamic Programming:**
   - Let $\text{dp}[i][j]$ be the minimum largest subarray sum splitting the prefix $\text{nums}[0 \dots i-1]$ into $j$ subarrays.
   - Transition:
     $$\text{dp}[i][j] = \min_{j - 1 \le x < i} \max\left( \text{dp}[x][j - 1], \sum_{m=x}^{i-1} \text{nums}[m] \right)$$
   - Using prefix sums: $\sum_{m=x}^{i-1} \text{nums}[m] = \text{prefix}[i] - \text{prefix}[x]$.
   - Base case: $\text{dp}[i][1] = \text{prefix}[i]$.

---

### Solution Approach (Step-by-Step)

#### Approach 1: Binary Search on Answer (Recommended)
1. Set `low = max(nums)` and `high = sum(nums)`.
2. While `low < high`:
   - `mid = low + (high - low) // 2`.
   - Greedily count required subarrays:
     - `count = 1, current_sum = 0`.
     - For `x` in `nums`:
       - If `current_sum + x > mid`:
         - `count += 1, current_sum = x`.
       - Else:
         - `current_sum += x`.
   - If `count <= k`:
     - `high = mid`.
   - Else:
     - `low = mid + 1`.
3. Return `low`.

---

### Visual Algorithm Walkthrough

For `nums = [7, 2, 5, 10, 8]` and $k = 2$:
- `low = max(nums) = 10`
- `high = sum(nums) = 32`

```
Iteration 1:
  mid = 10 + (32 - 10) / 2 = 21
  Greedy test with limit 21:
    [7, 2, 5] -> sum 14 <= 21
    + 10 = 24 > 21 -> split! Subarray 1: [7, 2, 5] (sum 14)
    [10, 8] -> sum 18 <= 21
    Total pieces needed = 2 <= k (2) -> Feasible!
  high = 21

Iteration 2:
  mid = 10 + (21 - 10) / 2 = 15
  Greedy test with limit 15:
    [7, 2, 5] -> sum 14 <= 15
    + 10 = 24 > 15 -> split! Subarray 1: [7, 2, 5] (sum 14)
    [10] <= 15, + 8 = 18 > 15 -> split! Subarray 2: [10] (sum 10)
    Subarray 3: [8] (sum 8)
    Total pieces needed = 3 > k (2) -> Infeasible (exceeds k=2)!
  low = 16

Iteration 3:
  mid = 16 + (21 - 16) / 2 = 18
  Greedy test with limit 18:
    [7, 2, 5] -> sum 14 <= 18
    + 10 = 24 > 18 -> split! Subarray 1: [7, 2, 5] (sum 14)
    [10, 8] -> sum 18 <= 18 (Subarray 2)
    Total pieces needed = 2 <= k (2) -> Feasible!
  high = 18

... Converges to low = 18.
Optimal Split: [7, 2, 5] and [10, 8], max subarray sum = 18.
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | `k` | Optimal Partition | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[7, 2, 5, 10, 8]` | `2` | `[7, 2, 5]` (14) & `[10, 8]` (18) | `18` | Minimized maximum sum |
| **All Equal Single Element** | `[1, 2, 3, 4, 5]` | `2` | `[1, 2, 3]` (6) & `[4, 5]` (9) | `9` | Equal or closest split |
| **$k == 1$** | `[1, 4, 4]` | `1` | `[1, 4, 4]` (9) | `9` | Entire array is single partition |
| **$k == n$** | `[1, 4, 4]` | `3` | `[1]`, `[4]`, `[4]` | `4` | Max element alone |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        low = max(nums)
        high = sum(nums)
        
        def can_split(max_sum: int) -> bool:
            subarrays = 1
            curr_sum = 0
            for num in nums:
                if curr_sum + num > max_sum:
                    subarrays += 1
                    curr_sum = num
                else:
                    curr_sum += num
            return subarrays <= k

        while low < high:
            mid = low + (high - low) // 2
            if can_split(mid):
                high = mid
            else:
                low = mid + 1
                
        return low
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int splitArray(std::vector<int>& nums, int k) {
        long long low = *std::max_element(nums.begin(), nums.end());
        long long high = std::accumulate(nums.begin(), nums.end(), 0LL);

        auto can_split = [&](long long max_sum) -> bool {
            int subarrays = 1;
            long long curr_sum = 0;
            for (int num : nums) {
                if (curr_sum + num > max_sum) {
                    subarrays++;
                    curr_sum = num;
                } else {
                    curr_sum += num;
                }
            }
            return subarrays <= k;
        };

        while (low < high) {
            long long mid = low + (high - low) / 2;
            if (can_split(mid)) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        return static_cast<int>(low);
    }
};
```

#### 3. Java (Modern, Typed)
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
            if (canSplit(nums, k, mid)) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        return (int) low;
    }

    private boolean canSplit(int[] nums, int k, long maxSum) {
        int subarrays = 1;
        long currSum = 0;

        for (int num : nums) {
            if (currSum + num > maxSum) {
                subarrays++;
                currSum = num;
            } else {
                currSum += num;
            }
        }

        return subarrays <= k;
    }
}
```

---

### Complexity Analysis

- **Binary Search on Answer (Optimal):**
  - **Time Complexity:** $\mathcal{O}(N \log(\sum \text{nums} - \max(\text{nums})))$  
    Each feasibility check takes $\mathcal{O}(N)$ linear time. The search range is at most $1000 \times 10^6 = 10^9$, requiring at most $\approx 30$ binary search iterations. Total operations $\approx 30 \times 1000 = 3 \times 10^4$ ($< 2$ ms).
  - **Space Complexity:** $\mathcal{O}(1)$ auxiliary space.
- **Partition DP (Theoretical):**
  - **Time Complexity:** $\mathcal{O}(N^2 \cdot k)$
  - **Space Complexity:** $\mathcal{O}(N \cdot k)$

---

### Takeaway Pattern & Interview Traps

1. **Recognizing "Minimize the Maximum":**
   - The phrase "minimize the largest" or "maximize the minimum" is a hallmark signature of **Binary Search on the Answer**. When contiguous monotonicity holds, testing feasibility in $\mathcal{O}(N)$ transforms an otherwise $\mathcal{O}(N^2 k)$ DP into an ultra-fast $\mathcal{O}(N \log S)$ solution.
2. **Same Family of Problems:**
   - LeetCode 1011 (Capacity To Ship Packages Within D Days)
   - LeetCode 875 (Koko Eating Bananas)
   - LeetCode 1482 (Minimum Number of Days to Make m Bouquets)
3. **Integer Overflow in Sum:**
   - When calculating `high = sum(nums)`, total sum can reach $1000 \times 10^6 = 10^9$. In C++ and Java, use `long long` / `long` to avoid 32-bit signed integer overflow during `low + high`.