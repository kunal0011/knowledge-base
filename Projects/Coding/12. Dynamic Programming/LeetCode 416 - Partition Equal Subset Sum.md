---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 416: Partition Equal Subset Sum"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - knapsack
  - 0-1-knapsack
  - subset-sum
  - amazon
  - meta
  - google
---

# LeetCode 416: Partition Equal Subset Sum

**Target Companies:** Amazon, Meta, Google, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** 0/1 Knapsack / Subset Sum / Space-Optimized Dynamic Programming  

---

### Problem Statement

Given an integer array `nums`, return `true` if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of positive integers.
- **Output:** `bool` — `true` if equal partition exists, `false` otherwise.
- **Constraints:**
  - $1 \le \text{nums.length} \le 200$
  - $1 \le \text{nums}[i] \le 100$
  - Maximum possible sum of array elements $\le 200 \times 100 = 20,000$.
  - Maximum target sum $\le 10,000$.

---

### Key Idea & Intuition

1. **Reduction to 0/1 Knapsack (Subset Sum):**
   - Let $S = \sum \text{nums}$.
   - If $S$ is odd, it is mathematically impossible to divide it into two equal integer halves $\implies$ immediately return `false`.
   - If $S$ is even, our target is:
     $$\text{target} = \frac{S}{2}$$
   - The problem reduces to: **Does there exist a subset of `nums` that sums up to exactly `target`?**
   - Each number in `nums` can either be chosen or skipped once, which is the exact definition of the **0/1 Knapsack Problem**.

2. **State Definition & Transition:**
   - Let $\text{dp}[s]$ be a boolean indicating whether a subset sum of $s$ is achievable.
   - Base Case: $\text{dp}[0] = \text{true}$ (the empty subset achieves sum 0).
   - For each number $x \in \text{nums}$:
     - If we include $x$, any sum $s$ that was previously formed can now form $s + x$.
     - Transition:
       $$\text{dp}[s] = \text{dp}[s] \lor \text{dp}[s - x]$$

3. **Critical Invariant: Reverse Iteration for 1D DP:**
   - In 0/1 knapsack, each element can only be used **once**.
   - If we iterate $s$ forwards from $x$ to $\text{target}$, $\text{dp}[s - x]$ might already have used $x$ in the current outer loop step (unbounded knapsack).
   - Iterating $s$ **backwards** from $\text{target}$ down to $x$ guarantees that $\text{dp}[s - x]$ represents the state from the *previous* element, ensuring each number is used at most once.

4. **Early Termination:**
   - If $\text{dp}[\text{target}]$ becomes `true` at any point, we can terminate early and return `true`.

---

### Solution Approach (Step-by-Step)

1. **Check Parity:**
   - Compute `total_sum = sum(nums)`. If `total_sum % 2 != 0`, return `false`.
   - Set `target = total_sum // 2`.
2. **Initialize DP Table:**
   - Create a boolean array `dp` of length `target + 1` filled with `false`.
   - Set `dp[0] = true`.
3. **Iterate Numbers:**
   - For each number $x$ in `nums`:
     - If $x > \text{target}$, return `false` (single element exceeds half sum).
     - Iterate $s$ backwards from `target` down to $x$:
       - `dp[s] = dp[s] or dp[s - x]`
     - If `dp[target]` is `true`, return `true`.
4. **Return:**
   - Return `dp[target]`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 5, 11, 5]`:
- Sum $S = 22$ (even). Target $= 11$.
- Init: `dp[0] = True`, all other `dp[1..11] = False`.

```
Process x = 1:
  Iterate s from 11 down to 1:
  s = 1: dp[1] = dp[1] or dp[0] = True
  Achievable sums: {0, 1}

Process x = 5:
  Iterate s from 11 down to 5:
  s = 6: dp[6] = dp[6] or dp[1] = True
  s = 5: dp[5] = dp[5] or dp[0] = True
  Achievable sums: {0, 1, 5, 6}

Process x = 11:
  Iterate s from 11 down to 11:
  s = 11: dp[11] = dp[11] or dp[0] = True
  dp[target] is True! -> Early return True!

Subsets: [11] and [1, 5, 5] both sum to 11.
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Sum $S$ | Target $S/2$ | Achievable Subsets | Result | Explanation |
|---|---|---|---|---|---|---|
| **Standard Equal** | `[1, 5, 11, 5]` | `22` | `11` | `{1, 5, 5}` & `{11}` | `true` | Equal partition found |
| **Odd Total Sum** | `[1, 2, 3, 5]` | `11` | N/A | Impossible to split odd into equal ints | `false` | Immediate parity check exit |
| **Even Sum But No Split** | `[1, 2, 5]` | `8` | `4` | Achievable sums: `{0,1,2,3,5,6,7,8}` | `false` | Sum 4 cannot be formed |
| **Two Equal Elements** | `[2, 2]` | `4` | `2` | `{2}` & `{2}` | `true` | Trivially partitioned |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        # If total sum is odd, cannot partition into two equal integers
        if total_sum % 2 != 0:
            return False
        
        target = total_sum // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            if num > target:
                return False
            # Iterate backwards to preserve 0/1 knapsack invariant
            for s in range(target, num - 1, -1):
                dp[s] = dp[s] or dp[s - num]
            if dp[target]:
                return True
                
        return dp[target]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <bitset>

class Solution {
public:
    // Standard 1D DP Solution
    bool canPartition(std::vector<int>& nums) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        if (total_sum % 2 != 0) return false;

        int target = total_sum / 2;
        std::vector<bool> dp(target + 1, false);
        dp[0] = true;

        for (int num : nums) {
            if (num > target) return false;
            for (int s = target; s >= num; --s) {
                dp[s] = dp[s] || dp[s - num];
            }
            if (dp[target]) return true;
        }

        return dp[target];
    }

    // High-Performance Bitset Alternative: O(N * target / 64)
    bool canPartitionBitset(std::vector<int>& nums) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        if (total_sum % 2 != 0) return false;

        int target = total_sum / 2;
        std::bitset<10001> bits(1); // bit 0 is set

        for (int num : nums) {
            bits |= (bits << num);
            if (bits[target]) return true;
        }

        return bits[target];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public boolean canPartition(int[] nums) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        if (totalSum % 2 != 0) {
            return false;
        }

        int target = totalSum / 2;
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;

        for (int num : nums) {
            if (num > target) {
                return false;
            }
            // Reverse iteration to prevent element reuse
            for (int s = target; s >= num; s--) {
                dp[s] = dp[s] || dp[s - num];
            }
            if (dp[target]) {
                return true;
            }
        }

        return dp[target];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \times \text{target}) = \mathcal{O}(N \times \frac{S}{2})$  
  Where $N \le 200$ and $\text{target} \le 10^4$. Total operations $\le 200 \times 10^4 = 2 \times 10^6$, which finishes in $\approx 5$ ms. With the C++ bitset technique, bitwise word parallelism reduces this by a factor of 64 ($\approx 3 \times 10^4$ operations, $< 1$ ms).
- **Space Complexity:** $\mathcal{O}(\text{target})$  
  The 1D boolean array requires $\text{target} + 1 \le 10,001$ booleans ($\approx 10$ KB).

---

### Takeaway Pattern & Interview Traps

1. **Why Backward Iteration is Mandatory:**
   - If iterated forward ($s = num \dots \text{target}$): `dp[num]` becomes true, then `dp[2 * num]` reads `dp[num]` in the *same pass*, allowing the single element `num` to be used multiple times (Unbounded Knapsack).
   - Iterating backward guarantees that $\text{dp}[s - num]$ was computed using only items up to the $(i-1)$-th element.
2. **The 0/1 Knapsack Family:**
   - LeetCode 416: Partition Equal Subset Sum (Decision version)
   - LeetCode 494: Target Sum (Counting ways)
   - LeetCode 1049: Last Stone Weight II (Minimizing difference between two partitions)