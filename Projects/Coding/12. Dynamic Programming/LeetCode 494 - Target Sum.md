---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 494: Target Sum"
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

# LeetCode 494: Target Sum

**Target Companies:** Amazon, Meta, Google, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Mathematical 0/1 Knapsack Transformation / Subset Sum Count  

---

### Problem Statement

You are given an integer array `nums` and an integer `target`.

You want to build an **expression** out of nums by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1 = 1"`.

Return the number of different **expressions** that you can build, which evaluates to `target`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums: List[int]` — Array of non-negative integers.
  - `target: int` — Desired evaluation result.
- **Output:**
  - `int` — Count of valid assignment configurations.
- **Constraints:**
  - $1 \le \text{nums.length} \le 20$
  - $0 \le \text{nums}[i] \le 1000$
  - $0 \le \sum \text{nums}[i] \le 1000$
  - $-1000 \le \text{target} \le 1000$

---

### Key Idea & Intuition

1. **Algebraic Reduction to Subset Sum:**
   - Partition the elements of `nums` into two sets:
     - $P$: elements assigned `+`
     - $N$: elements assigned `-`
   - By definition:
     $$\sum P - \sum N = \text{target}$$
     $$\sum P + \sum N = \text{totalSum}$$
   - Adding the two equations together:
     $$2 \sum P = \text{target} + \text{totalSum} \implies \sum P = \frac{\text{target} + \text{totalSum}}{2}$$
   - Therefore, the problem is mathematically equivalent to:  
     **"How many subsets of `nums` sum up to $S = \frac{\text{target} + \text{totalSum}}{2}$?"**

2. **Feasibility Validation:**
   - If $\text{totalSum} + \text{target} < 0$, or $|\text{target}| > \text{totalSum}$, no valid assignment exists $\implies$ return `0`.
   - If $(\text{totalSum} + \text{target})$ is odd, dividing by 2 cannot yield an integer sum $\implies$ return `0`.

3. **Counting 0/1 Knapsack DP:**
   - Let $\text{dp}[s]$ be the number of subsets with sum $s$.
   - Base Case: $\text{dp}[0] = 1$ (the empty subset achieves sum 0).
   - For each number $x \in \text{nums}$:
     - Iterate $s$ backwards from $S$ down to $x$:
       $$\text{dp}[s] += \text{dp}[s - x]$$
   - Backwards iteration prevents reusing the same number multiple times.

---

### Solution Approach (Step-by-Step)

1. **Validate Parity & Bounds:**
   - Compute `total_sum = sum(nums)`.
   - If `abs(target) > total_sum` or `(total_sum + target) % 2 != 0`:
     - Return `0`.
   - Set `subset_target = (total_sum + target) // 2`.
2. **Initialize DP Array:**
   - Create array `dp` of length `subset_target + 1` filled with `0`.
   - Set `dp[0] = 1`.
3. **Transition:**
   - For each `num` in `nums`:
     - For $s$ from `subset_target` down to `num`:
       - `dp[s] += dp[s - num]`.
4. **Return:**
   - Return `dp[subset_target]`.

---

### Visual Algorithm Walkthrough

Suppose `nums = [1, 1, 1, 1, 1]` and `target = 3`.
- `total_sum = 5`.
- `subset_target = (5 + 3) // 2 = 4`.
- We need to count subsets that sum to `4`.

```
Initialize: dp = [1, 0, 0, 0, 0] (target sums 0 to 4)

Process num = 1 (first 1):
  s=4..1: dp[1] += dp[0] = 1
  dp = [1, 1, 0, 0, 0]

Process num = 1 (second 1):
  s=2: dp[2] += dp[1] = 1
  s=1: dp[1] += dp[0] = 2
  dp = [1, 2, 1, 0, 0]

Process num = 1 (third 1):
  s=3: dp[3] += dp[2] = 1
  s=2: dp[2] += dp[1] = 3
  s=1: dp[1] += dp[0] = 3
  dp = [1, 3, 3, 1, 0]

Process num = 1 (fourth 1):
  s=4: dp[4] += dp[3] = 1
  s=3: dp[3] += dp[2] = 4
  s=2: dp[2] += dp[1] = 6
  s=1: dp[1] += dp[0] = 4
  dp = [1, 4, 6, 4, 1]

Process num = 1 (fifth 1):
  s=4: dp[4] += dp[3] = 1 + 4 = 5
  ...

Final dp[4] = 5!
Matches combinations: choosing any 4 out of 5 ones to be '+', and 1 to be '-':
(+1+1+1+1-1 = 3), with 5 distinct choices.
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | `target` | $S = (\text{total} + \text{target})/2$ | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[1, 1, 1, 1, 1]` | `3` | $(5+3)/2 = 4$ | `5` | $\binom{5}{4} = 5$ combinations |
| **Single Match** | `[1]` | `1` | $(1+1)/2 = 1$ | `1` | `+1 = 1` |
| **Negative Target** | `[1]` | `-1` | $(1-1)/2 = 0$ | `1` | `-1 = -1` |
| **Zeros Present** | `[0, 0, 0, 0, 0]` | `0` | $(0+0)/2 = 0$ | `32` | $2^5 = 32$ sign permutations |
| **Impossible Parity** | `[1, 2]` | `2` | $(3+2)/2 = 2.5$ | `0` | Sum parity cannot match target |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        
        # Validations
        if abs(target) > total_sum or (total_sum + target) % 2 != 0:
            return 0
            
        subset_target = (total_sum + target) // 2
        
        dp = [0] * (subset_target + 1)
        dp[0] = 1
        
        for num in nums:
            # Reverse iteration to prevent multiple inclusion of the same element
            for s in range(subset_target, num - 1, -1):
                dp[s] += dp[s - num]
                
        return dp[subset_target]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <numeric>
#include <cmath>

class Solution {
public:
    int findTargetSumWays(std::vector<int>& nums, int target) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);

        if (std::abs(target) > total_sum || (total_sum + target) % 2 != 0) {
            return 0;
        }

        int subset_target = (total_sum + target) / 2;
        std::vector<int> dp(subset_target + 1, 0);
        dp[0] = 1;

        for (int num : nums) {
            for (int s = subset_target; s >= num; --s) {
                dp[s] += dp[s - num];
            }
        }

        return dp[subset_target];
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        if (Math.abs(target) > totalSum || (totalSum + target) % 2 != 0) {
            return 0;
        }

        int subsetTarget = (totalSum + target) / 2;
        int[] dp = new int[subsetTarget + 1];
        dp[0] = 1;

        for (int num : nums) {
            for (int s = subsetTarget; s >= num; s--) {
                dp[s] += dp[s - num];
            }
        }

        return dp[subsetTarget];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \times \text{subset\_target}) \le \mathcal{O}(N \times \text{totalSum})$  
  Where $N \le 20$ and $\text{subset\_target} \le 1000$. Maximum operations $\le 20 \times 1000 = 2 \times 10^4$, which runs in $< 1$ ms.
- **Space Complexity:** $\mathcal{O}(\text{subset\_target}) \le \mathcal{O}(\text{totalSum})$  
  Only a 1D array of size up to $1001$ integers is allocated.

---

### Takeaway Pattern & Interview Traps

1. **Handling Zeros:**
   - In standard knapsack, $0$-weight items can cause issues if loop bounds are mishandled.
   - When $num = 0$, the backward loop down to $0$ correctly executes: `dp[s] += dp[s - 0]`, which doubles `dp[s]`. This is precisely correct because picking $+0$ or $-0$ doubles the number of valid sign permutations!
2. **Negative Target vs. Sum Abs Bound:**
   - Remember that `target` can be negative. Check `abs(target) > total_sum` before doing `(total_sum + target) // 2`.