---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 494: Target Sum"
tags:
  - leetcode
  - coding
  - backtracking
  - dynamic-programming
  - array
  - amazon
  - google
---

# LeetCode 494: Target Sum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Memoized DFS / Subset Sum Reduction  

---

### Problem Statement

You are given an integer array `nums` and an integer `target`.

You want to build an expression out of nums by adding one of the symbols `'+'` and `'-'` before each integer in `nums` and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1 = 1"`.

Return the number of different **expressions** that you can build, which evaluate to `target`.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `target: int`
- **Output:** `int` (number of ways to achieve `target`)
- **Constraints:**
  - $1 \le \text{nums.length} \le 20$
  - $0 \le \text{nums}[i] \le 1000$
  - $0 \le \sum \text{nums}[i] \le 1000$
  - $-1000 \le \text{target} \le 1000$

---

### Key Idea & Intuition

- **Binary Decision State Space:**
  - At every index $i \in [0, N - 1]$, there are exactly two choices: assign `+nums[i]` or `-nums[i]`.
  - A naive backtracking search visits $2^N$ leaves. For $N = 20$, $2^{20} = 1,048,576$ states.
  - Many branches arrive at the same intermediate sum at the same index (overlapping subproblems).
- **Approach 1: Top-Down DFS with Memoization:**
  - Define `dfs(idx, curr_sum)`.
  - State: `(idx, curr_sum)` $\implies$ number of valid assignments from `idx` onward to reach `target`.
  - `dfs(idx, curr_sum) = dfs(idx + 1, curr_sum + nums[idx]) + dfs(idx + 1, curr_sum - nums[idx])`.
- **Approach 2: Mathematical Reduction to 0/1 Knapsack (Subset Sum):**
  - Partition `nums` into two disjoint subsets: $P$ (elements preceded by `+`) and $M$ (elements preceded by `-`).
  - We have:
    $$P - M = \text{target}$$
    $$P + M = \text{total\_sum}$$
  - Adding the two equations:
    $$2P = \text{total\_sum} + \text{target} \implies P = \frac{\text{total\_sum} + \text{target}}{2}$$
  - **Validity Checks:**
    1. If $\text{total\_sum} < |\text{target}|$, target is unreachable $\implies 0$.
    2. If $(\text{total\_sum} + \text{target})$ is odd, $P$ cannot be an integer $\implies 0$.
  - The problem reduces to finding the number of subsets of `nums` that sum up to $P$.

---

### Solution Approach (Step-by-Step)

#### Approach 1: Top-Down Memoized Backtracking
1. Create a memoization hash map or 2D array `memo[(idx, curr_sum)]`.
2. Define `dfs(idx, curr_sum)`:
   - If `idx == len(nums)`:
     - Return $1$ if `curr_sum == target` else $0$.
   - If `(idx, curr_sum)` in `memo`:
     - Return `memo[(idx, curr_sum)]`.
   - `add_ways = dfs(idx + 1, curr_sum + nums[idx])`
   - `sub_ways = dfs(idx + 1, curr_sum - nums[idx])`
   - `memo[(idx, curr_sum)] = add_ways + sub_ways`
   - Return `memo[(idx, curr_sum)]`.
3. Call `dfs(0, 0)`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 1]`, `target = 1`.

```
                              dfs(idx=0, sum=0)
                            /                   \
                   +1 (idx=1, sum=1)       -1 (idx=1, sum=-1)
                   /               \               /        \
            +1 (2, 2)            -1 (2, 0)     +1 (2, 0)   -1 (2, -2)
            /       \            /       \        |           |
        +1(3,3)   -1(3,1)*    +1(3,1)* -1(3,-1) (Memo hit!)  ...
          x         MATCH       MATCH      x
```

At `(idx=2, sum=0)`, both the `[+1, -1]` and `[-1, +1]` prefixes merge into the same subproblem state. Memoization computes this state once and caches the result.

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `target` | Total Sum | Valid Subsets / Expressions | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 1, 1, 1, 1]` | `3` | `5` | $P = (5 + 3)/2 = 4$. Subsets summing to 4: 5 ways | `5` |
| **Zero Elements** | `[0, 0, 0, 0, 0, 0, 0, 0, 1]` | `1` | `1` | Each `0` can be `+` or `-` ($2^8 = 256$ ways) | `256` |
| **Target Impossible** | `[1, 2]` | `4` | `3` | Total sum $< 4$ | `0` |
| **Odd Parity** | `[1]` | `2` | `1` | $(1 + 2) \% 2 \ne 0$ | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List, Dict, Tuple

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        """
        Calculates number of expressions evaluating to target.
        Approach: Top-down DFS with memoization.
        """
        memo: Dict[Tuple[int, int], int] = {}

        def dfs(idx: int, current_sum: int) -> int:
            if idx == len(nums):
                return 1 if current_sum == target else 0

            state = (idx, current_sum)
            if state in memo:
                return memo[state]

            # Recurse on adding and subtracting nums[idx]
            add_branch = dfs(idx + 1, current_sum + nums[idx])
            sub_branch = dfs(idx + 1, current_sum - nums[idx])

            memo[state] = add_branch + sub_branch
            return memo[state]

        return dfs(0, 0)

    def findTargetSumWaysDP(self, nums: List[int], target: int) -> int:
        """
        Alternative: 1D DP 0/1 Knapsack subset sum.
        Time: O(N * P), Space: O(P).
        """
        total_sum = sum(nums)
        if total_sum < abs(target) or (total_sum + target) % 2 != 0:
            return 0

        p = (total_sum + target) // 2
        dp = [0] * (p + 1)
        dp[0] = 1

        for num in nums:
            for s in range(p, num - 1, -1):
                dp[s] += dp[s - num]

        return dp[p]
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <cmath>

class Solution {
public:
    int findTargetSumWays(std::vector<int>& nums, int target) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        if (total_sum < std::abs(target) || (total_sum + target) % 2 != 0) {
            return 0;
        }

        int p = (total_sum + target) / 2;
        std::vector<int> dp(p + 1, 0);
        dp[0] = 1;

        for (int num : nums) {
            for (int s = p; s >= num; --s) {
                dp[s] += dp[s - num];
            }
        }

        return dp[p];
    }
};
```

#### Java
```java
import java.util.Arrays;

class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        if (totalSum < Math.abs(target) || (totalSum + target) % 2 != 0) {
            return 0;
        }

        int p = (totalSum + target) / 2;
        int[] dp = new int[p + 1];
        dp[0] = 1;

        for (int num : nums) {
            for (int s = p; s >= num; s--) {
                dp[s] += dp[s - num];
            }
        }

        return dp[p];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Memoized DFS:** $\mathcal{O}(N \cdot S)$ where $N = \text{len(nums)}$ and $S = \sum \text{nums}[i] \le 1000$. There are at most $20 \times 2001 \approx 4 \times 10^4$ unique states.
  - **1D DP (Subset Sum):** $\mathcal{O}(N \cdot P)$ where $P = \frac{\sum \text{nums} + \text{target}}{2} \le 1000$. Operations $\le 20 \times 1000 = 2 \times 10^4$ iterations $\implies < 2 \text{ ms}$.
- **Space Complexity:**
  - **Memoized DFS:** $\mathcal{O}(N \cdot S)$ for the memoization map and recursion stack.
  - **1D DP:** $\mathcal{O}(P)$ space for the 1D DP table.

---

### Takeaway Pattern & Interview Traps

- **Mathematical Symmetry Reduction:** Expressing $+/-$ choices as positive and negative partitions $P - M = \text{target}$ transforms an exponential branching problem into a standard 0/1 Knapsack problem.
- **Zeros in the Input:** Notice that elements in `nums` can be `0`. Each `0` doubles the number of ways because $+0$ and $-0$ are distinct expression choices! The 0/1 knapsack formula naturally handles zeros when descending down to `num` ($s \ge \text{num}$).