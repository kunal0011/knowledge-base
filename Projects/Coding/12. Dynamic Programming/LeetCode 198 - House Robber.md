---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 198: House Robber"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - array
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 198: House Robber

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `nums` ($1 \le |nums| \le 100$).
- **Output:** An integer representing the maximum money that can be robbed.
- **Constraints:**
  - `1 <= nums.length <= 100`
  - `0 <= nums[i] <= 400`

---

### Key Idea & Intuition

#### The Optimal Substructure Decision
At each house $i$, the robber faces a binary decision:
1. **Rob house $i$:** The robber collects $nums[i]$ dollars. By the adjacency constraint, the robber **cannot** have robbed house $i - 1$. The maximum money available prior to house $i$ is the optimal total up to house $i - 2$:
   $$\text{profit} = dp[i - 2] + nums[i]$$
2. **Skip house $i$:** The robber collects nothing from house $i$. The maximum money available is simply the optimal total up to house $i - 1$:
   $$\text{profit} = dp[i - 1]$$

#### Recurrence Relation
Combining both mutually exclusive options:
$$dp[i] = \max\Big(dp[i - 1], \, dp[i - 2] + nums[i]\Big)$$

#### Space Optimization to $\mathcal{O}(1)$
Notice that computing $dp[i]$ only requires values from the previous two houses ($dp[i - 1]$ and $dp[i - 2]$). We do not need an entire array: two scalar variables (`prev1` and `prev2`) are sufficient, reducing auxiliary memory to $\mathcal{O}(1)$.

---

### Solution Approach (Step-by-Step)

1. **Base Case Handling:**
   - If `len(nums) == 0`, return 0.
   - If `len(nums) == 1`, return `nums[0]`.
2. **Rolling Variables Initialization:**
   - `prev2 = 0` (represents $dp[i - 2]$)
   - `prev1 = 0` (represents $dp[i - 1]$)
3. **Iterative Update:**
   - For each number $x$ in `nums`:
     - $current = \max(prev1, prev2 + x)$
     - $prev2 = prev1$
     - $prev1 = current$
4. **Return Output:**
   - Return `prev1`.

---

### Visual Algorithm Walkthrough

#### Trace for `nums = [2, 7, 9, 3, 1]`
```
Initial: prev2 = 0, prev1 = 0

House 0 (val = 2):
  current = max(prev1=0, prev2=0 + 2) = 2
  prev2 = 0, prev1 = 2

House 1 (val = 7):
  current = max(prev1=2, prev2=0 + 7) = 7
  prev2 = 2, prev1 = 7

House 2 (val = 9):
  current = max(prev1=7, prev2=2 + 9) = max(7, 11) = 11
  prev2 = 7, prev1 = 11

House 3 (val = 3):
  current = max(prev1=11, prev2=7 + 3) = max(11, 10) = 11
  prev2 = 11, prev1 = 11

House 4 (val = 1):
  current = max(prev1=11, prev2=11 + 1) = max(11, 12) = 12
  prev2 = 11, prev1 = 12

Final Max Money Robbed: 12
Optimal Houses Robbed: House 0 (2) + House 2 (9) + House 4 (1) = 12.
```

---

### Solved Examples with Multiple Inputs

| `nums` | Decisions per House | Optimal Subset Robbed | Output |
|---|---|---|---|
| `[1, 2, 3, 1]` | House 0 (1), House 2 (3) | Houses at index 0 and 2 | `4` |
| `[2, 7, 9, 3, 1]` | $2 \to 7 \to 11 \to 11 \to 12$ | Houses at index 0, 2, and 4 | `12` |
| `[2, 1, 1, 2]` | $2 \to 2 \to 3 \to 4$ | Houses at index 0 and 3 | `4` |
| `[0]` | Single house | House 0 | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2: int = 0
        prev1: int = 0
        
        for x in nums:
            current: int = max(prev1, prev2 + x)
            prev2 = prev1
            prev1 = current
            
        return prev1
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int rob(const std::vector<int>& nums) {
        int prev2 = 0;
        int prev1 = 0;

        for (int x : nums) {
            int current = std::max(prev1, prev2 + x);
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
};
```

#### Java 17
```java
class Solution {
    public int rob(int[] nums) {
        int prev2 = 0;
        int prev1 = 0;

        for (int x : nums) {
            int current = Math.max(prev1, prev2 + x);
            prev2 = prev1;
            prev1 = current;
        }

        return prev1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |nums|$. We iterate through the array once, performing constant-time operations at each step.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only two scalar variables (`prev1`, `prev2`) are used.

---

### Takeaway Pattern & Interview Traps

1. **The "Take or Skip" Canonical Pattern:** House Robber is the foundational template for non-adjacent subset selection (appearing in Delete and Earn LC 740, Maximum Alternating Subsequence Sum LC 1911).
2. **0-Length Initialization:** Starting `prev1 = 0` and `prev2 = 0` elegantly absorbs base cases $N = 1$ and $N = 2$ without needing explicit boundary if-statements.
3. **Relation to Fibonacci:** Notice that if all $nums[i] = 1$, the state transition $dp[i] = dp[i-1] + dp[i-2]$ mirrors the Fibonacci sequence recurrence.