---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 377: Combination Sum IV"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - unbounded-knapsack
  - permutations
  - google
  - amazon
  - meta
---

# LeetCode 377: Combination Sum IV

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Permutations with Repetition / 1D Target DP  

---

### Problem Statement

Given an array of **distinct** integers `nums` and a target integer `target`, return the number of possible combinations that add up to `target`.

The test cases are generated so that the answer can fit in a **32-bit** integer.

**Note:** Different sequences (permutations) are counted as different combinations (e.g., `(1, 2)` and `(2, 1)` are distinct).

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums: List[int]` — Array of distinct positive integers.
  - `target: int` — Desired total sum.
- **Output:**
  - `int` — Total number of ordered sequences summing to `target`.
- **Constraints:**
  - $1 \le \text{nums.length} \le 200$
  - $1 \le \text{nums}[i] \le 1000$
  - All elements of `nums` are **unique**.
  - $1 \le \text{target} \le 1000$
  - The answer is guaranteed to fit in a 32-bit signed integer.

---

### Key Idea & Intuition

1. **Permutations vs. Combinations (Loop Order Significance):**
   - Despite the name "Combination Sum IV", this problem asks for **ordered sequences** (permutations with repetition).
   - **Coin Change II (LeetCode 518) — Unordered Combinations:**
     - Outer loop: `coins`, Inner loop: `target`. Each coin is considered sequentially, so sequences like `(1, 2)` are counted, but `(2, 1)` cannot be formed afterwards.
   - **Combination Sum IV (LeetCode 377) — Ordered Permutations:**
     - Outer loop: `target`, Inner loop: `nums`. At each sum step $t$, any number from `nums` can be chosen as the last element. Hence, both `(1, 2)` and `(2, 1)` are counted.

2. **Optimal Substructure & Recurrence:**
   - Let $\text{dp}[t]$ be the number of ordered sequences summing to $t$.
   - To form sum $t$, suppose the last added number is $x \in \text{nums}$ ($x \le t$). The remaining sum is $t - x$.
   - The number of ways to end with $x$ is simply $\text{dp}[t - x]$.
   - Summing over all valid choices of $x$:
     $$\text{dp}[t] = \sum_{\substack{x \in \text{nums} \\ x \le t}} \text{dp}[t - x]$$
   - Base case: $\text{dp}[0] = 1$ (the empty set represents the single unique way to sum to 0).

3. **C++ Integer Overflow Nuance:**
   - Although the final answer $\text{dp}[\text{target}]$ fits in a 32-bit integer, intermediate subproblem states during evaluation can exceed $2^{31} - 1$ in test suites. Using `unsigned int` or `unsigned long long` in C++ prevents signed integer overflow errors.

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Table:**
   - Create an array `dp` of length `target + 1` filled with `0`.
   - Set base case `dp[0] = 1`.
2. **Double Loop:**
   - Outer loop: $t$ from $1$ to `target`.
   - Inner loop: for each $x$ in `nums`:
     - If $t \ge x$:
       - `dp[t] += dp[t - x]`.
3. **Return Result:**
   - Return `dp[target]`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 2, 3]` and `target = 4`.

```
Initialize: dp = [1, 0, 0, 0, 0]
Indices:          0  1  2  3  4

t = 1:
  x = 1: dp[1] += dp[0] = 1
  Valid sequences: (1)
  dp = [1, 1, 0, 0, 0]

t = 2:
  x = 1: dp[2] += dp[1] = 1 -> sequences: (1, 1)
  x = 2: dp[2] += dp[0] = 1 -> sequences: (2)
  Total dp[2] = 2
  dp = [1, 1, 2, 0, 0]

t = 3:
  x = 1: dp[3] += dp[2] = 2 -> (1, 1, 1), (2, 1)
  x = 2: dp[3] += dp[1] = 1 -> (1, 2)
  x = 3: dp[3] += dp[0] = 1 -> (3)
  Total dp[3] = 4
  dp = [1, 1, 2, 4, 0]

t = 4:
  x = 1: dp[4] += dp[3] = 4 -> (1,1,1,1), (2,1,1), (1,2,1), (3,1)
  x = 2: dp[4] += dp[2] = 2 -> (1,1,2), (2,2)
  x = 3: dp[4] += dp[1] = 1 -> (1,3)
  Total dp[4] = 4 + 2 + 1 = 7
  dp = [1, 1, 2, 4, 7]

Result: dp[4] = 7
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | `target` | DP Transitions (`dp[0...target]`) | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[1, 2, 3]` | `4` | `[1, 1, 2, 4, 7]` | `7` | Permutations: `(1,1,1,1),(1,1,2),(1,2,1),(2,1,1),(2,2),(1,3),(3,1)` |
| **Single Element Multiple** | `[9]` | `3` | `[1, 0, 0, 0]` | `0` | Cannot reach 3 using coins of 9 |
| **Exact Match Only** | `[5]` | `5` | `dp[0]=1, dp[5]=1` | `1` | Only sequence `(5)` |
| **All Large Coins** | `[2, 4]` | `7` | All odd target values remain `0` | `0` | Sum of even numbers is always even |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (target + 1)
        dp[0] = 1
        
        for t in range(1, target + 1):
            for num in nums:
                if t >= num:
                    dp[t] += dp[t - num]
                    
        return dp[target]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int combinationSum4(std::vector<int>& nums, int target) {
        // Use unsigned int to safely avoid signed overflow in intermediate tests
        std::vector<unsigned int> dp(target + 1, 0);
        dp[0] = 1;

        for (int t = 1; t <= target; ++t) {
            for (int num : nums) {
                if (t >= num) {
                    dp[t] += dp[t - num];
                }
            }
        }

        return static_cast<int>(dp[target]);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;

        for (int t = 1; t <= target; t++) {
            for (int num : nums) {
                if (t >= num) {
                    dp[t] += dp[t - num];
                }
            }
        }

        return dp[target];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\text{target} \times |\text{nums}|)$  
  Outer loop runs $\text{target}$ times, inner loop runs $|\text{nums}|$ times. With $\text{target} \le 1000$ and $|\text{nums}| \le 200$, total operations $\le 2 \times 10^5$, executing in $< 2$ milliseconds.
- **Space Complexity:** $\mathcal{O}(\text{target})$  
  1D array of size `target + 1`.

---

### Takeaway Pattern & Interview Traps

1. **Loop Order Dictates Permutation vs. Combination:**
   - If target is outer loop $\to$ **Permutations** (order matters).
   - If items are outer loop $\to$ **Combinations** (order does not matter).
2. **Follow-Up (Negative Numbers):**
   - What if negative numbers are allowed?
   - Negative numbers permit infinite cycles that sum to zero (e.g. $+1$ and $-1$), leading to an infinite number of combinations.
   - To make the problem well-defined with negative numbers, either negative cycles must be disallowed or a maximum sequence length must be imposed.