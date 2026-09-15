---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 213: House Robber II"
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

# LeetCode 213: House Robber II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are **arranged in a circle**. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight without alerting the police*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `nums` ($1 \le |nums| \le 100$).
- **Output:** An integer denoting the maximum money that can be robbed.
- **Constraints:**
  - `1 <= nums.length <= 100`
  - `0 <= nums[i] <= 1000`

---

### Key Idea & Intuition

#### The Circular Invariant & Decomposition
In standard House Robber (LeetCode 198), houses are arranged in a linear line.
In House Robber II, the circular layout creates a single mutual exclusion constraint:
$$\text{House } 0 \text{ and House } n - 1 \text{ cannot both be robbed}$$

Because the robber cannot rob both house $0$ and house $n - 1$, any valid robbing plan must satisfy at least one of two mutually exclusive conditions:
1. **Case 1 (Exclude Last House):**
   The robber considers houses from index $0$ to index $n - 2$.
   *(House $n - 1$ is never robbed, so house $0$ is free to be robbed or skipped)*.
2. **Case 2 (Exclude First House):**
   The robber considers houses from index $1$ to index $n - 1$.
   *(House $0$ is never robbed, so house $n - 1$ is free to be robbed or skipped)*.

#### Reusing the Linear $\mathcal{O}(1)$ Subroutine
Notice that both Case 1 and Case 2 are standard linear House Robber I problems on contiguous slices of length $n - 1$:
$$\text{Max Money} = \max\Big(\text{rob\_linear}(nums[0 \dots n - 2]), \, \text{rob\_linear}(nums[1 \dots n - 1])\Big)$$

For the base case where $n = 1$, neither slice is valid; we simply return $nums[0]$.

---

### Solution Approach (Step-by-Step)

1. **Base Case:**
   - If $n == 1$, return $nums[0]$ immediately.
2. **Linear Helper Function (`rob_linear(houses)`):**
   - Maintain rolling variables `prev2 = 0` and `prev1 = 0`.
   - For each value $x$ in `houses`:
     - $curr = \max(prev1, prev2 + x)$
     - $prev2 = prev1$
     - $prev1 = curr$
   - Return `prev1`.
3. **Combine Results:**
   - Return $\max(\text{rob\_linear}(nums[0..n-2]), \, \text{rob\_linear}(nums[1..n-1]))$.

---

### Visual Algorithm Walkthrough

#### Trace for `nums = [2, 3, 2]` ($n = 3$)
```
Circular arrangement:
  House 0 (2) is adjacent to House 1 (3) AND House 2 (2).

Decomposition:
Case 1: Range [0 .. 1] -> nums[0:2] = [2, 3]
  - x = 2: curr = max(0, 0 + 2) = 2 -> prev2 = 0, prev1 = 2
  - x = 3: curr = max(2, 0 + 3) = 3 -> prev2 = 2, prev1 = 3
  Case 1 Result = 3

Case 2: Range [1 .. 2] -> nums[1:3] = [3, 2]
  - x = 3: curr = max(0, 0 + 3) = 3 -> prev2 = 0, prev1 = 3
  - x = 2: curr = max(3, 0 + 2) = 3 -> prev2 = 3, prev1 = 3
  Case 2 Result = 3

Final Result = max(Case 1, Case 2) = max(3, 3) = 3.
(Cannot rob both 0 and 2; robbing house 1 yields 3).
```

---

### Solved Examples with Multiple Inputs

| `nums` | Subarray 1 ($0 \dots n-2$) | Subarray 2 ($1 \dots n-1$) | Max(Sub1, Sub2) | Output |
|---|---|---|---|---|
| `[2, 3, 2]` | `[2, 3]` $\to 3$ | `[3, 2]` $\to 3$ | $\max(3, 3)$ | `3` |
| `[1, 2, 3, 1]` | `[1, 2, 3]` $\to 4$ | `[2, 3, 1]` $\to 3$ | $\max(4, 3)$ | `4` |
| `[1, 2, 3]` | `[1, 2]` $\to 2$ | `[2, 3]` $\to 3$ | $\max(2, 3)$ | `3` |
| `[7]` | Base case: $n=1$ | None | Direct return | `7` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
            
        def rob_linear(houses: list[int]) -> int:
            prev2: int = 0
            prev1: int = 0
            for x in houses:
                curr = max(prev1, prev2 + x)
                prev2 = prev1
                prev1 = curr
            return prev1
            
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
private:
    int robLinear(const std::vector<int>& nums, int start, int end) {
        int prev2 = 0;
        int prev1 = 0;

        for (int i = start; i <= end; ++i) {
            int curr = std::max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }

public:
    int rob(const std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        if (n == 1) return nums[0];

        // Case 1: Exclude last house (0 to n - 2)
        // Case 2: Exclude first house (1 to n - 1)
        return std::max(robLinear(nums, 0, n - 2), robLinear(nums, 1, n - 1));
    }
};
```

#### Java 17
```java
class Solution {
    private int robLinear(int[] nums, int start, int end) {
        int prev2 = 0;
        int prev1 = 0;

        for (int i = start; i <= end; i++) {
            int curr = Math.max(prev1, prev2 + nums[i]);
            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }

    public int rob(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return nums[0];
        }

        return Math.max(robLinear(nums, 0, n - 2), robLinear(nums, 1, n - 1));
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |nums|$. We run the linear House Robber routine twice across arrays of size $N - 1$, resulting in $2(N - 1) = \mathcal{O}(N)$ operations.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Passing start/end indices in C++ and Java avoids copying subarrays, keeping memory consumption strictly $\mathcal{O}(1)$.

---

### Takeaway Pattern & Interview Traps

1. **Circular Symmetry Resolution:** The classic technique for handling circular array dependencies is breaking the loop at an arbitrary boundary (here, separating whether index $0$ is allowed or prohibited).
2. **Single-Element Base Case Trap:** If $n = 1$, slicing $nums[:-1]$ results in an empty array `[]` returning $0$ instead of $nums[0]$. Checking `if n == 1: return nums[0]` at the very beginning is mandatory.
3. **Index Ranges in C++/Java:** In C++ and Java, pass `start` and `end` indices to `robLinear` to avoid allocating new sub-vectors or copying arrays.