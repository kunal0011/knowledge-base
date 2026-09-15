---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 53: Maximum Subarray"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - kadanes-algorithm
  - divide-and-conquer
  - amazon
  - google
  - meta
---

# LeetCode 53: Maximum Subarray

**Target Companies:** Amazon (Top #1), Microsoft, Apple, Google, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** 1D Dynamic Programming / Kadane's Algorithm / Divide and Conquer  

---

### Problem Statement

Given an integer array `nums`, find the **subarray** with the largest sum, and return *its sum*.

A **subarray** is a contiguous non-empty sequence of elements within an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers.
- **Output:** `int` — Maximum contiguous subarray sum.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$
- **Follow-up:** If you have figured out the $\mathcal{O}(n)$ solution, try coding another solution using the **divide and conquer** approach, which is more subtle.

---

### Key Idea & Intuition

1. **Kadane's Algorithm as 1D Dynamic Programming:**
   - Let $\text{dp}[i]$ be the maximum subarray sum that **ends exactly at index $i$**.
   - For index $i$, we have two choices:
     1. **Extend** the optimal subarray ending at $i - 1$: $\text{dp}[i - 1] + \text{nums}[i]$.
     2. **Start fresh** with a new subarray beginning at $i$: $\text{nums}[i]$.
   - Recurrence:
     $$\text{dp}[i] = \max(\text{nums}[i], \ \text{dp}[i - 1] + \text{nums}[i])$$
   - The global maximum subarray can end at any index:
     $$\text{max\_sum} = \max_{0 \le i < n} \text{dp}[i]$$

2. **Space Optimization ($\mathcal{O}(1)$ Space):**
   - $\text{dp}[i]$ only depends on $\text{dp}[i - 1]$.
   - We replace the array with a single rolling accumulator `curr_sum`:
     $$\text{curr\_sum} = \max(\text{nums}[i], \ \text{curr\_sum} + \text{nums}[i])$$

3. **Divide and Conquer Alternative ($\mathcal{O}(N)$ Tree / $\mathcal{O}(N \log N)$):**
   - Partition array into left and right halves. The maximum subarray either:
     - Lies entirely in the left half.
     - Lies entirely in the right half.
     - Crosses the midpoint (maximum suffix of left half + maximum prefix of right half).

---

### Solution Approach (Step-by-Step)

1. **Initialize State:**
   - `curr_sum = nums[0]`
   - `max_sum = nums[0]`
2. **Iterate from Index 1:**
   - For each number $x$ in `nums[1:]`:
     - `curr_sum = max(x, curr_sum + x)`
     - `max_sum = max(max_sum, curr_sum)`
3. **Return:**
   - Return `max_sum`.

---

### Visual Algorithm Walkthrough

For `nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]`:

```
i = 0 (val = -2):
  curr_sum = -2, max_sum = -2

i = 1 (val = 1):
  curr_sum = max(1, -2 + 1) = 1 (start fresh at 1)
  max_sum = max(-2, 1) = 1

i = 2 (val = -3):
  curr_sum = max(-3, 1 - 3) = -2 (extend)
  max_sum = 1

i = 3 (val = 4):
  curr_sum = max(4, -2 + 4) = 4 (start fresh at 4)
  max_sum = max(1, 4) = 4

i = 4 (val = -1):
  curr_sum = max(-1, 4 - 1) = 3 (extend)
  max_sum = 4

i = 5 (val = 2):
  curr_sum = max(2, 3 + 2) = 5 (extend)
  max_sum = 5

i = 6 (val = 1):
  curr_sum = max(1, 5 + 1) = 6 (extend)
  max_sum = 6

i = 7 (val = -5):
  curr_sum = max(-5, 6 - 5) = 1 (extend)
  max_sum = 6

i = 8 (val = 4):
  curr_sum = max(4, 1 + 4) = 5 (extend)
  max_sum = 6

Result: max_sum = 6 (Subarray: [4, -1, 2, 1])
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Optimal Subarray | Result | Explanation |
|---|---|---|---|---|
| **Standard** | `[-2,1,-3,4,-1,2,1,-5,4]` | `[4, -1, 2, 1]` | `6` | Peak contiguous subarray sum |
| **Single Element** | `[1]` | `[1]` | `1` | Only 1 element |
| **All Negative** | `[-5, -1, -8, -3]` | `[-1]` | `-1` | Must pick at least one element (largest negative) |
| **All Positive** | `[5, 4, 1, 7, 8]` | Entire array | `25` | Sum of all elements |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Kadane's $\mathcal{O}(1)$ Space)
```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        curr_sum = nums[0]
        max_sum = nums[0]
        
        for num in nums[1:]:
            curr_sum = max(num, curr_sum + num)
            max_sum = max(max_sum, curr_sum)
            
        return max_sum
```

#### 2. C++ (C++17 / STL — Kadane's $\mathcal{O}(1)$ Space)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxSubArray(std::vector<int>& nums) {
        int curr_sum = nums[0];
        int max_sum = nums[0];

        for (size_t i = 1; i < nums.size(); ++i) {
            curr_sum = std::max(nums[i], curr_sum + nums[i]);
            max_sum = std::max(max_sum, curr_sum);
        }

        return max_sum;
    }
};
```

#### 3. Java (Modern, Typed — Kadane's $\mathcal{O}(1)$ Space)
```java
class Solution {
    public int maxSubArray(int[] nums) {
        int currSum = nums[0];
        int maxSum = nums[0];

        for (int i = 1; i < nums.length; i++) {
            currSum = Math.max(nums[i], currSum + nums[i]);
            maxSum = Math.max(maxSum, currSum);
        }

        return maxSum;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  A single linear pass from index $1$ to $N - 1$. Each iteration performs $\mathcal{O}(1)$ additions and maximum comparisons. For $N = 10^5$, finishes in $< 3$ ms.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only two scalar accumulator variables (`curr_sum`, `max_sum`) are stored.

---

### Takeaway Pattern & Interview Traps

1. **All-Negative Numbers Trap:**
   - Initializing `max_sum = 0` is a classic bug when all numbers in the array are negative (e.g., `nums = [-3, -2, -5]`).
   - The problem statement dictates that the subarray must be non-empty (contain at least one number). If initialized to $0$, the algorithm would incorrectly return $0$ instead of $-2$. Always initialize `max_sum = nums[0]`!
2. **Kadane's Generalization:**
   - LeetCode 152: Maximum Product Subarray (Tracking both min and max).
   - LeetCode 918: Maximum Sum Circular Subarray (Comparing max subarray with total minus min subarray).