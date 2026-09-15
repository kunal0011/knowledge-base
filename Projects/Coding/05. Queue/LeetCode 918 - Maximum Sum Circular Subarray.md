---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 918: Maximum Sum Circular Subarray"
tags:
  - leetcode
  - coding
  - queue
  - dynamic-programming
  - kadane
  - amazon
  - google
---

# LeetCode 918: Maximum Sum Circular Subarray

**Target Companies:** Amazon, Microsoft, Google, Meta  
**Difficulty:** Medium  
**Topic:** Kadane's Dual Inversion / Monotonic Deque over Circular Prefix Sums  

---

### Problem Statement

Given a **circular integer array** `nums` of length $n$, return the **maximum possible sum of a non-empty subarray** of `nums`.

A **circular array** means the end of the array connects to the beginning of the array. Formally, the next element of `nums[i]` is `nums[(i + 1) % n]` and the previous element of `nums[i]` is `nums[(i - 1 + n) % n]`.

A **subarray** may only include each element of the fixed buffer `nums` at most once. Formally, for a subarray `nums[i], nums[i + 1], ..., nums[j]`, there does not exist $i \le k1, k2 \le j$ with $k1 \not\equiv k2 \pmod n$.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (maximum circular subarray sum)
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le n \le 3 \times 10^4$
  - $-3 \times 10^4 \le \text{nums}[i] \le 3 \times 10^4$

---

### Key Idea & Intuition

Any maximum circular subarray falls into one of two configurations:

```
Case 1: Standard Linear Subarray (Does NOT wrap around)
[ ... | [ max subarray ] | ... ]
Solved directly by classic Kadane's Algorithm.

Case 2: Circular Wrapped Subarray (Wraps around boundaries)
[ prefix part ] | ... [ min subarray ] ... | [ suffix part ]
```

Notice that if the optimal subarray wraps around the ends:
$$\text{Circular Sum} = \text{Total Array Sum} - \text{Minimum Subarray Sum}$$
To maximize the outer wrapped sum, we must **minimize** the contiguous middle subarray that gets excluded!

#### The All-Negative Trap:
If every number in `nums` is negative (e.g. `[-3, -2, -3]`):
- $\text{Total Sum} = -8$
- $\text{Minimum Subarray Sum} = -8$ (the entire array is taken as minimum)
- $\text{Circular Sum} = -8 - (-8) = 0$
However, the problem requires a **non-empty** subarray! Returning `0` would correspond to taking an empty subarray.
Therefore, if $\text{max\_subarray} < 0$, we must immediately return $\text{max\_subarray}$!

---

### Solution Approach (Step-by-Step)

1. Maintain running variables in a single linear pass:
   - `total_sum`: Sum of all elements.
   - `max_sum`: Maximum contiguous subarray sum seen so far.
   - `cur_max`: Current running maximum ending at current index.
   - `min_sum`: Minimum contiguous subarray sum seen so far.
   - `cur_min`: Current running minimum ending at current index.
2. For each element $x$ in `nums`:
   - `cur_max = max(x, cur_max + x)`, `max_sum = max(max_sum, cur_max)`.
   - `cur_min = min(x, cur_min + x)`, `min_sum = min(min_sum, cur_min)`.
   - `total_sum += x`.
3. Check the all-negative condition:
   - If `max_sum < 0`: return `max_sum`.
4. Otherwise, return `max(max_sum, total_sum - min_sum)`.

---

### Visual Algorithm Walkthrough

```
nums = [5, -3, 5]
Total sum = 5 + (-3) + 5 = 7

Element 5:
  cur_max = 5, max_sum = 5
  cur_min = 5, min_sum = 5

Element -3:
  cur_max = max(-3, 5 + (-3)) = 2,  max_sum = 5
  cur_min = min(-3, 5 + (-3)) = -3, min_sum = -3

Element 5:
  cur_max = max(5, 2 + 5) = 7,  max_sum = 7
  cur_min = min(5, -3 + 5) = 2, min_sum = -3

Summary:
- Max Linear Subarray: max_sum = 7 (the entire array [5, -3, 5])
- Min Linear Subarray: min_sum = -3 (the middle element [-3])
- Max Wrapped Subarray: total_sum - min_sum = 7 - (-3) = 10
  (wrapping subarray includes nums[2] and nums[0]: [5, 5])

Result: max(7, 10) = 10!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Wrapped Maximum
- **Input:** `nums = [5, -3, 5]`
- **Linear Max:** `7`
- **Wrapped Max:** `7 - (-3) = 10`
- **Output:** `10`

#### Example 2: Normal Linear Maximum Better
- **Input:** `nums = [1, -2, 3, -2]`
- **Linear Max:** `3` (subarray `[3]`)
- **Total Sum:** `0`, **Min Subarray:** `-2`
- **Wrapped Max:** `0 - (-2) = 2`
- **Output:** `3`

#### Example 3: All-Negative Edge Case
- **Input:** `nums = [-3, -2, -3]`
- **Linear Max:** `-2`
- **Total Sum:** `-8`, **Min Subarray:** `-8`
- **Wrapped Max:** `-8 - (-8) = 0` (INVALID: corresponds to empty subarray)
- **Output:** `-2`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        total_sum = 0
        cur_max = 0
        max_sum = nums[0]
        cur_min = 0
        min_sum = nums[0]
        
        for x in nums:
            total_sum += x
            
            cur_max = max(x, cur_max + x)
            max_sum = max(max_sum, cur_max)
            
            cur_min = min(x, cur_min + x)
            min_sum = min(min_sum, cur_min)
            
        # If all numbers are negative, wrapped sum would be empty subarray
        if max_sum < 0:
            return max_sum
            
        return max(max_sum, total_sum - min_sum)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxSubarraySumCircular(std::vector<int>& nums) {
        int totalSum = 0;
        int curMax = 0, maxSum = nums[0];
        int curMin = 0, minSum = nums[0];

        for (int x : nums) {
            totalSum += x;

            curMax = std::max(x, curMax + x);
            maxSum = std::max(maxSum, curMax);

            curMin = std::min(x, curMin + x);
            minSum = std::min(minSum, curMin);
        }

        if (maxSum < 0) {
            return maxSum;
        }

        return std::max(maxSum, totalSum - minSum);
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int maxSubarraySumCircular(int[] nums) {
        int totalSum = 0;
        int curMax = 0, maxSum = nums[0];
        int curMin = 0, minSum = nums[0];

        for (int x : nums) {
            totalSum += x;

            curMax = Math.max(x, curMax + x);
            maxSum = Math.max(maxSum, curMax);

            curMin = Math.min(x, curMin + x);
            minSum = Math.min(minSum, curMin);
        }

        if (maxSum < 0) {
            return maxSum;
        }

        return Math.max(maxSum, totalSum - minSum);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass through `nums` updating running max and min in constant time per element.
- **Space Complexity:** $O(1)$ auxiliary space — Only scalar accumulation variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Circular Array Inversion Trick: Transforming a wrapped outer maximum into an excluded inner minimum ($\text{total} - \text{min}$).
- **Trap:** Forgetting the all-negative edge case. When all elements are negative, `total_sum == min_sum`, resulting in `total_sum - min_sum = 0`, which represents an empty subarray (forbidden by the problem definition).