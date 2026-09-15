---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 643: Maximum Average Subarray I"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - amazon
  - google
---

# LeetCode 643: Maximum Average Subarray I

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Easy  
**Topic:** Sliding Window / Fixed Window / Array  

---

### Problem Statement

You are given an integer array `nums` consisting of $n$ elements, and an integer $k$.

Find a contiguous subarray whose length is equal to $k$ that has the maximum average value and return this value. Any answer with a calculation error less than $10^{-5}$ will be accepted.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le k \le n \le 10^5$).
  - `k`: `int` ($1 \le k \le n$).
- **Output:**
  - `float` / `double` — the maximum average of any contiguous subarray of size $k$.
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le k \le n \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$

---

### Key Idea & Intuition

The average of any subarray of fixed size $k$ is given by:
$$\text{Average} = \frac{\sum_{j=i}^{i+k-1} \text{nums}[j]}{k}$$
Because the denominator $k > 0$ is constant for every candidate window, maximizing the average is mathematically equivalent to **maximizing the window sum**:
$$\max \left(\frac{\text{Sum}}{k}\right) = \frac{\max(\text{Sum})}{k}$$

By computing the window sum using integer arithmetic throughout and dividing by $k$ only once at the very end:
1. We eliminate floating-point precision loss and rounding accumulation errors during window transitions.
2. We can maintain the rolling sum in $\mathcal{O}(1)$ time per step by adding the incoming element `nums[i]` and subtracting the outgoing element `nums[i - k]`.

---

### Solution Approach (Step-by-Step)

1. Compute the sum of the first $k$ elements: `window_sum = sum(nums[:k])`.
2. Initialize `max_sum = window_sum`.
3. Slide the window from index $i = k$ to $n - 1$:
   - `window_sum += nums[i] - nums[i - k]`
   - `max_sum = max(max_sum, window_sum)`
4. Return `max_sum / k` as a floating point number.

---

### Visual Algorithm Walkthrough

For `nums = [1, 12, -5, -6, 50, 3]`, $k = 4$:

```
Initial Window [0..3]: [1, 12, -5, -6]
  window_sum = 1 + 12 - 5 - 6 = 2
  max_sum = 2

Slide 1 (i = 4, val = 50, outgoing = nums[0] = 1):
  window_sum = 2 + 50 - 1 = 51
  Window: [12, -5, -6, 50]
  max_sum = max(2, 51) = 51

Slide 2 (i = 5, val = 3, outgoing = nums[1] = 12):
  window_sum = 51 + 3 - 12 = 42
  Window: [-5, -6, 50, 3]
  max_sum = max(51, 42) = 51

End of array.
Max Average = 51 / 4 = 12.75
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 12, -5, -6, 50, 3]`, `k = 4`
- **Output:** `12.75000`

#### Example 2:
- **Input:** `nums = [5]`, `k = 1`
- **Output:** `5.00000`

#### Example 3 (Negative Elements):
- **Input:** `nums = [-1, -3, -5, -2]`, `k = 2`
- **Tracing:**
  - Window 1: $[-1, -3] \rightarrow \text{sum} = -4$
  - Window 2: $[-3, -5] \rightarrow \text{sum} = -8$
  - Window 3: $[-5, -2] \rightarrow \text{sum} = -7$
  - $\max(\text{sum}) = -4$, average = $-4 / 2 = -2.0$
- **Output:** `-2.00000`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        # Initial sum of the first k elements
        window_sum = sum(nums[:k])
        max_sum = window_sum
        
        # Slide window across remaining elements
        for i in range(k, len(nums)):
            window_sum += nums[i] - nums[i - k]
            if window_sum > max_sum:
                max_sum = window_sum
                
        return max_sum / k
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    double findMaxAverage(const std::vector<int>& nums, int k) {
        long long window_sum = 0;
        for (int i = 0; i < k; ++i) {
            window_sum += nums[i];
        }
        
        long long max_sum = window_sum;
        int n = static_cast<int>(nums.size());
        
        for (int i = k; i < n; ++i) {
            window_sum += nums[i] - nums[i - k];
            max_sum = std::max(max_sum, window_sum);
        }
        
        return static_cast<double>(max_sum) / k;
    }
};
```

#### Java 17
```java
class Solution {
    public double findMaxAverage(int[] nums, int k) {
        long windowSum = 0;
        for (int i = 0; i < k; i++) {
            windowSum += nums[i];
        }
        
        long maxSum = windowSum;
        for (int i = k; i < nums.length; i++) {
            windowSum += nums[i] - nums[i - k];
            maxSum = Math.max(maxSum, windowSum);
        }
        
        return (double) maxSum / k;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - The first $k$ elements are summed in $\mathcal{O}(k)$ time.
  - The remaining $n - k$ elements are processed with two operations (one addition, one subtraction) each in $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only scalar integers/floating point registers are used.

---

### Takeaway Pattern & Interview Traps

- **Delaying Division to Prevent Floating-Point Drift:** Always compute the running window sum as integer additions and subtractions, delaying the division by $k$ to the return statement. Doing floating-point addition/subtraction in a loop accumulates IEEE 754 precision error.
- **Integer Overflow Trap:** If $k = 10^5$ and $\text{nums}[i] = 10^4$, total sum reaches $10^9$, which fits in signed 32-bit integer (up to $2 \times 10^9$), but using `long long` in C++ and `long` in Java is good defensive practice.