---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1343: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - amazon
  - google
---

# LeetCode 1343: Number of Sub-arrays of Size K and Average Greater than or Equal to Threshold

**Target Companies:** Amazon, Google, Microsoft  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Sum Threshold Invariant  

---

### Problem Statement

Given an array of integers `arr` and two integers `k` and `threshold`, return *the number of sub-arrays of size `k` and average greater than or equal to `threshold`*.

---

### Input & Output Formats & Constraints

- **Input:** `arr: List[int]`, `k: int`, `threshold: int`
- **Output:** `int` (count of valid sub-arrays)
- **Constraints:**
  - $1 \le \text{arr.length} \le 10^5$
  - $1 \le \text{arr}[i] \le 10^4$
  - $1 \le k \le \text{arr.length}$
  - $0 \le \text{threshold} \le 10^4$

---

### Key Idea & Intuition

- **Avoiding Floating-Point Division:**
  - The condition $\frac{\text{sum}}{k} \ge \text{threshold}$ can be rewritten as:
    $$\text{sum} \ge k \times \text{threshold}$$
  - Computing $\text{target\_sum} = k \times \text{threshold}$ once upfront allows exact integer comparisons, eliminating all floating-point division and rounding inaccuracies.
- **Fixed-Size Sliding Window of Size $k$:**
  - Compute the sum of the first $k$ elements ($i = 0 \dots k - 1$).
  - If `curr_sum >= target_sum`, increment `count`.
  - For $i = k \dots n - 1$:
    - Add incoming element: `curr_sum += arr[i]`.
    - Subtract outgoing element: `curr_sum -= arr[i - k]`.
    - If `curr_sum >= target_sum`, increment `count`.
  - Runs in strictly $\mathcal{O}(N)$ time with $\mathcal{O}(1)$ space.

---

### Solution Approach (Step-by-Step)

1. Compute `target_sum = k * threshold`.
2. Compute `curr_sum` for the first $k$ elements (`sum(arr[:k])`).
3. Initialize `count = 1 if curr_sum >= target_sum else 0`.
4. Iterate `i` from $k$ to $\text{len}(arr) - 1$:
   - `curr_sum += arr[i] - arr[i - k]`
   - If `curr_sum >= target_sum`:
     - `count += 1`
5. Return `count`.

---

### Visual Algorithm Walkthrough

Let `arr = [2, 2, 2, 2, 5, 5, 5, 8]`, `k = 3`, `threshold = 4`.
Target Sum $= k \times \text{threshold} = 3 \times 4 = 12$.

```
Indices:    0  1  2  3  4  5  6  7
Values:    [2, 2, 2, 2, 5, 5, 5, 8]

Initial Window [0..2]:
- Elements: [2, 2, 2] -> sum = 6 < 12 (No)

Slide to [1..3]:
- Elements: [2, 2, 2] -> sum = 6 - 2 + 2 = 6 < 12 (No)

Slide to [2..4]:
- Elements: [2, 2, 5] -> sum = 6 - 2 + 5 = 9 < 12 (No)

Slide to [3..5]:
- Elements: [2, 5, 5] -> sum = 9 - 2 + 5 = 12 >= 12 (YES! count = 1)

Slide to [4..6]:
- Elements: [5, 5, 5] -> sum = 12 - 2 + 5 = 15 >= 12 (YES! count = 2)

Slide to [5..7]:
- Elements: [5, 5, 8] -> sum = 15 - 5 + 8 = 18 >= 12 (YES! count = 3)

Total Count: 3
```

---

### Solved Examples with Multiple Inputs

| Test Case | `arr` | `k` | `threshold` | Target Sum ($k \times \text{thresh}$) | Valid Windows | Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[2,2,2,2,5,5,5,8]` | `3` | `4` | 12 | `[2,5,5]`, `[5,5,5]`, `[5,5,8]` | `3` |
| **Example 2** | `[11,13,17,23,29,31,7,5,2,3]` | `3` | `5` | 15 | 6 windows exceed average 5 | `6` |
| **All Below Threshold** | `[1, 1, 1]` | `2` | `5` | 10 | Max sum is 2 | `0` |
| **k == arr.length** | `[5, 5, 5]` | `3` | `5` | 15 | Exactly 1 window with sum 15 | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
        """
        Counts subarrays of size k with average >= threshold.
        Uses fixed-size sliding window with integer threshold multiplication.
        """
        target_sum = k * threshold
        curr_sum = sum(arr[:k])
        count = 1 if curr_sum >= target_sum else 0

        for i in range(k, len(arr)):
            # Slide window: add incoming, remove outgoing
            curr_sum += arr[i] - arr[i - k]
            if curr_sum >= target_sum:
                count += 1

        return count
```

#### C++17
```cpp
#include <vector>
#include <numeric>

class Solution {
public:
    int numOfSubarrays(const std::vector<int>& arr, int k, int threshold) {
        int target_sum = k * threshold;
        int curr_sum = 0;

        for (int i = 0; i < k; ++i) {
            curr_sum += arr[i];
        }

        int count = (curr_sum >= target_sum) ? 1 : 0;

        for (size_t i = k; i < arr.size(); ++i) {
            curr_sum += arr[i] - arr[i - k];
            if (curr_sum >= target_sum) {
                count++;
            }
        }

        return count;
    }
};
```

#### Java
```java
class Solution {
    public int numOfSubarrays(int[] arr, int k, int threshold) {
        int targetSum = k * threshold;
        int currSum = 0;

        for (int i = 0; i < k; i++) {
            currSum += arr[i];
        }

        int count = (currSum >= targetSum) ? 1 : 0;

        for (int i = k; i < arr.length; i++) {
            currSum += arr[i] - arr[i - k];
            if (currSum >= targetSum) {
                count++;
            }
        }

        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{arr.length}$.
  - Computing the initial window sum takes $\mathcal{O}(k)$ time.
  - The loop runs $N - k$ times, performing constant-time $\mathcal{O}(1)$ arithmetic per step.
  - Overall time is $\mathcal{O}(N)$, running in $< 5 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space as only scalar integer variables are allocated.

---

### Takeaway Pattern & Interview Traps

- **Integer Arithmetic vs Floating-Point Division:** Comparing `sum / k >= threshold` repeatedly can introduce precision errors and is slower than multiplying `target_sum = k * threshold` once.
- **Fixed Window Update Pattern:** Updating `curr_sum += arr[i] - arr[i - k]` directly updates the running sum in 1 line of code.