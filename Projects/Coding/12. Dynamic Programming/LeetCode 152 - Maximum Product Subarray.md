---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 152: Maximum Product Subarray"
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

# LeetCode 152: Maximum Product Subarray

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, LinkedIn  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Array  

---

### Problem Statement

Given an integer array `nums`, find a subarray that has the largest product, and return *the product*.

The test cases are generated so that the answer will fit in a **32-bit** integer.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `nums` ($1 \le |nums| \le 2 \times 10^4$).
- **Output:** An integer representing the maximum product of any contiguous subarray.
- **Constraints:**
  - `1 <= nums.length <= 2 * 10^4`
  - `-10 <= nums[i] <= 10`
  - The product of any prefix or suffix of `nums` is guaranteed to fit in a **32-bit integer**.

---

### Key Idea & Intuition

#### Why Kadane's Algorithm for Sum Fails for Products
In the Maximum Subarray Sum problem (LeetCode 53), adding numbers preserves monotonicity: adding a negative number always decreases the sum.
For products, signs are non-monotonic:
- Multiplying two negative numbers produces a **positive number** ($\text{negative} \times \text{negative} = \text{positive}$).
- A large negative product from an earlier prefix can suddenly become a massive positive product when multiplied by a negative current element!

#### Tracking Both Max and Min Products
At any index $i$, the maximum product ending at $i$ can originate from three possible sources:
1. **Starting Fresh:** The element $nums[i]$ alone.
2. **Extending Previous Max:** $nums[i] \times \text{prev\_max}$.
3. **Extending Previous Min:** $nums[i] \times \text{prev\_min}$ (if $nums[i] < 0$ and $\text{prev\_min} < 0$).

Therefore, we must simultaneously track **both** the maximum product and the minimum product ending at the current position:
$$\text{cur\_max} = \max\Big(nums[i], \, nums[i] \times \text{prev\_max}, \, nums[i] \times \text{prev\_min}\Big)$$
$$\text{cur\_min} = \min\Big(nums[i], \, nums[i] \times \text{prev\_max}, \, nums[i] \times \text{prev\_min}\Big)$$

#### The Negative Swap Trick
Notice that when $nums[i] < 0$, multiplying by $nums[i]$ reverses the relative order of values: the maximum becomes the minimum, and the minimum becomes the maximum.
Hence, whenever $nums[i] < 0$, we can simply swap `cur_max` and `cur_min` before applying the update, simplifying the code to:
```python
if nums[i] < 0:
    cur_max, cur_min = cur_min, cur_max
cur_max = max(nums[i], cur_max * nums[i])
cur_min = min(nums[i], cur_min * nums[i])
```

---

### Solution Approach (Step-by-Step)

1. **State Initialization:**
   - `cur_max = nums[0]`
   - `cur_min = nums[0]`
   - `global_max = nums[0]`
2. **Linear Traversal:**
   - Iterate through elements $x$ in `nums[1:]`:
     - If $x < 0$, swap `cur_max` and `cur_min`.
     - Update `cur_max = max(x, cur_max * x)`.
     - Update `cur_min = min(x, cur_min * x)`.
     - Update `global_max = max(global_max, cur_max)`.
3. **Return:**
   - Return `global_max`.

---

### Visual Algorithm Walkthrough

#### Trace for `nums = [2, 3, -2, 4]`
```
Initial: cur_max = 2, cur_min = 2, global_max = 2

i = 1 (x = 3, positive):
  cur_max = max(3, 2 * 3) = 6
  cur_min = min(3, 2 * 3) = 3
  global_max = max(2, 6) = 6

i = 2 (x = -2, negative -> SWAP cur_max and cur_min):
  Before swap: cur_max = 6, cur_min = 3
  After swap:  cur_max = 3, cur_min = 6
  cur_max = max(-2, 3 * -2) = max(-2, -6) = -2
  cur_min = min(-2, 6 * -2) = min(-2, -12) = -12
  global_max = max(6, -2) = 6

i = 3 (x = 4, positive):
  cur_max = max(4, -2 * 4) = max(4, -8) = 4
  cur_min = min(4, -12 * 4) = min(4, -48) = -48
  global_max = max(6, 4) = 6

Result: global_max = 6 (Subarray [2, 3]).
```

---

### Solved Examples with Multiple Inputs

| `nums` | Element Sequence | `cur_max` Evolution | `cur_min` Evolution | Output |
|---|---|---|---|---|
| `[2, 3, -2, 4]` | $2 \to 3 \to -2 \to 4$ | $2 \to 6 \to -2 \to 4$ | $2 \to 3 \to -12 \to -48$ | `6` |
| `[-2, 0, -1]` | $-2 \to 0 \to -1$ | $-2 \to 0 \to 0$ | $-2 \to 0 \to -1$ | `0` |
| `[-2, 3, -4]` | $-2 \to 3 \to -4$ | $-2 \to 3 \to 24$ | $-2 \to -6 \to -12$ | `24` |
| `[-2]` | $-2$ | $-2$ | $-2$ | `-2` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        cur_max: int = nums[0]
        cur_min: int = nums[0]
        global_max: int = nums[0]
        
        for x in nums[1:]:
            # If current number is negative, min product * x could become new max
            if x < 0:
                cur_max, cur_min = cur_min, cur_max
                
            cur_max = max(x, cur_max * x)
            cur_min = min(x, cur_min * x)
            
            global_max = max(global_max, cur_max)
            
        return global_max
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxProduct(const std::vector<int>& nums) {
        int cur_max = nums[0];
        int cur_min = nums[0];
        int global_max = nums[0];
        int n = static_cast<int>(nums.size());

        for (int i = 1; i < n; ++i) {
            int x = nums[i];
            if (x < 0) {
                std::swap(cur_max, cur_min);
            }

            cur_max = std::max(x, cur_max * x);
            cur_min = std::min(x, cur_min * x);

            global_max = std::max(global_max, cur_max);
        }

        return global_max;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxProduct(int[] nums) {
        int curMax = nums[0];
        int curMin = nums[0];
        int globalMax = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int x = nums[i];
            if (x < 0) {
                int temp = curMax;
                curMax = curMin;
                curMin = temp;
            }

            curMax = Math.max(x, curMax * x);
            curMin = Math.min(x, curMin * x);

            globalMax = Math.max(globalMax, curMax);
        }

        return globalMax;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |nums|$. We iterate through the array once, performing $\mathcal{O}(1)$ swaps and arithmetic comparisons per element.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only three scalar variables (`cur_max`, `cur_min`, `global_max`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Zero Handling:** When $x = 0$, both `cur_max` and `cur_min` become $\max(0, 0) = 0$ and $\min(0, 0) = 0$, effectively resetting the contiguous subarray window at the next non-zero number without needing explicit conditional branching.
2. **Alternative Two-Pass Prefix/Suffix Scan:** An elegant alternative observation is that the maximum product subarray is always either a prefix or a suffix bounded by zeros. Multiplying from left-to-right and right-to-left while resetting on zero also achieves the maximum in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space.
3. **Integer Overflow Caution:** In languages like C++, if the constraint were larger ($N \ge 10^5$), 32-bit products could overflow. Always check the constraint guarantee (`product fits in 32-bit signed int`).