---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1004: Max Consecutive Ones III"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - two-pointers
  - amazon
  - google
---

# LeetCode 1004: Max Consecutive Ones III

**Target Companies:** Meta (Top Favorite), Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Sliding Window / Subarray Zero-Count Invariant  

---

### Problem Statement

Given a binary array `nums` and an integer `k`, return *the maximum number of consecutive `1`'s in the array if you can flip at most `k` `0`'s*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (maximum length of a contiguous subarray)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - `nums[i]` is either `0` or `1`.
  - $0 \le k \le \text{nums.length}$

---

### Key Idea & Intuition

- **Problem Reinterpretation:**
  - "Flipping at most $k$ zeros to ones" is completely equivalent to:
    $$\text{Find the longest contiguous subarray containing at most } k \text{ zeros.}$$
- **Dynamic Sliding Window Invariant:**
  - Maintain a window $[L, R]$ with a counter `zero_count` representing the number of zeros in the window:
    1. **Expand Right ($R$):** Include `nums[R]`. If `nums[R] == 0`, increment `zero_count`.
    2. **Contract Left ($L$):** While `zero_count > k`, the current window is invalid. Increment $L$; if `nums[L] == 0`, decrement `zero_count`.
    3. **Record Maximum:** Once valid (`zero_count <= k`), the length of the window is $R - L + 1$. Update $\text{max\_len} = \max(\text{max\_len}, R - L + 1)$.
- **Non-Shrinking Window Variant ($\mathcal{O}(N)$ strictly single-pass):**
  - Instead of contracting the window with a `while` loop, we can use an `if zero_count > k`: increment $L$ by $1$ and decrement `zero_count` if `nums[L] == 0`.
  - Under this approach, the window size never shrinks; it only expands or shifts right, and the final answer is simply `R - L + 1`!

---

### Solution Approach (Step-by-Step)

1. Initialize `left = 0`, `zero_count = 0`, and `max_len = 0`.
2. Iterate `right` from $0$ to $\text{len}(nums) - 1$:
   - If `nums[right] == 0`:
     - `zero_count += 1`
   - While `zero_count > k`:
     - If `nums[left] == 0`:
       - `zero_count -= 1`
     - `left += 1`
   - `max_len = max(max_len, right - left + 1)`
3. Return `max_len`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]`, `k = 2`.

```
Index:    0  1  2  3  4  5  6  7  8  9  10
Nums:    [1, 1, 1, 0, 0, 0, 1, 1, 1, 1,  0]

R=0..2:  [1, 1, 1], zeros=0 <= 2 -> max_len = 3
R=3:     [1, 1, 1, 0], zeros=1 <= 2 -> max_len = 4
R=4:     [1, 1, 1, 0, 0], zeros=2 <= 2 -> max_len = 5
R=5:     [1, 1, 1, 0, 0, 0], zeros=3 > 2! (INVALID)
         Shrink L until zeros <= 2:
         - L moves past 0, 1, 2, 3 (pops a 0): L = 4.
         Window: [0, 0] (indices 4..5), zeros=2.
R=6..9:  Expand to R=9:
         Window: [0, 0, 1, 1, 1, 1] (indices 4..9), zeros=2 <= 2.
         Length = 9 - 4 + 1 = 6 -> max_len = 6!
R=10:    Nums[10]=0, zeros=3 > 2!
         Shrink L past index 4 (pops 0): L = 5.
         Window indices 5..10: [0, 1, 1, 1, 1, 0], length = 6.

Final Maximum Length: 6 (Subarray [0, 0, 1, 1, 1, 1] with 2 zeros flipped).
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `k` | Zero Count in Best Subarray | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[1,1,1,0,0,0,1,1,1,1,0]` | `2` | 2 zeros flipped | `6` |
| **Example 2** | `[0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]` | `3` | 3 zeros flipped | `10` |
| **k == 0** | `[1,1,0,1,1,1]` | `0` | Longest run of pure 1s | `3` |
| **All Zeros** | `[0,0,0,0]` | `2` | Can flip 2 zeros | `2` |
| **k >= len(nums)** | `[0,1,0]` | `5` | Flip all zeros | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        """
        Finds the maximum consecutive 1s after flipping at most k zeros.
        Uses standard contracting sliding window.
        """
        left = 0
        zero_count = 0
        max_len = 0

        for right in range(len(nums)):
            if nums[right] == 0:
                zero_count += 1

            # Contract window from left if zeros exceed allowed threshold k
            while zero_count > k:
                if nums[left] == 0:
                    zero_count -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len

    def longestOnesNonShrinking(self, nums: List[int], k: int) -> int:
        """
        Alternative: Non-shrinking sliding window.
        Window size never shrinks, maintaining maximum length achieved.
        """
        left = 0
        for right in range(len(nums)):
            if nums[right] == 0:
                k -= 1
            if k < 0:
                if nums[left] == 0:
                    k += 1
                left += 1
        return len(nums) - left
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestOnes(const std::vector<int>& nums, int k) {
        int left = 0;
        int zero_count = 0;
        int max_len = 0;

        for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
            if (nums[right] == 0) {
                zero_count++;
            }

            while (zero_count > k) {
                if (nums[left] == 0) {
                    zero_count--;
                }
                left++;
            }

            max_len = std::max(max_len, right - left + 1);
        }

        return max_len;
    }
};
```

#### Java
```java
class Solution {
    public int longestOnes(int[] nums, int k) {
        int left = 0;
        int zeroCount = 0;
        int maxLen = 0;

        for (int right = 0; right < nums.length; right++) {
            if (nums[right] == 0) {
                zeroCount++;
            }

            while (zeroCount > k) {
                if (nums[left] == 0) {
                    zeroCount--;
                }
                left++;
            }

            maxLen = Math.max(maxLen, right - left + 1);
        }

        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - The `right` pointer increments $N$ times.
  - The `left` pointer increments at most $N$ times across the entire loop.
  - Each element is processed at most twice $\implies 2N$ operations, running in $< 15 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space as only two pointers and a counter are maintained.

---

### Takeaway Pattern & Interview Traps

- **Problem Reframing:** Whenever a problem asks to "flip at most $K$ elements", reframe it into finding the "longest contiguous subarray containing at most $K$ target elements".
- **Two Pointers Invariant:** The `left` pointer only moves forward; it never resets backwards, guaranteeing linear amortized time complexity.