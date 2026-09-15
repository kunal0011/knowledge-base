---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3095: Shortest Subarray With OR at Least K I"
tags:
  - leetcode
  - coding
  - sliding-window
  - bit-manipulation
  - two-pointers
  - amazon
  - google
---

# LeetCode 3095: Shortest Subarray With OR at Least K I

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Easy  
**Topic:** Sliding Window / Bit Manipulation / Two Pointers  

---

### Problem Statement

You are given an array `nums` of non-negative integers and an integer `k`.

An array is called **special** if the bitwise `OR` of all of its elements is at least `k`.

Return the length of the **shortest special non-empty subarray** of `nums`, or return `-1` if no special subarray exists.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — array of non-negative integers ($1 \le \text{nums.length} \le 50$, $0 \le \text{nums}[i] \le 50$).
  - `k`: `int` — target bitwise OR threshold ($0 \le k \le 64$).
- **Output:**
  - `int` — minimum length of a contiguous subarray whose bitwise OR is $\ge k$, or `-1` if impossible.
- **Constraints:**
  - $1 \le \text{nums.length} \le 50$
  - $0 \le \text{nums}[i] \le 50$
  - $0 \le k \le 64$
  *(Note: The optimal Sliding Window with Bit Counter technique presented below scales directly to $N = 2 \times 10^5$ and $k \le 10^9$ as required in LeetCode 3097).*

---

### Key Idea & Intuition

Bitwise `OR` is **monotonically non-decreasing** under set inclusion: for any non-negative integers, $A \mid B \ge A$.
This monotonic property suggests a sliding window:
- Expanding the right boundary $r$ can only add set bits, increasing or preserving the window's total OR value.
- Shrinking the left boundary $l$ can only remove bits, decreasing or preserving the OR value.

However, unlike addition (where subtraction easily reverses an addition), **bitwise OR has no direct algebraic inverse**: if $A \mid B = C$, knowing $C$ and $B$ does not allow you to recover $A$.

#### The Invertible State Technique (Bit Count Frequency Array):
To make sliding window eviction possible in $\mathcal{O}(1)$ time:
- Maintain an array `bit_counts` of size 32, where `bit_counts[b]` records the number of elements currently in the window that have their $b$-th bit set.
- **Add an element `val`:** For every bit position $b \in [0, 31]$, if `(val >> b) & 1`, increment `bit_counts[b] += 1`.
- **Remove an element `val`:** For every bit position $b \in [0, 31]$, if `(val >> b) & 1`, decrement `bit_counts[b] -= 1`.
- **Compute current window OR:**
  $$\text{window\_or} = \sum_{b=0}^{31} (1 \ll b) \quad \text{for all } b \text{ where } \text{bit\_counts}[b] > 0$$
  This calculation takes $\mathcal{O}(32) = \mathcal{O}(1)$ time!

With this, when `window_or >= k`, we update `min_len = min(min_len, r - l + 1)` and shrink from `l` until `window_or < k`.

---

### Solution Approach (Step-by-Step)

1. If $k = 0$, any single element subarray has bitwise OR $\ge 0$, return $1$.
2. Initialize `min_len = infinity`, `left = 0`, and `bit_counts = [0] * 32`.
3. Iterate `right` from $0$ to $n - 1$:
   - Add `nums[right]` to `bit_counts`.
   - While `left <= right` and `get_or(bit_counts) >= k`:
     - Update `min_len = min(min_len, right - left + 1)`.
     - Remove `nums[left]` from `bit_counts`.
     - Increment `left += 1`.
4. Return `min_len` if `min_len != infinity` else `-1`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 2, 32, 21]`, `k = 55`:
Binary representations:
- $1 = (000001)_2$
- $2 = (000010)_2$
- $32 = (100000)_2$
- $21 = (010101)_2$
Target $k = 55 = (110111)_2$

```
Window [0..r]:
r = 0 (val = 1):
  bit_counts: bit 0: 1
  OR = 1 < 55

r = 1 (val = 2):
  bit_counts: bit 0: 1, bit 1: 1
  OR = 3 < 55

r = 2 (val = 32):
  bit_counts: bit 0: 1, bit 1: 1, bit 5: 1
  OR = 35 < 55

r = 3 (val = 21):
  bit_counts: bit 0: 2, bit 1: 1, bit 2: 1, bit 4: 1, bit 5: 1
  Active bits: {0, 1, 2, 4, 5}
  OR = 1 | 2 | 4 | 16 | 32 = 55 >= 55! -> VALID!
  Current window [0..3], len = 4. min_len = 4.

  Shrink from left:
  Evict nums[0] = 1 (bit 0 count becomes 1):
    Active bits still {0, 1, 2, 4, 5} -> OR = 55 >= 55!
    Window [1..3], len = 3. min_len = 3.
    left moves to 2.

  Evict nums[1] = 2 (bit 1 count becomes 0):
    Active bits now {0, 2, 4, 5} -> OR = 1 | 4 | 16 | 32 = 53 < 55!
    Shrink stops. left stays at 2.

Result: min_len = 3 (subarray [2, 32, 21]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 2, 3]`, `k = 2`
- **Tracing:** `nums[1] = 2 >= 2` -> length 1.
- **Output:** `1`

#### Example 2:
- **Input:** `nums = [2, 1, 8]`, `k = 10`
- **Tracing:**
  - $2 \mid 1 = 3 < 10$
  - $2 \mid 1 \mid 8 = 11 \ge 10$ (length 3)
  - Shrink left: $1 \mid 8 = 9 < 10$. Stop.
- **Output:** `3`

#### Example 3 (Impossible):
- **Input:** `nums = [1, 2]`, `k = 0`
- **Output:** `1`

#### Example 4 (No valid subarray):
- **Input:** `nums = [1, 2]`, `k = 5`
- **Output:** `-1` (maximum possible OR is $1 \mid 2 = 3 < 5$)

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        if k == 0:
            return 1
            
        n = len(nums)
        bit_counts = [0] * 32
        min_len = float('inf')
        left = 0
        
        def update_bits(val: int, delta: int) -> None:
            for b in range(32):
                if (val >> b) & 1:
                    bit_counts[b] += delta
                    
        def current_or() -> int:
            ans = 0
            for b in range(32):
                if bit_counts[b] > 0:
                    ans |= (1 << b)
            return ans
            
        for right in range(n):
            update_bits(nums[right], 1)
            
            while left <= right and current_or() >= k:
                min_len = min(min_len, right - left + 1)
                update_bits(nums[left], -1)
                left += 1
                
        return min_len if min_len != float('inf') else -1
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int minimumSubarrayLength(const std::vector<int>& nums, int k) {
        if (k == 0) return 1;
        
        int n = static_cast<int>(nums.size());
        std::vector<int> bit_counts(32, 0);
        int min_len = INT_MAX;
        int left = 0;
        
        auto update_bits = [&](int val, int delta) {
            for (int b = 0; b < 32; ++b) {
                if ((val >> b) & 1) {
                    bit_counts[b] += delta;
                }
            }
        };
        
        auto get_current_or = [&]() -> int {
            int res = 0;
            for (int b = 0; b < 32; ++b) {
                if (bit_counts[b] > 0) {
                    res |= (1 << b);
                }
            }
            return res;
        };
        
        for (int right = 0; right < n; ++right) {
            update_bits(nums[right], 1);
            
            while (left <= right && get_current_or() >= k) {
                min_len = std::min(min_len, right - left + 1);
                update_bits(nums[left], -1);
                ++left;
            }
        }
        
        return min_len == INT_MAX ? -1 : min_len;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int minimumSubarrayLength(int[] nums, int k) {
        if (k == 0) return 1;
        
        int n = nums.length;
        int[] bitCounts = new int[32];
        int minLen = Integer.MAX_VALUE;
        int left = 0;
        
        for (int right = 0; right < n; right++) {
            updateBits(bitCounts, nums[right], 1);
            
            while (left <= right && getOr(bitCounts) >= k) {
                minLen = Math.min(minLen, right - left + 1);
                updateBits(bitCounts, nums[left], -1);
                left++;
            }
        }
        
        return minLen == Integer.MAX_VALUE ? -1 : minLen;
    }
    
    private void updateBits(int[] bitCounts, int val, int delta) {
        for (int b = 0; b < 32; b++) {
            if (((val >> b) & 1) == 1) {
                bitCounts[b] += delta;
            }
        }
    }
    
    private int getOr(int[] bitCounts) {
        int res = 0;
        for (int b = 0; b < 32; b++) {
            if (bitCounts[b] > 0) {
                res |= (1 << b);
            }
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(32 \cdot n) = \mathcal{O}(n)$
  - Pointer `right` visits each element once; pointer `left` visits each element at most once.
  - Updating bit counts and recomputing the OR takes 32 bit operations ($\mathcal{O}(1)$ time).
  - Total time is $\mathcal{O}(n)$, which runs in $< 2\text{ ms}$ for $n = 50$ and scales smoothly to $n = 2 \times 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The bit counts array uses exactly 32 integers.

---

### Takeaway Pattern & Interview Traps

- **Non-Invertible Aggregations in Sliding Window:** Bitwise OR, AND, and GCD do not support simple subtraction. Whenever sliding window is required over non-invertible operations, maintain per-bit frequency counters (for bitwise OR/AND) or two stacks / queue using amortized rollbacks.
- **Corner Case $k = 0$:** If $k = 0$, any non-empty subarray has an OR of at least $0$. Handling $k = 0$ immediately returns 1.
- **Edge Case: No Subarray Achieves $k$:** If the total OR of the whole array is $< k$, no subarray can ever reach $k$, returning `-1`.