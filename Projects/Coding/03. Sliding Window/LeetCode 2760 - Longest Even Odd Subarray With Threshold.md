---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2760: Longest Even Odd Subarray With Threshold"
tags:
  - leetcode
  - coding
  - sliding-window
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 2760: Longest Even Odd Subarray With Threshold

**Target Companies:** Amazon, Google, Microsoft, Adobe  
**Difficulty:** Easy  
**Topic:** Sliding Window / Two Pointers / Linear Scan  

---

### Problem Statement

You are given a 0-indexed integer array `nums` and an integer `threshold`.

Find the length of the longest subarray of `nums` starting at index $l$ and ending at index $r$ ($0 \le l \le r < \text{nums.length}$) that satisfies the following conditions:

1. `nums[l] % 2 == 0` (the first element must be even).
2. For all indices $i$ in the range $[l, r - 1]$, `nums[i] % 2 != nums[i + 1] % 2` (adjacent elements alternate parity).
3. For all indices $i$ in the range $[l, r]$, `nums[i] <= threshold` (all elements are $\le \text{threshold}$).

Return an integer denoting the length of the longest such subarray. If no such subarray exists, return $0$.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — array of integers ($1 \le \text{nums.length} \le 100$).
  - `threshold`: `int` ($1 \le \text{threshold} \le 100$).
- **Output:**
  - `int` — the maximum length of a valid alternating even-odd subarray $\le \text{threshold}$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 100$
  - $1 \le \text{nums}[i] \le 100$
  - $1 \le \text{threshold} \le 100$

---

### Key Idea & Intuition

The problem asks for the maximum length of a contiguous subarray that:
1. Starts with an even number $\le \text{threshold}$.
2. Continues as long as each subsequent element is $\le \text{threshold}$ AND alternates parity with the preceding element: `nums[r] % 2 != nums[r - 1] % 2`.

When an alternation or threshold condition is violated at index $r$, the subarray starting at $l$ cannot be extended further.
Crucially, where does the next potential subarray start?
- If `nums[r]` is odd or `nums[r] > threshold`, `nums[r]` itself cannot start a valid subarray, so we scan forward for the next even number $\le \text{threshold}$.
- If `nums[r]` is even and $\le \text{threshold}$, `nums[r]` could potentially start a new valid subarray right at $r$.

Using a sliding window / two-pointer approach:
- We iterate $l$ through `nums`.
- Whenever `nums[l] % 2 == 0` and `nums[l] <= threshold`, we expand $r$ from $l$:
  - While $r + 1 < n$, `nums[r + 1] <= threshold`, and `nums[r] % 2 != nums[r + 1] % 2`, advance $r$.
  - Record $\max(\text{ans}, r - l + 1)$.
  - We can then safely advance $l = r$ (or $l = r + 1$), running in strict $\mathcal{O}(n)$ time.

---

### Solution Approach (Step-by-Step)

1. Initialize `max_len = 0`, `n = len(nums)`, and pointer `l = 0`.
2. While `l < n`:
   - Check if `nums[l] % 2 == 0` and `nums[l] <= threshold`:
     - If yes, initialize `r = l`.
     - Expand `r` while $r + 1 < n$, `nums[r + 1] <= threshold`, and `(nums[r] % 2) != (nums[r + 1] % 2)`.
     - Update `max_len = max(max_len, r - l + 1)`.
     - Set `l = r + 1` (or if `nums[r + 1] <= threshold` and `nums[r + 1] % 2 == 0`, $r + 1$ will be checked on next iteration).
   - If no, simply increment `l += 1`.
3. Return `max_len`.

---

### Visual Algorithm Walkthrough

For `nums = [3, 2, 5, 4], threshold = 5`:

```
Array:      [ 3,    2,    5,    4 ]
Indices:      0     1     2     3

l = 0: nums[0] = 3 (Odd) -> Cannot start. l moves to 1.

l = 1: nums[1] = 2 (Even <= 5) -> Valid start!
  r = 1
  Check r+1 = 2 (nums[2] = 5):
    5 <= 5 (Threshold OK)
    2 % 2 (0) != 5 % 2 (1) (Parity alternates OK)
    -> r advances to 2 (Window: [2, 5], length 2)
  Check r+1 = 3 (nums[3] = 4):
    4 <= 5 (Threshold OK)
    5 % 2 (1) != 4 % 2 (0) (Parity alternates OK)
    -> r advances to 3 (Window: [2, 5, 4], length 3)
  Check r+1 = 4 (Out of bounds)
  -> Max len updated to 3.
  l advances to 4.

Loop terminates.
Result = 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [3, 2, 5, 4]`, `threshold = 5`
- **Tracing:** Valid window `[2, 5, 4]` of length 3.
- **Output:** `3`

#### Example 2:
- **Input:** `nums = [1, 2]`, `threshold = 2`
- **Tracing:** `nums[0] = 1` (odd). `nums[1] = 2` (even $\le 2$), length 1.
- **Output:** `1`

#### Example 3:
- **Input:** `nums = [2, 3, 4, 5]`, `threshold = 4`
- **Tracing:**
  - $l = 0$: `[2, 3, 4]` -> at index 3, `nums[3] = 5 > 4` (fails threshold). Window is `[2, 3, 4]`, length 3.
- **Output:** `3`

#### Example 4 (All Exceed Threshold or All Odd):
- **Input:** `nums = [7, 9, 11]`, `threshold = 5`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0
        l = 0
        
        while l < n:
            # Subarray must start with an even number <= threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                r = l
                # Greedily expand r while conditions are satisfied
                while (r + 1 < n and 
                       nums[r + 1] <= threshold and 
                       nums[r] % 2 != nums[r + 1] % 2):
                    r += 1
                
                max_len = max(max_len, r - l + 1)
                # Next potential start can begin at r (if even) or r + 1
                l = r
            l += 1
            
        return max_len
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestAlternatingSubarray(const std::vector<int>& nums, int threshold) {
        int n = static_cast<int>(nums.size());
        int max_len = 0;
        int l = 0;
        
        while (l < n) {
            // Must start with an even number <= threshold
            if (nums[l] % 2 == 0 && nums[l] <= threshold) {
                int r = l;
                while (r + 1 < n && 
                       nums[r + 1] <= threshold && 
                       (nums[r] % 2 != nums[r + 1] % 2)) {
                    ++r;
                }
                max_len = std::max(max_len, r - l + 1);
                l = r; // Skip ahead
            }
            ++l;
        }
        
        return max_len;
    }
};
```

#### Java 17
```java
class Solution {
    public int longestAlternatingSubarray(int[] nums, int threshold) {
        int n = nums.length;
        int maxLen = 0;
        int l = 0;
        
        while (l < n) {
            // Must begin with an even integer <= threshold
            if (nums[l] % 2 == 0 && nums[l] <= threshold) {
                int r = l;
                while (r + 1 < n && 
                       nums[r + 1] <= threshold && 
                       (nums[r] % 2 != nums[r + 1] % 2)) {
                    r++;
                }
                maxLen = Math.max(maxLen, r - l + 1);
                l = r; // Skip evaluated segment
            }
            l++;
        }
        
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Pointer `l` advances monotonically and pointer `r` traverses contiguous alternating segments. By setting `l = r`, each element is inspected at most twice throughout the entire execution.
- **Space Complexity:** $\mathcal{O}(1)$
  - Only a few scalar integer variables (`l`, `r`, `max_len`, `n`) are used.

---

### Takeaway Pattern & Interview Traps

- **Trap: Forgetting the Even Start Requirement:** It is easy to miss condition 1 (`nums[l] % 2 == 0`). Even if alternating parity holds, the segment cannot start on an odd number.
- **Trap: Forgetting Threshold on the First Element:** `nums[l] <= threshold` is required for the start; checking only alternating pairs will fail if the initial even number exceeds `threshold`.
- **Optimal Skipping:** After expanding to `r`, no index $k \in (l, r]$ can start a longer valid subarray than $r - l + 1$ unless we jump forward, because parity is rigidly fixed. Skipping $l = r$ gives clean $\mathcal{O}(n)$ without redundant inner checks.