---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Two Pointers / Array  

---

### Problem Statement

Given an integer array `nums`, you need to find one **continuous subarray** such that if you only sort this subarray in non-decreasing order, then the whole array will be sorted in non-decreasing order.

Return the shortest such subarray and output its **length**.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 10^4$).
- **Output:**
  - `int` — the minimum length of a contiguous subarray that needs sorting, or `0` if already sorted.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-10^5 \le \text{nums}[i] \le 10^5$
  - Follow up: Can you solve it in $\mathcal{O}(n)$ time complexity?

---

### Key Idea & Intuition

If the array is split into three parts:
$$\text{nums} = [ \text{Prefix}, \text{Subarray}, \text{Suffix} ]$$
where only $\text{Subarray}$ is rearranged to make the whole array sorted:
1. Every element in $\text{Prefix}$ must be $\le \min(\text{Subarray} \cup \text{Suffix})$.
2. Every element in $\text{Suffix}$ must be $\ge \max(\text{Prefix} \cup \text{Subarray})$.

#### The Boundary Invariant:
- **Right boundary ($end$):**
  - Moving left to right, we maintain `max_seen`.
  - In a properly sorted array, each element is $\ge$ all elements before it (`nums[i] >= max_seen`).
  - If we encounter `nums[i] < max_seen`, then `nums[i]` is in an illegal position and must be included in the unsorted subarray!
  - The **furthest right** index $i$ where `nums[i] < max_seen` is the right boundary $end$.
- **Left boundary ($start$):**
  - Moving right to left, we maintain `min_seen`.
  - In a properly sorted array, each element is $\le$ all elements after it (`nums[i] <= min_seen`).
  - If we encounter `nums[i] > min_seen`, then `nums[i]` is in an illegal position and must be included in the unsorted subarray!
  - The **furthest left** index $i$ where `nums[i] > min_seen` is the left boundary $start$.

If no violations occur ($end == -1$), the array is already sorted $\rightarrow$ return $0$.
Otherwise, the shortest unsorted continuous subarray is $[start, end]$, with length $end - start + 1$.

---

### Solution Approach (Step-by-Step)

1. Initialize `n = len(nums)`, `start = -1`, `end = -2`.
2. Initialize `max_val = -infinity` and `min_val = infinity`.
3. Loop $i$ from $0$ to $n - 1$:
   - Let $j = n - 1 - i$.
   - **Forward check (for right boundary):**
     - `max_val = max(max_val, nums[i])`
     - If `nums[i] < max_val`:
       - `end = i`
   - **Backward check (for left boundary):**
     - `min_val = min(min_val, nums[j])`
     - If `nums[j] > min_val`:
       - `start = j`
4. Return `end - start + 1` if `end >= start` else `0`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 6, 4, 8, 10, 9, 15]`:

```
Indices:    0   1   2   3   4   5   6
Values:    [2,  6,  4,  8, 10,  9, 15]

Forward Pass (detecting end):
  i=0 (2):  max=2.  nums[0] == max.
  i=1 (6):  max=6.  nums[1] == max.
  i=2 (4):  max=6.  nums[2]=4 < max(6)! -> Violation at index 2. end = 2.
  i=3 (8):  max=8.  nums[3] == max.
  i=4 (10): max=10. nums[4] == max.
  i=5 (9):  max=10. nums[5]=9 < max(10)! -> Violation at index 5. end = 5.
  i=6 (15): max=15. nums[6] == max.
Right boundary end = 5.

Backward Pass (detecting start):
  j=6 (15): min=15. nums[6] == min.
  j=5 (9):  min=9.  nums[5] == min.
  j=4 (10): min=9.  nums[4]=10 > min(9)! -> Violation at index 4. start = 4.
  j=3 (8):  min=8.  nums[3] == min.
  j=2 (4):  min=4.  nums[2] == min.
  j=1 (6):  min=4.  nums[1]=6 > min(4)! -> Violation at index 1. start = 1.
  j=0 (2):  min=2.  nums[0] == min.
Left boundary start = 1.

Unsorted Subarray: nums[1..5] = [6, 4, 8, 10, 9]
Length = 5 - 1 + 1 = 5.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [2, 6, 4, 8, 10, 9, 15]`
- **Output:** `5` (Subarray `[6, 4, 8, 10, 9]` must be sorted)

#### Example 2 (Already Sorted):
- **Input:** `nums = [1, 2, 3, 4]`
- **Tracing:** No violations in either direction.
- **Output:** `0`

#### Example 3 (Entire Array Unsorted):
- **Input:** `nums = [1, 3, 2, 2, 2]`
- **Tracing:**
  - $end = 4$ ($2 < 3$)
  - $start = 1$ ($3 > 2$)
  - Length = $4 - 1 + 1 = 4$.
- **Output:** `4`

#### Example 4 (Single Element):
- **Input:** `nums = [1]`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        start, end = -1, -2
        max_val = float('-inf')
        min_val = float('inf')
        
        for i in range(n):
            # Left-to-right pass for right boundary
            max_val = max(max_val, nums[i])
            if nums[i] < max_val:
                end = i
                
            # Right-to-left pass for left boundary
            j = n - 1 - i
            min_val = min(min_val, nums[j])
            if nums[j] > min_val:
                start = j
                
        return end - start + 1 if end >= start else 0
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int findUnsortedSubarray(const std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        int start = -1, end = -2;
        int max_val = INT_MIN;
        int min_val = INT_MAX;
        
        for (int i = 0; i < n; ++i) {
            max_val = std::max(max_val, nums[i]);
            if (nums[i] < max_val) {
                end = i;
            }
            
            int j = n - 1 - i;
            min_val = std::min(min_val, nums[j]);
            if (nums[j] > min_val) {
                start = j;
            }
        }
        
        return (end >= start) ? (end - start + 1) : 0;
    }
};
```

#### Java 17
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int start = -1, end = -2;
        int maxVal = Integer.MIN_VALUE;
        int minVal = Integer.MAX_VALUE;
        
        for (int i = 0; i < n; i++) {
            maxVal = Math.max(maxVal, nums[i]);
            if (nums[i] < maxVal) {
                end = i;
            }
            
            int j = n - 1 - i;
            minVal = Math.min(minVal, nums[j]);
            if (nums[j] > minVal) {
                start = j;
            }
        }
        
        return (end >= start) ? (end - start + 1) : 0;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Both boundaries $start$ and $end$ are resolved in a single concurrent linear scan of length $n$.
  - Avoids the $\mathcal{O}(n \log n)$ time required by sorting-comparison approaches.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only scalar registers (`start`, `end`, `max_val`, `min_val`).

---

### Takeaway Pattern & Interview Traps

- **Concurrent Dual-Sweep Trick:** Running $i$ forward and $j = n - 1 - i$ backward in the same loop allows identifying both the earliest inversion and latest inversion in a single pass of $n$ cycles.
- **Initializing `start = -1, end = -2`:** Setting `end = -2` and `start = -1` makes the formula `end - start + 1 = -2 - (-1) + 1 = 0` automatically return $0$ when the array is already completely sorted without requiring extra branches!