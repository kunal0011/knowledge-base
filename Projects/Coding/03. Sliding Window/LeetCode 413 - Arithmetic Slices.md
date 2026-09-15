---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 413: Arithmetic Slices"
tags:
  - leetcode
  - coding
  - sliding-window
  - dynamic-programming
  - array
  - amazon
  - google
---

# LeetCode 413: Arithmetic Slices

**Target Companies:** Amazon, Google, Microsoft, Meta, Cisco  
**Difficulty:** Medium  
**Topic:** Sliding Window / Dynamic Programming / Two Pointers  

---

### Problem Statement

An integer array is called **arithmetic** if it consists of at least three elements and if the difference between any two consecutive elements is the same.

- For example, `[1,3,5,7,9]`, `[7,7,7,7]`, and `[3,-1,-5,-9]` are arithmetic sequences.

Given an integer array `nums`, return the number of arithmetic **subarrays** (contiguous slices) of `nums`.

A **subarray** is a contiguous subsequence of the array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 5000$).
- **Output:**
  - `int` — total count of contiguous arithmetic slices of length $\ge 3$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 5000$
  - $-1000 \le \text{nums}[i] \le 1000$

---

### Key Idea & Intuition

#### Perspective 1: Incremental Extension ($\mathcal{O}(n)$ time, $\mathcal{O}(1)$ space)
Let `current_slices` be the number of arithmetic slices ending at index $i - 1$.
When we examine index $i$:
If `nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]`:
- Every arithmetic slice that previously ended at index $i - 1$ can be extended by one element to end at index $i$.
- In addition, the 3-element slice `[nums[i-2], nums[i-1], nums[i]]` forms a brand new arithmetic slice.
- Thus, the number of arithmetic slices ending at index $i$ is:
  $$\text{slices}(i) = \text{slices}(i - 1) + 1$$
- We add $\text{slices}(i)$ to the total answer.
If the difference changes (`nums[i] - nums[i - 1] != nums[i - 1] - nums[i - 2]`):
- The arithmetic sequence is broken: `current_slices = 0`.

#### Perspective 2: Maximal Sliding Window Formula
Any maximal contiguous arithmetic subarray of length $L \ge 3$ produces:
$$\sum_{len=3}^{L} (L - len + 1) = \frac{(L - 1)(L - 2)}{2} \text{ arithmetic slices}$$
Both mathematical perspectives lead to the exact same closed-form result.

---

### Solution Approach (Step-by-Step)

1. If `len(nums) < 3`, return `0`.
2. Initialize `total_slices = 0` and `current_slices = 0`.
3. Loop `i` from $2$ to $n - 1$:
   - Check if `nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]`:
     - If yes, increment `current_slices += 1`, and add `total_slices += current_slices`.
     - If no, reset `current_slices = 0`.
4. Return `total_slices`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 2, 3, 4, 5]`:

```
Indices:    0    1    2    3    4
Values:    [1,   2,   3,   4,   5]

i = 2:
  nums[2]-nums[1] = 3 - 2 = 1
  nums[1]-nums[0] = 2 - 1 = 1
  Equal! -> current_slices = 0 + 1 = 1 ([1, 2, 3])
  total = 1

i = 3:
  nums[3]-nums[2] = 4 - 3 = 1
  nums[2]-nums[1] = 3 - 2 = 1
  Equal! -> current_slices = 1 + 1 = 2 ([1, 2, 3, 4] and [2, 3, 4])
  total = 1 + 2 = 3

i = 4:
  nums[4]-nums[3] = 5 - 4 = 1
  nums[3]-nums[2] = 4 - 3 = 1
  Equal! -> current_slices = 2 + 1 = 3 ([1..5], [2..5], [3..5])
  total = 3 + 3 = 6

Final Answer = 6 slices.
Check formula: Length L = 5 -> (5 - 1)*(5 - 2) / 2 = 4 * 3 / 2 = 6. (Matches!)
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 2, 3, 4]`
- **Output:** `3` (Slices: `[1, 2, 3]`, `[2, 3, 4]`, `[1, 2, 3, 4]`)

#### Example 2 (Short Array):
- **Input:** `nums = [1]`
- **Output:** `0` (Length $< 3$)

#### Example 3 (Broken Sequence):
- **Input:** `nums = [1, 3, 5, 8, 11, 14]`
- **Tracing:**
  - `[1, 3, 5]`: diff = 2, length = 3 -> 1 slice.
  - At 8: diff becomes 3 ($8 - 5 = 3 \ne 2$) -> reset.
  - `[5, 8, 11, 14]`: diff = 3, length = 4 -> 3 slices.
  - Total = $1 + 3 = 4$.
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
            
        total_slices = 0
        current_slices = 0
        
        for i in range(2, n):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                current_slices += 1
                total_slices += current_slices
            else:
                current_slices = 0
                
        return total_slices
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int numberOfArithmeticSlices(const std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        if (n < 3) return 0;
        
        int total_slices = 0;
        int current_slices = 0;
        
        for (int i = 2; i < n; ++i) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                current_slices += 1;
                total_slices += current_slices;
            } else {
                current_slices = 0;
            }
        }
        
        return total_slices;
    }
};
```

#### Java 17
```java
class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n < 3) return 0;
        
        int totalSlices = 0;
        int currentSlices = 0;
        
        for (int i = 2; i < n; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                currentSlices++;
                totalSlices += currentSlices;
            } else {
                currentSlices = 0;
            }
        }
        
        return totalSlices;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear scan from index $2$ to $n - 1$ performing constant-time arithmetic checks at each index.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only two integer counters `total_slices` and `current_slices`.

---

### Takeaway Pattern & Interview Traps

- **Consecutive Difference Matching:** Whenever a problem asks for contiguous segments satisfying a local binary relation (like identical differences), checking whether condition holds between $i$ and $i-1$ vs $i-1$ and $i-2$ reduces the problem to counting runs of consecutive truth values.
- **Difference Between Subarray (contiguous) and Subsequence (non-contiguous):**
  - This problem is **Arithmetic Slices** (LeetCode 413, contiguous subarray) which is $\mathcal{O}(n)$.
  - Do not confuse with **Arithmetic Slices II - Subsequence** (LeetCode 446, non-contiguous subsequence) which requires DP with HashMaps in $\mathcal{O}(n^2)$ time.