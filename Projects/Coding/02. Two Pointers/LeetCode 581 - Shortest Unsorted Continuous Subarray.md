---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - two-pointers
  - stack
  - greedy
  - amazon
  - google
  - meta
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Monotonic Extremes Sweep in $\mathcal{O}(n)$ Time and $\mathcal{O}(1)$ Space

---

### Problem Statement

Given an integer array `nums`, you need to find one **continuous subarray** that if you only sort this subarray in ascending order, then the whole array will be sorted in ascending order.

Return *the shortest length of such a subarray*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $-10^5 \le \text{nums}[i] \le 10^5$
- **Follow up:** Can you solve it in $\mathcal{O}(n)$ time complexity?

---

### Key Idea & Intuition

#### 1. Sorted vs Unsorted Subarray Boundaries:
A fully sorted array satisfies:
$$\text{nums}[0] \le \text{nums}[1] \le \dots \le \text{nums}[n-1]$$
If we partition the array into three sections:
$$\text{Prefix (Sorted)} \quad | \quad \text{Middle Unsorted Subarray } [start, end] \quad | \quad \text{Suffix (Sorted)}$$
The boundaries $start$ and $end$ must satisfy:
1. Every element in the prefix must be $\le$ the **minimum** element in the entire remainder of the array to its right.
2. Every element in the suffix must be $\ge$ the **maximum** element in the entire remainder of the array to its left.

#### 2. Determining `end` (Left-to-Right Pass):
- Maintain `max_so_far` as we scan from left to right.
- In a sorted array, each element should be $\ge \text{max\_so\_far}$.
- If we encounter any element $\text{nums}[i] < \text{max\_so\_far}$, it is out of order! Thus, the right boundary of the unsorted subarray must extend at least to index $i$:
  $$\text{end} = i$$

#### 3. Determining `start` (Right-to-Left Pass):
- Maintain `min_so_far` as we scan from right to left.
- In a sorted array, each element should be $\le \text{min\_so\_far}$.
- If we encounter any element $\text{nums}[i] > \text{min\_so\_far}$, it is out of order! Thus, the left boundary of the unsorted subarray must extend at least down to index $i$:
  $$\text{start} = i$$

If no violations occur (`end == -1`), the array is already completely sorted $\implies$ return 0. Otherwise, the length is $end - start + 1$.

---

### Solution Approach (Step-by-Step)

1. **Initialize State:**
   - `n = len(nums)`, `end = -1`, `start = 0`.
   - `max_so_far = nums[0]`, `min_so_far = nums[-1]`.
2. **Left-to-Right Scan (Find `end`):**
   - For $i$ from 1 to $n - 1$:
     - If `nums[i] < max_so_far`: `end = i`.
     - Else: `max_so_far = nums[i]`.
3. **Right-to-Left Scan (Find `start`):**
   - For $i$ from $n - 2$ down to 0:
     - If `nums[i] > min_so_far`: `start = i`.
     - Else: `min_so_far = nums[i]`.
4. **Compute Length:**
   - If `end == -1`, return 0.
   - Else, return `end - start + 1`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [2, 6, 4, 8, 10, 9, 15]` ($N = 7$)

```
Pass 1 (Left to Right):
i=0: max_so_far = 2
i=1: val=6 >= 2 -> max_so_far = 6
i=2: val=4 < 6  -> VIOLATION! end = 2
i=3: val=8 >= 6 -> max_so_far = 8
i=4: val=10 >= 8 -> max_so_far = 10
i=5: val=9 < 10 -> VIOLATION! end = 5
i=6: val=15 >= 10 -> max_so_far = 15
Final end = 5 (element 9)

Pass 2 (Right to Left):
i=6: min_so_far = 15
i=5: val=9 <= 15 -> min_so_far = 9
i=4: val=10 > 9 -> VIOLATION! start = 4
i=3: val=8 <= 9 -> min_so_far = 8
i=2: val=4 <= 8 -> min_so_far = 4
i=1: val=6 > 4  -> VIOLATION! start = 1
i=0: val=2 <= 4 -> min_so_far = 2
Final start = 1 (element 6)

Unsorted window: [start=1, end=5] -> elements [6, 4, 8, 10, 9]
Length = end - start + 1 = 5 - 1 + 1 = 5.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | `start` | `end` | Subarray to Sort | Output Length |
| :--- | :--- | :--- | :--- | :--- |
| `[2, 6, 4, 8, 10, 9, 15]` | 1 | 5 | `[6, 4, 8, 10, 9]` | `5` |
| `[1, 2, 3, 4]` | 0 | -1 | Already sorted | `0` |
| `[1]` | 0 | -1 | Single element | `0` |
| `[1, 3, 2, 2, 2]` | 1 | 4 | `[3, 2, 2, 2]` | `4` |
| `[5, 4, 3, 2, 1]` | 0 | 4 | Entire array | `5` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        end = -1
        max_so_far = nums[0]
        
        # Find rightmost boundary out of order
        for i in range(1, n):
            if nums[i] < max_so_far:
                end = i
            else:
                max_so_far = nums[i]
                
        start = 0
        min_so_far = nums[-1]
        
        # Find leftmost boundary out of order
        for i in range(n - 2, -1, -1):
            if nums[i] > min_so_far:
                start = i
            else:
                min_so_far = nums[i]
                
        return end - start + 1 if end != -1 else 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int findUnsortedSubarray(std::vector<int>& nums) {
        int n = nums.size();
        int end = -1, start = 0;
        int maxSoFar = nums[0];
        int minSoFar = nums[n - 1];

        for (int i = 1; i < n; ++i) {
            if (nums[i] < maxSoFar) {
                end = i;
            } else {
                maxSoFar = nums[i];
            }
        }

        for (int i = n - 2; i >= 0; --i) {
            if (nums[i] > minSoFar) {
                start = i;
            } else {
                minSoFar = nums[i];
            }
        }

        return end == -1 ? 0 : end - start + 1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int end = -1, start = 0;
        int maxSoFar = nums[0];
        int minSoFar = nums[n - 1];

        for (int i = 1; i < n; i++) {
            if (nums[i] < maxSoFar) {
                end = i;
            } else {
                maxSoFar = nums[i];
            }
        }

        for (int i = n - 2; i >= 0; i--) {
            if (nums[i] > minSoFar) {
                start = i;
            } else {
                minSoFar = nums[i];
            }
        }

        return end == -1 ? 0 : end - start + 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$. Two sequential linear passes through the array of length $n$, executing constant time $\mathcal{O}(1)$ comparisons per element.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only a few integer variables (`start`, `end`, `max_so_far`, `min_so_far`) are used.

---

### Takeaway Pattern & Interview Traps

1. **Sorting Baseline vs Linear Extremes:**
   - Cloning and sorting the array takes $\mathcal{O}(n \log n)$ time and $\mathcal{O}(n)$ space to find the first and last mismatches. The two-pointer running max/min sweep avoids sorting completely, achieving the optimal $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space.
2. **Already Sorted Condition:**
   - If the input is already sorted, `end` remains `-1`. Returning `end - start + 1` without checking `end != -1` would return negative or invalid values.
