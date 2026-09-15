---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 581: Shortest Unsorted Continuous Subarray"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - two-pointers
  - array
  - amazon
  - google
---

# LeetCode 581: Shortest Unsorted Continuous Subarray

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Monotonic Stack / Two Pointers / Array Scan

---

### Problem Statement

Given an integer array `nums`, you need to find one **continuous subarray** such that if you only sort this subarray in non-decreasing order, then the whole array will be sorted in non-decreasing order.

Return *the shortest such subarray and output its length*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{len}(nums) \le 10^4$.
  - $-10^5 \le nums[i] \le 10^5$.
- **Output:**
  - `int`: The length of the shortest unsorted continuous subarray. Returns `0` if the array is already sorted.
- **Constraints:**
  - Subarray must be contiguous.
  - Sorting this subarray must render the entire array globally sorted.

---

### Key Idea & Intuition

An array is globally sorted in non-decreasing order if and only if for every index $i$:
$$nums[i] \ge \max(nums[0 \dots i-1]) \quad \text{and} \quad nums[i] \le \min(nums[i+1 \dots n-1])$$

The unsorted window $[left, right]$ corresponds to all indices that violate these conditions:
- **Right Boundary ($right$):** The rightmost index $i$ such that $nums[i]$ is smaller than some element to its left. That is, $nums[i] < \max_{0 \le j < i}(nums[j])$.
- **Left Boundary ($left$):** The leftmost index $i$ such that $nums[i]$ is larger than some element to its right. That is, $nums[i] > \min_{i < j < n}(nums[j])$.

#### Approach 1: Monotonic Stack ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space)
1. **Finding `left`:** Scan from left to right using a monotonic increasing stack. If $nums[i] < nums[\text{stack.top()}]$, elements in the stack are out of order relative to $nums[i]$. We pop them and update $left = \min(left, \text{popped})$.
2. **Finding `right`:** Scan from right to left using a monotonic decreasing stack. If $nums[i] > nums[\text{stack.top()}]$, we pop and update $right = \max(right, \text{popped})$.
3. Length is $\max(0, right - left + 1)$.

#### Approach 2: Running Min / Max Linear Scan ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space - Optimal)
We can track boundaries without a stack by running two simultaneous scans:
- Traverse left-to-right maintaining `max_seen`. If $nums[i] < max\_seen$, $nums[i]$ violates sorted order, so update `right = i`.
- Traverse right-to-left maintaining `min_seen`. If $nums[i] > min\_seen$, $nums[i]$ violates sorted order, so update `left = i`.
- If `right <= left`, the array was already sorted $\implies$ return $0$. Otherwise, return $right - left + 1$.

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: $\mathcal{O}(1)$ Space Running Extrema (Optimal Standard)
1. Initialize `n = len(nums)`, `left = -1`, `right = -2`.
2. Initialize `max_seen = -inf`, `min_seen = inf`.
3. Loop $i$ from $0$ to $n - 1$:
   - `max_seen = max(max_seen, nums[i])`
   - If $nums[i] < max\_seen$:
     - `right = i`
   - $j = n - 1 - i$
   - `min_seen = min(min_seen, nums[j])`
   - If $nums[j] > min\_seen$:
     - `left = j`
4. Return `max(0, right - left + 1)`.

#### Algorithm 2: Monotonic Stack
1. `stack = []`, `left = n`, `right = 0`.
2. Left-to-right pass:
   - For $i \in [0, n-1]$:
     - While `stack` and $nums[i] < nums[stack[-1]]$:
       - `left = min(left, stack.pop())`
     - `stack.append(i)`
3. Right-to-left pass:
   - Clear `stack`.
   - For $i \in [n-1, 0]$:
     - While `stack` and $nums[i] > nums[stack[-1]]$:
       - `right = max(right, stack.pop())`
     - `stack.append(i)`
4. Return `max(0, right - left + 1)`.

---

### Visual Algorithm Walkthrough

Let `nums = [2, 6, 4, 8, 10, 9, 15]`:

```
Index:      0   1   2   3   4   5   6
Values:     2   6   4   8  10   9  15

Left-to-Right Scan (tracking max_seen):
i=0: val=2,  max_seen=2
i=1: val=6,  max_seen=6
i=2: val=4,  max_seen=6 -> 4 < 6! Violation -> right = 2
i=3: val=8,  max_seen=8
i=4: val=10, max_seen=10
i=5: val=9,  max_seen=10 -> 9 < 10! Violation -> right = 5
i=6: val=15, max_seen=15
-> right = 5

Right-to-Left Scan (tracking min_seen):
j=6: val=15, min_seen=15
j=5: val=9,  min_seen=9
j=4: val=10, min_seen=9 -> 10 > 9! Violation -> left = 4
j=3: val=8,  min_seen=8
j=2: val=4,  min_seen=4
j=1: val=6,  min_seen=4 -> 6 > 4! Violation -> left = 1
j=0: val=2,  min_seen=2
-> left = 1

Unsorted Subarray Window: [left=1, right=5] -> [6, 4, 8, 10, 9]
Length = 5 - 1 + 1 = 5
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Disordered Valley

- **Input:** `nums = [2, 6, 4, 8, 10, 9, 15]`
- **Output:** `5` (Subarray `[6, 4, 8, 10, 9]` sorted $\implies [2, 4, 6, 8, 9, 10, 15]$)

#### Example 2: Already Sorted Array

- **Input:** `nums = [1, 2, 3, 4]`
- **Tracing:** No violations in either direction. `right` remains $-2$, `left` remains $-1$.
- **Output:** `0`

#### Example 3: Completely Inverted Array

- **Input:** `nums = [5, 4, 3, 2, 1]`
- **Tracing:** Every element violates sorted order. `left = 0`, `right = 4`.
- **Output:** `5`

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(1)$ Space Approach
```python
from typing import List

class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_seen = float('-inf')
        min_seen = float('inf')
        left, right = -1, -2

        for i in range(n):
            max_seen = max(max_seen, nums[i])
            if nums[i] < max_seen:
                right = i

            j = n - 1 - i
            min_seen = min(min_seen, nums[j])
            if nums[j] > min_seen:
                left = j

        return max(0, right - left + 1)
```

##### Monotonic Stack Approach ($\mathcal{O}(N)$ Space)
```python
class SolutionStack:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        stack = []
        left = n
        right = 0

        # Find left boundary with monotonic increasing stack
        for i in range(n):
            while stack and nums[i] < nums[stack[-1]]:
                left = min(left, stack.pop())
            stack.append(i)

        stack.clear()

        # Find right boundary with monotonic decreasing stack
        for i in range(n - 1, -1, -1):
            while stack and nums[i] > nums[stack[-1]]:
                right = max(right, stack.pop())
            stack.append(i)

        return max(0, right - left + 1)
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
        int max_seen = INT_MIN;
        int min_seen = INT_MAX;
        int left = -1, right = -2;

        for (int i = 0; i < n; ++i) {
            max_seen = std::max(max_seen, nums[i]);
            if (nums[i] < max_seen) {
                right = i;
            }

            int j = n - 1 - i;
            min_seen = std::min(min_seen, nums[j]);
            if (nums[j] > min_seen) {
                left = j;
            }
        }

        return std::max(0, right - left + 1);
    }
};
```

#### Java

```java
public class Solution {
    public int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        int maxSeen = Integer.MIN_VALUE;
        int minSeen = Integer.MAX_VALUE;
        int left = -1, right = -2;

        for (int i = 0; i < n; i++) {
            maxSeen = Math.max(maxSeen, nums[i]);
            if (nums[i] < maxSeen) {
                right = i;
            }

            int j = n - 1 - i;
            minSeen = Math.min(minSeen, nums[j]);
            if (nums[j] > minSeen) {
                left = j;
            }
        }

        return Math.max(0, right - left + 1);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - A single linear loop simultaneously updates both left and right pointers in $\mathcal{O}(N)$ time.
- **Space Complexity:**
  - Optimal Running Extrema Approach: $\mathcal{O}(1)$ auxiliary space using primitive scalar variables.
  - Stack Approach: $\mathcal{O}(N)$ auxiliary space for indices.

---

### Takeaway Pattern & Interview Traps

1. **Initial Boundary Sentinels (`left = -1, right = -2`):**
   - Initializing `right = -2` and `left = -1` makes the formula $right - left + 1 = -2 - (-1) + 1 = 0$. If the array is already sorted, no updates happen, and the method cleanly returns $0$ without additional `if` branches.
2. **Duplicate Values:**
   - In non-decreasing order ($nums[i] \le nums[i+1]$), duplicates are valid. Using strict inequalities `nums[i] < max_seen` and `nums[j] > min_seen` ensures identical adjacent values don't trigger false violations.