---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1248: Count Number of Nice Subarrays"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - prefix-sum
  - two-pointers
  - amazon
  - google
---

# LeetCode 1248: Count Number of Nice Subarrays

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Sliding Window / Prefix Subarray Counting / "At Most K" Reduction  

---

### Problem Statement

Given an array of integers `nums` and an integer `k`. A continuous subarray is called **nice** if there are `k` odd numbers on it.

Return *the number of **nice** sub-arrays*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (total count of nice subarrays)
- **Constraints:**
  - $1 \le \text{nums.length} \le 5 \times 10^4$
  - $1 \le \text{nums}[i] \le 10^5$
  - $1 \le k \le \text{nums.length}$

---

### Key Idea & Intuition

- **Binary Array Transformation:**
  - If we replace every odd number with `1` and every even number with `0`, the problem becomes:
    $$\text{Find the number of continuous subarrays with sum equal to } k.$$
  - This is mathematically identical to LeetCode 930 (Binary Subarrays With Sum).
- **The "Exactly $K$" via "At Most $K$" Paradigm:**
  - Directly counting subarrays with *exactly* $k$ odd numbers using a single sliding window is tricky because shrinking $L$ when we hit $k$ odds might omit even numbers surrounding the boundaries.
  - However, counting subarrays with **at most** $k$ odd numbers is trivial with standard monotonic sliding window!
  - Therefore, we apply the inclusion-exclusion principle:
    $$\text{count}(\text{exact } k) = \text{at\_most}(k) - \text{at\_most}(k - 1)$$
- **Why `at_most(k)` Counts Subarrays Efficiently:**
  - For any valid window $[L, R]$ containing $\le k$ odd numbers, every subarray ending at index $R$ and starting at any index $i \in [L, R]$ also contains $\le k$ odd numbers.
  - The number of such valid subarrays is simply:
    $$\text{window\_length} = R - L + 1$$
  - Summing $(R - L + 1)$ across all $R$ computes the total count in $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space!

---

### Solution Approach (Step-by-Step)

1. Define helper function `at_most(goal)`:
   - If `goal < 0`: return 0.
   - Initialize `left = 0`, `odd_count = 0`, `total_subarrays = 0`.
   - Loop `right` from $0$ to $\text{len}(nums) - 1$:
     - If `nums[right] % 2 == 1`:
       - `odd_count += 1`
     - While `odd_count > goal`:
       - If `nums[left] % 2 == 1`:
         - `odd_count -= 1`
       - `left += 1`
     - `total_subarrays += (right - left + 1)`
   - Return `total_subarrays`.
2. Return `at_most(k) - at_most(k - 1)`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 2, 1, 1]`, `k = 3`.
Odd indicators: `[1, 1, 0, 1, 1]`.

```
at_most(3):
- R=0: [1], odds=1 <= 3 -> count += (0-0+1) = 1
- R=1: [1, 1], odds=2 <= 3 -> count += (1-0+1) = 2
- R=2: [1, 1, 2], odds=2 <= 3 -> count += (2-0+1) = 3
- R=3: [1, 1, 2, 1], odds=3 <= 3 -> count += (3-0+1) = 4
- R=4: [1, 1, 2, 1, 1], odds=4 > 3 -> Shrink L to 1:
       Window [1, 2, 1, 1] (indices 1..4), odds=3 <= 3
       count += (4-1+1) = 4
Total at_most(3) = 1 + 2 + 3 + 4 + 4 = 14

at_most(2):
- R=0: [1], odds=1 <= 2 -> count += 1
- R=1: [1, 1], odds=2 <= 2 -> count += 2
- R=2: [1, 1, 2], odds=2 <= 2 -> count += 3
- R=3: [1, 1, 2, 1], odds=3 > 2 -> Shrink L to 1:
       Window [1, 2, 1] (indices 1..3), odds=2
       count += (3-1+1) = 3
- R=4: [1, 2, 1, 1], odds=3 > 2 -> Shrink L to 2:
       Window [2, 1, 1] (indices 2..4), odds=2
       count += (4-2+1) = 3
Total at_most(2) = 1 + 2 + 3 + 3 + 3 = 12

Result = at_most(3) - at_most(2) = 14 - 12 = 2!
The 2 nice subarrays are:
- [1, 1, 2, 1] (indices 0..3)
- [1, 2, 1, 1] (indices 1..4)
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `k` | `at_most(k)` | `at_most(k-1)` | Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 1, 2, 1, 1]` | `3` | 14 | 12 | `2` |
| **No Nice Subarrays** | `[2, 4, 6]` | `1` | 0 | 0 | `0` |
| **With Surrounding Evens** | `[2, 2, 2, 1, 2, 2, 1, 2, 2, 2]` | `2` | 52 | 36 | `16` |
| **All Odds** | `[1, 3, 5]` | `2` | 5 | 3 | `2` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        """
        Counts the number of subarrays with exactly k odd numbers.
        Uses the reduction: exact(k) = at_most(k) - at_most(k - 1).
        """
        def at_most(goal: int) -> int:
            if goal < 0:
                return 0

            left = 0
            odd_count = 0
            total = 0

            for right in range(len(nums)):
                if nums[right] % 2 == 1:
                    odd_count += 1

                while odd_count > goal:
                    if nums[left] % 2 == 1:
                        odd_count -= 1
                    left += 1

                # Subarrays ending at right with start in [left, right]
                total += right - left + 1

            return total

        return at_most(k) - at_most(k - 1)
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int numberOfSubarrays(const std::vector<int>& nums, int k) {
        return atMost(nums, k) - atMost(nums, k - 1);
    }

private:
    int atMost(const std::vector<int>& nums, int goal) {
        if (goal < 0) return 0;

        int left = 0;
        int odd_count = 0;
        int total = 0;

        for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
            if (nums[right] % 2 == 1) {
                odd_count++;
            }

            while (odd_count > goal) {
                if (nums[left] % 2 == 1) {
                    odd_count--;
                }
                left++;
            }

            total += right - left + 1;
        }

        return total;
    }
};
```

#### Java
```java
class Solution {
    public int numberOfSubarrays(int[] nums, int k) {
        return atMost(nums, k) - atMost(nums, k - 1);
    }

    private int atMost(int[] nums, int goal) {
        if (goal < 0) return 0;

        int left = 0;
        int oddCount = 0;
        int total = 0;

        for (int right = 0; right < nums.length; right++) {
            if (nums[right] % 2 == 1) {
                oddCount++;
            }

            while (oddCount > goal) {
                if (nums[left] % 2 == 1) {
                    oddCount--;
                }
                left++;
            }

            total += right - left + 1;
        }

        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - Two passes of the sliding window (`at_most(k)` and `at_most(k - 1)`).
  - In each pass, both `right` and `left` pointers advance at most $N$ times.
  - Overall time is $\mathcal{O}(2N) = \mathcal{O}(N)$, finishing in $< 10 \text{ ms}$ for $N = 5 \times 10^4$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since only a few integer variables are maintained.

---

### Takeaway Pattern & Interview Traps

- **The `exact(K) = at_most(K) - at_most(K - 1)` Template:** This reduction is the universal gold standard for counting subarray problems with non-negative constraints (e.g. LeetCode 930, 992, 1248, 1358).
- **Subarrays Ending at `right`:** Adding `right - left + 1` at each step counts all valid subarrays ending at `right` without needing inner loops or combinations.