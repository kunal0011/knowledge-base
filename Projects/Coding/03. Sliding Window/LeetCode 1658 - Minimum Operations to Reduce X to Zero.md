---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1658: Minimum Operations to Reduce X to Zero"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - two-pointers
  - prefix-sum
  - amazon
  - google
---

# LeetCode 1658: Minimum Operations to Reduce X to Zero

**Target Companies:** Amazon, Google, Meta, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Sliding Window / Complementary Subarray / Two Pointers  

---

### Problem Statement

You are given an integer array `nums` and an integer `x`. In one operation, you can either remove the leftmost or the rightmost element from the array `nums` and subtract its value from `x`. Note that this modifies the array for future operations.

Return *the **minimum number of operations** to reduce `x` to exactly `0` if it is possible, or `-1` otherwise*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `x: int`
- **Output:** `int` (minimum operations, or `-1` if impossible)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^4$
  - $1 \le x \le 10^9$

---

### Key Idea & Intuition

- **Inverting the Problem (The Complementary Subarray Invariant):**
  - Removing elements from the leftmost and rightmost ends until their sum equals $x$ leaves a **contiguous subarray in the middle**.
  - The sum of this remaining contiguous subarray must be:
    $$\text{target} = \sum_{i=0}^{N-1} \text{nums}[i] - x$$
  - To **minimize** the number of removed elements from the ends, we must **maximize the length of the contiguous subarray whose sum is exactly `target`**!
  - If the longest such subarray has length $L_{\max}$, the minimum number of operations to remove the ends is:
    $$\text{operations} = N - L_{\max}$$
- **Why Sliding Window Works ($\text{nums}[i] \ge 1$):**
  - Because all elements in `nums` are strictly positive ($\text{nums}[i] \ge 1$), the running window sum is strictly monotonic:
    - Expanding $R$ strictly increases the sum.
    - Contracting $L$ strictly decreases the sum.
  - This allows a standard $\mathcal{O}(N)$ two-pointer sliding window without negative number complications.
- **Edge Cases:**
  - If `target < 0` ($x > \text{total\_sum}$): impossible $\implies$ return `-1`.
  - If `target == 0` ($x == \text{total\_sum}$): all $N$ elements must be removed $\implies$ return $N$.

---

### Solution Approach (Step-by-Step)

1. Compute `total_sum = sum(nums)`.
2. Compute `target = total_sum - x`.
3. If `target < 0`: return `-1`.
4. If `target == 0`: return `len(nums)`.
5. Initialize `left = 0`, `curr_sum = 0`, and `max_len = -1`.
6. Loop `right` from $0$ to $\text{len}(nums) - 1$:
   - `curr_sum += nums[right]`
   - While `curr_sum > target and left <= right`:
     - `curr_sum -= nums[left]`
     - `left += 1`
   - If `curr_sum == target`:
     - `max_len = max(max_len, right - left + 1)`
7. If `max_len == -1`: return `-1`.
8. Return `len(nums) - max_len`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 4, 2, 3]`, $x = 5$.
- Total sum $= 1 + 1 + 4 + 2 + 3 = 11$.
- Target middle sum $= \text{Total} - x = 11 - 5 = 6$.
- Goal: Find the longest contiguous subarray with sum equal to 6.

```
Indices:      0   1   2   3   4
Nums:        [1,  1,  4,  2,  3]

R=0 (1): sum=1 < 6
R=1 (1): sum=2 < 6
R=2 (4): sum=6 == 6! Window [0..2] ([1, 1, 4]), length = 3. max_len = 3.
R=3 (2): sum=8 > 6 -> shrink L:
         L=1: sum = 8 - 1 = 7 > 6
         L=2: sum = 7 - 1 = 6 == 6! Window [2..3] ([4, 2]), length = 2.
R=4 (3): sum = 6 + 3 = 9 > 6 -> shrink L:
         L=3: sum = 9 - 4 = 5 < 6.
         Window [3..4] ([2, 3]), sum = 5.

Longest Subarray with sum 6: length 3 (indices 0..2: [1, 1, 4]).
Complementary elements removed from ends: indices 3 and 4 -> [2, 3] from the right.
Min operations = N - max_len = 5 - 3 = 2!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `x` | Total Sum | Target ($S - x$) | Longest Middle Subarray | Min Operations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[1, 1, 4, 2, 3]` | `5` | 11 | 6 | `[1, 1, 4]` (len 3) | `5 - 3 = 2` |
| **Example 2** | `[5, 6, 7, 8, 9]` | `4` | 35 | 31 | No subarray sums to 31 | `-1` |
| **Example 3** | `[3, 2, 20, 1, 1, 3]` | `10` | 30 | 20 | `[20]` (len 1) | `6 - 1 = 5` |
| **Remove All** | `[1, 2, 3]` | `6` | 6 | 0 | Whole array removed | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minOperations(self, nums: List[int], x: int) -> int:
        """
        Finds minimum operations to reduce x to 0 by removing end elements.
        Transforms problem into finding the longest subarray summing to sum(nums) - x.
        """
        total_sum = sum(nums)
        target = total_sum - x

        if target < 0:
            return -1
        if target == 0:
            return len(nums)

        left = 0
        curr_sum = 0
        max_len = -1

        for right, val in enumerate(nums):
            curr_sum += val

            while curr_sum > target and left <= right:
                curr_sum -= nums[left]
                left += 1

            if curr_sum == target:
                max_len = max(max_len, right - left + 1)

        return -1 if max_len == -1 else len(nums) - max_len
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int minOperations(const std::vector<int>& nums, int x) {
        int total_sum = std::accumulate(nums.begin(), nums.end(), 0);
        int target = total_sum - x;

        if (target < 0) return -1;
        if (target == 0) return static_cast<int>(nums.size());

        int left = 0;
        int curr_sum = 0;
        int max_len = -1;
        int n = static_cast<int>(nums.size());

        for (int right = 0; right < n; ++right) {
            curr_sum += nums[right];

            while (curr_sum > target && left <= right) {
                curr_sum -= nums[left++];
            }

            if (curr_sum == target) {
                max_len = std::max(max_len, right - left + 1);
            }
        }

        return max_len == -1 ? -1 : n - max_len;
    }
};
```

#### Java
```java
class Solution {
    public int minOperations(int[] nums, int x) {
        int totalSum = 0;
        for (int num : nums) {
            totalSum += num;
        }

        int target = totalSum - x;
        if (target < 0) return -1;
        if (target == 0) return nums.length;

        int left = 0;
        int currSum = 0;
        int maxLen = -1;
        int n = nums.length;

        for (int right = 0; right < n; right++) {
            currSum += nums[right];

            while (currSum > target && left <= right) {
                currSum -= nums[left++];
            }

            if (currSum == target) {
                maxLen = Math.max(maxLen, right - left + 1);
            }
        }

        return maxLen == -1 ? -1 : n - maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - Computing total sum takes $\mathcal{O}(N)$.
  - Both `right` and `left` pointers advance at most $N$ steps each.
  - Overall time is $\mathcal{O}(N)$, running in $< 15 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since only scalar two-pointer and sum counters are maintained.

---

### Takeaway Pattern & Interview Traps

- **Inverting the Objective:** When removing elements from both ends with sum $x$, the remaining center elements form a contiguous subarray with sum $\text{total} - x$. Maximizing the center minimizes the ends.
- **Handling `target == 0`:** If $x == \text{total\_sum}$, the required center sum is 0. Since $\text{nums}[i] \ge 1$, the only valid center subarray has length 0, meaning all $N$ elements are removed. Checking `if target == 0: return len(nums)` handles this gracefully.