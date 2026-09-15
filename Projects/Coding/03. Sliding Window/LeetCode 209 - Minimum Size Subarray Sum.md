---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 209: Minimum Size Subarray Sum"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - binary-search
  - two-pointers
  - amazon
  - google
---

# LeetCode 209: Minimum Size Subarray Sum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Sliding Window / Two Pointers / Minimal Length Search  

---

### Problem Statement

Given an array of positive integers `nums` and a positive integer `target`, return *the **minimal length** of a subarray whose sum is greater than or equal to `target`*. If there is no such subarray, return `0` instead.

---

### Input & Output Formats & Constraints

- **Input:** `target: int`, `nums: List[int]`
- **Output:** `int` (minimum length, or `0` if no such subarray exists)
- **Constraints:**
  - $1 \le \text{target} \le 10^9$
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^4$

---

### Key Idea & Intuition

- **Why Sliding Window Works ($\text{nums}[i] > 0$):**
  - All elements in `nums` are strictly positive integers.
  - This strict positivity ensures monotonic behavior:
    - Moving the `right` pointer forward **strictly increases** the window sum.
    - Moving the `left` pointer forward **strictly decreases** the window sum.
- **Contracting Sliding Window:**
  - Expand $R$: add `nums[R]` to `curr_sum`.
  - While `curr_sum >= target`:
    - The window $[L, R]$ satisfies the condition.
    - Record candidate length: $\text{min\_len} = \min(\text{min\_len}, R - L + 1)$.
    - Attempt to find a smaller valid window by shedding `nums[L]` and advancing $L$:
      `curr_sum -= nums[L]; L += 1`.
  - Because each pointer moves at most $N$ times from left to right, this guarantees $\mathcal{O}(N)$ time.
- **Follow-up: $\mathcal{O}(N \log N)$ Binary Search Solution:**
  - Compute the prefix sum array `prefix_sum`.
  - Since elements are positive, `prefix_sum` is strictly increasing.
  - For each index $i$, find the smallest index $j$ such that $\text{prefix\_sum}[j] - \text{prefix\_sum}[i] \ge \text{target}$ using binary search (`bisect_left`). This takes $\mathcal{O}(N \log N)$ time.

---

### Solution Approach (Step-by-Step)

1. Initialize `left = 0`, `curr_sum = 0`, and `min_len = float('inf')`.
2. Iterate `right` from $0$ to $\text{len}(nums) - 1$:
   - `curr_sum += nums[right]`
   - While `curr_sum >= target`:
     - `min_len = min(min_len, right - left + 1)`
     - `curr_sum -= nums[left]`
     - `left += 1`
3. Return `0` if `min_len == float('inf')` else `min_len`.

---

### Visual Algorithm Walkthrough

Let `nums = [2, 3, 1, 2, 4, 3]`, `target = 7`.

```
Indices:   0  1  2  3  4  5
Nums:     [2, 3, 1, 2, 4, 3]

R=0 (2): sum = 2 < 7
R=1 (3): sum = 5 < 7
R=2 (1): sum = 6 < 7
R=3 (2): sum = 8 >= 7!
         - Window [0..3]: [2, 3, 1, 2], len = 4. min_len = 4.
         - Contract L=0: sum = 8 - 2 = 6 < 7. L = 1.
R=4 (4): sum = 6 + 4 = 10 >= 7!
         - Window [1..4]: [3, 1, 2, 4], len = 4. min_len = 4.
         - Contract L=1: sum = 10 - 3 = 7 >= 7!
           - Window [2..4]: [1, 2, 4], len = 3. min_len = 3!
           - Contract L=2: sum = 7 - 1 = 6 < 7. L = 3.
R=5 (3): sum = 6 + 3 = 9 >= 7!
         - Window [3..5]: [2, 4, 3], len = 3. min_len = 3.
         - Contract L=3: sum = 9 - 2 = 7 >= 7!
           - Window [4..5]: [4, 3], len = 2. min_len = 2!
           - Contract L=4: sum = 7 - 4 = 3 < 7. L = 5.

Final minimum length: 2 (Subarray [4, 3]).
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `target` | Shortest Subarray with Sum $\ge target$ | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[2,3,1,2,4,3]` | `7` | `[4, 3]` (sum 7) | `2` |
| **Single Element Match** | `[1, 4, 4]` | `4` | `[4]` (sum 4) | `1` |
| **No Solution** | `[1, 1, 1, 1, 1, 1, 1, 1]` | `11` | Total sum $= 8 < 11$ | `0` |
| **Exact Total Sum** | `[1, 2, 3, 4]` | `10` | Whole array `[1, 2, 3, 4]` | `4` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        """
        Finds the minimal length of a contiguous subarray whose sum >= target.
        Uses a dynamic contracting sliding window.
        """
        left = 0
        curr_sum = 0
        min_len = float('inf')

        for right, val in enumerate(nums):
            curr_sum += val

            # Contract window while condition is satisfied
            while curr_sum >= target:
                min_len = min(min_len, right - left + 1)
                curr_sum -= nums[left]
                left += 1

        return 0 if min_len == float('inf') else min_len
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int minSubArrayLen(int target, const std::vector<int>& nums) {
        int left = 0;
        long long curr_sum = 0;
        int min_len = INT_MAX;
        int n = static_cast<int>(nums.size());

        for (int right = 0; right < n; ++right) {
            curr_sum += nums[right];

            while (curr_sum >= target) {
                min_len = std::min(min_len, right - left + 1);
                curr_sum -= nums[left++];
            }
        }

        return min_len == INT_MAX ? 0 : min_len;
    }
};
```

#### Java
```java
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        long currSum = 0;
        int minLen = Integer.MAX_VALUE;
        int n = nums.length;

        for (int right = 0; right < n; right++) {
            currSum += nums[right];

            while (currSum >= target) {
                minLen = Math.min(minLen, right - left + 1);
                currSum -= nums[left++];
            }
        }

        return minLen == Integer.MAX_VALUE ? 0 : minLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - The `right` pointer iterates from $0$ to $N - 1$.
  - The `left` pointer advances at most $N$ times across the entire algorithm.
  - Total pointer movements $\le 2N$, running in $< 5 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space as only scalar integer pointers and sums are stored.

---

### Takeaway Pattern & Interview Traps

- **Minimum Length vs Maximum Length Window:**
  - To find **maximum length**, we update `ans = max(ans, R - L + 1)` *after* contracting the window until it becomes valid.
  - To find **minimum length**, we update `ans = min(ans, R - L + 1)` *inside* the contraction loop while the window remains valid.
- **Return 0 When No Solution Exists:** Ensure that if `curr_sum` never reaches `target`, the function correctly returns `0` rather than `inf`.