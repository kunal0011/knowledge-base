---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit"
tags:
  - leetcode
  - coding
  - sliding-window
  - monotonic-queue
  - deque
  - amazon
  - google
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

**Target Companies:** Google (Signature Problem), Uber, Meta, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Sliding Window / Dual Monotonic Deques / Min-Max Window Tracking  

---

### Problem Statement

Given an array of integers `nums` and an integer `limit`, return the size of the longest **non-empty** subarray such that the absolute difference between any two elements of this subarray is less than or equal to `limit`.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `limit: int`
- **Output:** `int` (maximum length of a valid subarray)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^9$
  - $0 \le \text{limit} \le 10^9$

---

### Key Idea & Intuition

- **Condition Equivalence:**
  - The absolute difference between *any* two elements in a subarray is $\le \text{limit}$ if and only if:
    $$\max(subarray) - \min(subarray) \le \text{limit}$$
- **The Challenge with Slopes:**
  - In a dynamic sliding window $[L, R]$, as $R$ expands and $L$ contracts, we need the window's maximum and minimum in $\mathcal{O}(1)$ time.
  - A multiset or two priority queues give $\mathcal{O}(N \log N)$ time.
  - To achieve optimal $\mathcal{O}(N)$ time, we use **Two Monotonic Deques**:
    1. **Monotonic Decreasing Deque (`max_deque`):**
       - Stores elements in decreasing order.
       - The front of the deque always holds the current window's **maximum**.
       - When pushing `nums[R]`, pop smaller elements from the back.
    2. **Monotonic Increasing Deque (`min_deque`):**
       - Stores elements in increasing order.
       - The front of the deque always holds the current window's **minimum**.
       - When pushing `nums[R]`, pop larger elements from the back.
- **Window Invariant:**
  - If `max_deque[0] - min_deque[0] > limit`, the window is invalid.
  - Increment $L$. If `nums[L] == max_deque[0]`, pop from `max_deque`. If `nums[L] == min_deque[0]`, pop from `min_deque`.

---

### Solution Approach (Step-by-Step)

1. Initialize two double-ended queues: `max_deque` and `min_deque`.
2. Initialize `left = 0` and `max_len = 0`.
3. Loop `right` from $0$ to $\text{len}(nums) - 1$:
   - **Maintain `max_deque`:** While `max_deque` is not empty and `max_deque[-1] < nums[right]`:
     - `max_deque.pop()`
   - Append `nums[right]` to `max_deque`.
   - **Maintain `min_deque`:** While `min_deque` is not empty and `min_deque[-1] > nums[right]`:
     - `min_deque.pop()`
   - Append `nums[right]` to `min_deque`.
   - **Contract Window:** While `max_deque[0] - min_deque[0] > limit`:
     - If `nums[left] == max_deque[0]`: `max_deque.popleft()`
     - If `nums[left] == min_deque[0]`: `min_deque.popleft()`
     - `left += 1`
   - Update `max_len = max(max_len, right - left + 1)`.
4. Return `max_len`.

---

### Visual Algorithm Walkthrough

Let `nums = [8, 2, 4, 7]`, `limit = 4`.

```
R=0, num=8:
- max_deque: [8]
- min_deque: [8]
- max - min = 8 - 8 = 0 <= 4 -> max_len = 1

R=1, num=2:
- max_deque: [8, 2]
- min_deque: pop 8, append 2 -> [2]
- max - min = 8 - 2 = 6 > 4 (INVALID!)
  Contract L=0: nums[0] == 8 == max_deque[0] -> popleft 8!
  L becomes 1.
  max_deque: [2], min_deque: [2] -> max - min = 0 <= 4.
- Window [1..1], length = 1.

R=2, num=4:
- max_deque: pop 2, append 4 -> [4]
- min_deque: [2, 4]
- max - min = 4 - 2 = 2 <= 4 -> max_len = max(1, 2 - 1 + 1) = 2.

R=3, num=7:
- max_deque: pop 4, append 7 -> [7]
- min_deque: [2, 4, 7]
- max - min = 7 - 2 = 5 > 4 (INVALID!)
  Contract L=1: nums[1] == 2 == min_deque[0] -> popleft 2!
  L becomes 2.
  max_deque: [7], min_deque: [4, 7]
  max - min = 7 - 4 = 3 <= 4!
- Window [2..3], length = 3 - 2 + 1 = 2.

Result max_len = 2 (Subarray [2, 4] or [4, 7]).
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | `limit` | Longest Valid Subarray | Max Length |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[8, 2, 4, 7]` | `4` | `[2, 4]` or `[4, 7]` | `2` |
| **Example 2** | `[10, 1, 2, 4, 7, 2]` | `5` | `[2, 4, 7, 2]` (diff: $7 - 2 = 5$) | `4` |
| **Example 3** | `[4, 2, 2, 2, 4, 4, 2, 2]` | `0` | `[2, 2, 2]` | `3` |
| **Monotonic Array** | `[1, 2, 3, 4, 5]` | `2` | `[1, 2, 3]` or `[2, 3, 4]` | `3` |

---

### Multi-Language Implementations

#### Python 3
```python
from collections import deque
from typing import List

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        """
        Finds the longest subarray where max - min <= limit.
        Uses two monotonic deques to maintain min and max in O(1) amortized time.
        """
        max_deque = deque()  # Decreasing order
        min_deque = deque()  # Increasing order
        left = 0
        max_len = 0

        for right, val in enumerate(nums):
            # Maintain decreasing max_deque
            while max_deque and max_deque[-1] < val:
                max_deque.pop()
            max_deque.append(val)

            # Maintain increasing min_deque
            while min_deque and min_deque[-1] > val:
                min_deque.pop()
            min_deque.append(val)

            # Shrink window if difference exceeds limit
            while max_deque[0] - min_deque[0] > limit:
                if nums[left] == max_deque[0]:
                    max_deque.popleft()
                if nums[left] == min_deque[0]:
                    min_deque.popleft()
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

#### C++17
```cpp
#include <vector>
#include <deque>
#include <algorithm>

class Solution {
public:
    int longestSubarray(const std::vector<int>& nums, int limit) {
        std::deque<int> max_deque;
        std::deque<int> min_deque;
        int left = 0;
        int max_len = 0;

        for (int right = 0; right < static_cast<int>(nums.size()); ++right) {
            int val = nums[right];

            while (!max_deque.empty() && max_deque.back() < val) {
                max_deque.pop_back();
            }
            max_deque.push_back(val);

            while (!min_deque.empty() && min_deque.back() > val) {
                min_deque.pop_back();
            }
            min_deque.push_back(val);

            while (max_deque.front() - min_deque.front() > limit) {
                if (nums[left] == max_deque.front()) {
                    max_deque.pop_front();
                }
                if (nums[left] == min_deque.front()) {
                    min_deque.pop_front();
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
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> maxDeque = new ArrayDeque<>();
        Deque<Integer> minDeque = new ArrayDeque<>();
        int left = 0;
        int maxLen = 0;

        for (int right = 0; right < nums.length; right++) {
            int val = nums[right];

            while (!maxDeque.isEmpty() && maxDeque.peekLast() < val) {
                maxDeque.pollLast();
            }
            maxDeque.addLast(val);

            while (!minDeque.isEmpty() && minDeque.peekLast() > val) {
                minDeque.pollLast();
            }
            minDeque.addLast(val);

            while (maxDeque.peekFirst() - minDeque.peekFirst() > limit) {
                if (nums[left] == maxDeque.peekFirst()) {
                    maxDeque.pollFirst();
                }
                if (nums[left] == minDeque.peekFirst()) {
                    minDeque.pollFirst();
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
  - Each element is pushed and popped from `max_deque` at most once.
  - Each element is pushed and popped from `min_deque` at most once.
  - The `left` pointer advances at most $N$ times.
  - Total operations are at most $4N$, executing in $< 25 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space in the worst case to store elements in the two deques.

---

### Takeaway Pattern & Interview Traps

- **Dual Monotonic Deque Pattern:** When a problem requires both dynamic maximum and dynamic minimum over a sliding window simultaneously, using two opposing monotonic deques (decreasing for max, increasing for min) guarantees $\mathcal{O}(1)$ query time and $\mathcal{O}(N)$ total runtime.
- **Popping by Value:** When contracting the left edge, check `nums[left] == max_deque[0]` and `nums[left] == min_deque[0]` independently. Notice that an element might be the maximum, the minimum, or neither.