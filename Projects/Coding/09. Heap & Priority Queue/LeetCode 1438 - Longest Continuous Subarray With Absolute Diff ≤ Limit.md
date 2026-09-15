---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff ≤ Limit"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - sliding-window
  - monotonic-queue
  - amazon
  - google
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff ≤ Limit

**Target Companies:** Google, Amazon, Uber, Bloomberg, Meta  
**Difficulty:** Medium  
**Topic:** Priority Queue (Two Heaps) / Monotonic Deque / Sliding Window

---

### Problem Statement

Given an array of integers `nums` and an integer `limit`, return the size of the longest **non-empty** subarray such that the absolute difference between any two elements of this subarray is less than or equal to `limit`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 10^5$.
  - `limit`: `int`, where $0 \le limit \le 10^9$.
  - $1 \le nums[i] \le 10^9$.
- **Output:**
  - `int`: Maximum length of a contiguous subarray where $\max(\text{window}) - \min(\text{window}) \le limit$.
- **Constraints:**
  - The absolute difference condition $\max - \min \le limit$ must hold for all pairs in the subarray.

---

### Key Idea & Intuition

For any subarray $nums[L \dots R]$, the condition that $|nums[i] - nums[j]| \le limit$ for *all* $i, j \in [L, R]$ is mathematically equivalent to:
$$\max_{i \in [L, R]}(nums[i]) - \min_{i \in [L, R]}(nums[i]) \le limit$$

As the right boundary $R$ advances, the window grows. If at any point $\max - \min > limit$, we must increment the left boundary $L$ to shrink the window until the difference becomes $\le limit$.

To query the dynamic minimum and maximum in the window, we have two standard patterns:

#### Approach 1: Dual Heaps with Lazy Deletion ($\mathcal{O}(N \log N)$ Time, $\mathcal{O}(N)$ Space)
- Maintain a **max-heap** for window maximums: `(-val, idx)`.
- Maintain a **min-heap** for window minimums: `(val, idx)`.
- When $\max - \min > limit$, increment $L$ by 1, and lazily pop from both heaps whenever their top element has an index $< L$.

#### Approach 2: Dual Monotonic Deques ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space - Optimal)
- Maintain a **monotonic decreasing deque** `max_dq` where front is always the window maximum.
- Maintain a **monotonic increasing deque** `min_dq` where front is always the window minimum.
- Each element is pushed and popped at most once across both deques, yielding strictly $\mathcal{O}(N)$ linear time!

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Dual Heaps with Lazy Deletion
1. Initialize `min_heap = []`, `max_heap = []`, `left = 0`, `max_len = 0`.
2. For each `(right, num)` in `enumerate(nums)`:
   - Push `(num, right)` to `min_heap`.
   - Push `(-num, right)` to `max_heap`.
   - While `-max_heap[0][0] - min_heap[0][0] > limit`:
     - Increment `left += 1`.
     - Evict expired elements:
       - While `min_heap[0][1] < left`: `heappop(min_heap)`
       - While `max_heap[0][1] < left`: `heappop(max_heap)`
   - `max_len = max(max_len, right - left + 1)`.
3. Return `max_len`.

#### Algorithm 2: Dual Monotonic Deques ($\mathcal{O}(N)$ Optimal)
1. Initialize `min_dq = deque()`, `max_dq = deque()`, `left = 0`, `max_len = 0`.
2. For each `right` from $0$ to $N - 1$:
   - Maintain decreasing order in `max_dq`: pop from back while `max_dq[-1] < nums[right]`.
   - Maintain increasing order in `min_dq`: pop from back while `min_dq[-1] > nums[right]`.
   - Append `nums[right]` to both deques.
   - While `max_dq[0] - min_dq[0] > limit`:
     - If `max_dq[0] == nums[left]`: `max_dq.popleft()`.
     - If `min_dq[0] == nums[left]`: `min_dq.popleft()`.
     - `left += 1`.
   - `max_len = max(max_len, right - left + 1)`.
3. Return `max_len`.

---

### Visual Algorithm Walkthrough

Let `nums = [8, 2, 4, 7]`, `limit = 4`:

```
Indices:   0   1   2   3
Values:    8   2   4   7

R=0, val=8:
  min_dq = [8], max_dq = [8]
  max - min = 8 - 8 = 0 <= 4. Window [0..0], len = 1.

R=1, val=2:
  min_dq: 2 < 8 -> pop 8 -> [2]
  max_dq: 2 < 8 -> [8, 2]
  max - min = 8 - 2 = 6 > 4! VIOLATION!
  Shrink window:
    nums[left=0] is 8.
    8 == max_dq[0] -> max_dq.popleft() -> max_dq = [2]
    left becomes 1.
  Now max - min = 2 - 2 = 0 <= 4. Window [1..1], len = 1.

R=2, val=4:
  min_dq: 4 > 2 -> [2, 4]
  max_dq: 4 > 2 -> pop 2 -> [4]
  max - min = 4 - 2 = 2 <= 4. Window [1..2], len = 2.

R=3, val=7:
  min_dq: 7 > 4 -> [2, 4, 7]
  max_dq: 7 > 4 -> pop 4 -> [7]
  max - min = 7 - 2 = 5 > 4! VIOLATION!
  Shrink window:
    nums[left=1] is 2.
    2 == min_dq[0] -> min_dq.popleft() -> min_dq = [4, 7]
    left becomes 2.
  Now max - min = 7 - 4 = 3 <= 4. Window [2..3], len = 2.

Result: Maximum length = 2.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Disjoint Ranges

- **Input:** `nums = [8, 2, 4, 7]`, `limit = 4`
- **Output:** `2` (Subarrays `[2, 4]` or `[4, 7]`)

#### Example 2: Uniform Constant Array

- **Input:** `nums = [10, 1, 2, 4, 7, 2]`, `limit = 5`
- **Output:** `4` (Subarray `[2, 4, 7, 2]`, $\max - \min = 7 - 2 = 5 \le 5$)

#### Example 3: Limit Zero (Strictly Identical Adjacent Elements)

- **Input:** `nums = [4, 2, 2, 2, 4, 4, 2, 2]`, `limit = 0`
- **Output:** `3` (Subarray `[2, 2, 2]`)

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(N)$ Monotonic Deque Approach
```python
from collections import deque
from typing import List

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        max_dq = deque()  # stores elements in decreasing order
        min_dq = deque()  # stores elements in increasing order
        left = 0
        max_len = 0

        for right, val in enumerate(nums):
            while max_dq and max_dq[-1] < val:
                max_dq.pop()
            max_dq.append(val)

            while min_dq and min_dq[-1] > val:
                min_dq.pop()
            min_dq.append(val)

            # If current window violates limit, shrink from left
            while max_dq[0] - min_dq[0] > limit:
                if max_dq[0] == nums[left]:
                    max_dq.popleft()
                if min_dq[0] == nums[left]:
                    min_dq.popleft()
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

##### Priority Queue Approach ($\mathcal{O}(N \log N)$ Two Heaps)
```python
import heapq
from typing import List

class SolutionHeap:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_heap = []  # stores (value, index)
        max_heap = []  # stores (-value, index)
        left = 0
        max_len = 0

        for right, val in enumerate(nums):
            heapq.heappush(min_heap, (val, right))
            heapq.heappush(max_heap, (-val, right))

            while -max_heap[0][0] - min_heap[0][0] > limit:
                left += 1
                while min_heap[0][1] < left:
                    heapq.heappop(min_heap)
                while max_heap[0][1] < left:
                    heapq.heappop(max_heap)

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
        std::deque<int> max_dq;
        std::deque<int> min_dq;
        int left = 0;
        int max_len = 0;
        int n = static_cast<int>(nums.size());

        for (int right = 0; right < n; ++right) {
            int val = nums[right];

            while (!max_dq.empty() && max_dq.back() < val) {
                max_dq.pop_back();
            }
            max_dq.push_back(val);

            while (!min_dq.empty() && min_dq.back() > val) {
                min_dq.pop_back();
            }
            min_dq.push_back(val);

            while (max_dq.front() - min_dq.front() > limit) {
                if (max_dq.front() == nums[left]) {
                    max_dq.pop_front();
                }
                if (min_dq.front() == nums[left]) {
                    min_dq.pop_front();
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

public class Solution {
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> maxDq = new ArrayDeque<>();
        Deque<Integer> minDq = new ArrayDeque<>();
        int left = 0;
        int maxLen = 0;

        for (int right = 0; right < nums.length; right++) {
            int val = nums[right];

            while (!maxDq.isEmpty() && maxDq.peekLast() < val) {
                maxDq.pollLast();
            }
            maxDq.offerLast(val);

            while (!minDq.isEmpty() && minDq.peekLast() > val) {
                minDq.pollLast();
            }
            minDq.offerLast(val);

            while (maxDq.peekFirst() - minDq.peekFirst() > limit) {
                if (maxDq.peekFirst() == nums[left]) {
                    maxDq.pollFirst();
                }
                if (minDq.peekFirst() == nums[left]) {
                    minDq.pollFirst();
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

- **Time Complexity:**
  - **Monotonic Deque Approach:** $\mathcal{O}(N)$. Each element is inserted into each deque once and removed at most once. Amortized $\mathcal{O}(1)$ per window transition.
  - **Dual Heaps Approach:** $\mathcal{O}(N \log N)$. Each element is pushed into both heaps once and popped once.
- **Space Complexity:** $\mathcal{O}(N)$ to store elements/indices in the deques or heaps.

---

### Takeaway Pattern & Interview Traps

1. **Trade-off between Deque and Heap:**
   - Mentioning both Dual Heaps (with lazy deletion) and Dual Monotonic Deques shows complete mastery of dynamic sliding window extrema.
2. **Lazy Deletion Invariant:**
   - In the heap approach, you don't need to delete elements the exact instant they exit the sliding window. You only need to delete them when they reach the top of the heap and their index is $< left$.