---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit"
tags:
  - leetcode
  - coding
  - queue
  - monotonic-queue
  - sliding-window
  - google
  - amazon
---

# LeetCode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

**Target Companies:** Google, Amazon, Uber, Microsoft  
**Difficulty:** Medium  
**Topic:** Dual Monotonic Deques (Min & Max) with Dynamic Sliding Window  

---

### Problem Statement

Given an array of integers `nums` and an integer `limit`, return the size of the **longest continuous subarray** such that the **absolute difference between any two elements** of this subarray is less than or equal to `limit`.

If there is no continuous subarray satisfying this condition, return `0`.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `limit: int`
- **Output:** `int` (maximum length)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $1 \le \text{nums}[i] \le 10^9$
  - $0 \le \text{limit} \le 10^9$

---

### Key Idea & Intuition

The condition that *the absolute difference between any two elements in a subarray is $\le \text{limit}$* is mathematically equivalent to:
$$\max(\text{subarray}) - \min(\text{subarray}) \le \text{limit}$$

If we use a multiset or balanced BST, we get $O(N \log N)$ runtime.
To achieve optimal $O(N)$ linear time:
1. We maintain a variable-length sliding window `[left ... right]`.
2. To query the running maximum and minimum of the active window in $O(1)$ time, we use **Two Monotonic Deques**:
   - `max_dq`: Monotonic **decreasing** deque. `max_dq[0]` holds the maximum element in the window.
   - `min_dq`: Monotonic **increasing** deque. `min_dq[0]` holds the minimum element in the window.
3. For each incoming element `nums[right]`:
   - Evict elements smaller than `nums[right]` from the back of `max_dq`.
   - Evict elements larger than `nums[right]` from the back of `min_dq`.
   - Push `nums[right]` to both deques.
4. If `max_dq[0] - min_dq[0] > limit`:
   - The window has violated the constraint.
   - Contract from the left: if `nums[left] == max_dq[0]`, pop from `max_dq`; if `nums[left] == min_dq[0]`, pop from `min_dq`.
   - Increment `left++`.
5. Update maximum window size: `ans = max(ans, right - left + 1)`.

---

### Solution Approach (Step-by-Step)

1. Initialize `max_dq = deque()`, `min_dq = deque()`, `left = 0`, `ans = 0`.
2. Loop `right` from $0$ to $n - 1$:
   - Maintain `max_dq`: while `max_dq` and `max_dq[-1] < nums[right]`: `max_dq.pop()`. Append `nums[right]`.
   - Maintain `min_dq`: while `min_dq` and `min_dq[-1] > nums[right]`: `min_dq.pop()`. Append `nums[right]`.
   - While `max_dq[0] - min_dq[0] > limit`:
     - If `nums[left] == max_dq[0]`: `max_dq.popleft()`.
     - If `nums[left] == min_dq[0]`: `min_dq.popleft()`.
     - `left += 1`.
   - `ans = max(ans, right - left + 1)`.
3. Return `ans`.

---

### Visual Algorithm Walkthrough

```
nums = [8, 2, 4, 7], limit = 4

right = 0 (val = 8):
  max_dq = [8], min_dq = [8]
  max - min = 8 - 8 = 0 <= 4. Valid!
  Window: [8], length = 1. ans = 1.

right = 1 (val = 2):
  max_dq: 2 < 8 -> push 2 -> [8, 2]
  min_dq: 2 < 8 -> pop 8, push 2 -> [2]
  max - min = 8 - 2 = 6 > 4 (INVALID!)
  Contract left:
    nums[left=0] is 8 == max_dq[0] -> popleft max_dq -> max_dq = [2]
    left++ -> left = 1
  Now max - min = 2 - 2 = 0 <= 4. Valid!
  Window: [2], length = 1. ans = 1.

right = 2 (val = 4):
  max_dq: 4 > 2 -> pop 2, push 4 -> [4]
  min_dq: 4 > 2 -> push 4 -> [2, 4]
  max - min = 4 - 2 = 2 <= 4. Valid!
  Window: [2, 4], length = 2. ans = max(1, 2) = 2.

right = 3 (val = 7):
  max_dq: 7 > 4 -> pop 4, push 7 -> [7]
  min_dq: 7 > 4 -> push 7 -> [2, 4, 7]
  max - min = 7 - 2 = 5 > 4 (INVALID!)
  Contract left:
    nums[left=1] is 2 == min_dq[0] -> popleft min_dq -> min_dq = [4, 7]
    left++ -> left = 2
  Now max - min = 7 - 4 = 3 <= 4. Valid!
  Window: [4, 7], length = 2. ans = max(2, 2) = 2.

Final answer = 2 (valid subarrays: [2, 4] and [4, 7]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Input
- **Input:** `nums = [8, 2, 4, 7]`, `limit = 4`
- **Output:** `2`

#### Example 2: Larger Span with Interleaved Extremes
- **Input:** `nums = [10, 1, 2, 4, 7, 2]`, `limit = 5`
- **Trace:**
  - Subarray `[2, 4, 7, 2]` has $\max = 7, \min = 2, 7 - 2 = 5 \le 5$. Length = 4.
- **Output:** `4`

#### Example 3: Zero Limit
- **Input:** `nums = [4, 2, 2, 2, 4, 4, 2, 2]`, `limit = 0`
- **Trace:** Requires all elements in subarray to be identical. Longest streak of identical elements is `[2, 2, 2]`.
- **Output:** `3`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        max_dq = deque()  # Monotonic decreasing (front is max)
        min_dq = deque()  # Monotonic increasing (front is min)
        
        left = 0
        ans = 0
        
        for right, num in enumerate(nums):
            # Maintain max_dq
            while max_dq and max_dq[-1] < num:
                max_dq.pop()
            max_dq.append(num)
            
            # Maintain min_dq
            while min_dq and min_dq[-1] > num:
                min_dq.pop()
            min_dq.append(num)
            
            # Contract window from left if difference exceeds limit
            while max_dq[0] - min_dq[0] > limit:
                if nums[left] == max_dq[0]:
                    max_dq.popleft()
                if nums[left] == min_dq[0]:
                    min_dq.popleft()
                left += 1
                
            ans = max(ans, right - left + 1)
            
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>
#include <algorithm>

class Solution {
public:
    int longestSubarray(std::vector<int>& nums, int limit) {
        std::deque<int> maxDq; // decreasing
        std::deque<int> minDq; // increasing
        
        int left = 0;
        int ans = 0;
        int n = nums.size();

        for (int right = 0; right < n; ++right) {
            while (!maxDq.empty() && maxDq.back() < nums[right]) {
                maxDq.pop_back();
            }
            maxDq.push_back(nums[right]);

            while (!minDq.empty() && minDq.back() > nums[right]) {
                minDq.pop_back();
            }
            minDq.push_back(nums[right]);

            while (maxDq.front() - minDq.front() > limit) {
                if (nums[left] == maxDq.front()) {
                    maxDq.pop_front();
                }
                if (nums[left] == minDq.front()) {
                    minDq.pop_front();
                }
                left++;
            }

            ans = std::max(ans, right - left + 1);
        }

        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int longestSubarray(int[] nums, int limit) {
        Deque<Integer> maxDq = new ArrayDeque<>(); // decreasing
        Deque<Integer> minDq = new ArrayDeque<>(); // increasing

        int left = 0;
        int ans = 0;

        for (int right = 0; right < nums.length; right++) {
            while (!maxDq.isEmpty() && maxDq.peekLast() < nums[right]) {
                maxDq.pollLast();
            }
            maxDq.offerLast(nums[right]);

            while (!minDq.isEmpty() && minDq.peekLast() > nums[right]) {
                minDq.pollLast();
            }
            minDq.offerLast(nums[right]);

            while (maxDq.peekFirst() - minDq.peekFirst() > limit) {
                if (nums[left] == maxDq.peekFirst()) {
                    maxDq.pollFirst();
                }
                if (nums[left] == minDq.peekFirst()) {
                    minDq.pollFirst();
                }
                left++;
            }

            ans = Math.max(ans, right - left + 1);
        }

        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Every element in `nums` enters both `max_dq` and `min_dq` at most once and is popped at most once. The pointer `left` only increments forward.
- **Space Complexity:** $O(N)$ auxiliary space — Both monotonic deques store at most $N$ elements in the worst case.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Dual Monotonic Deque Pattern for Sliding Window Dynamic Extremes.
- **Trap:** Forgetting that both deques can contain duplicate values: when popping from the back during maintenance, use strict inequality (`<` or `>`) rather than `<=` or `>=` so duplicate values are preserved until their specific index exits the window.