---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 239: Sliding Window Maximum"
tags:
  - leetcode
  - coding
  - queue
  - monotonic-queue
  - sliding-window
  - amazon
  - google
---

# LeetCode 239: Sliding Window Maximum

**Target Companies:** Amazon (Top #1 Signature Hard), Google, Meta, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** Monotonic Decreasing Deque for Sliding Window Extremes  

---

### Problem Statement

You are given an array of integers `nums`, there is a sliding window of size $k$ which is moving from the very left of the array to the very right. You can only see the $k$ numbers in the window. Each time the sliding window moves right by one position.

Return the **max sliding window** for each window position as an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `List[int]` (array of size $n - k + 1$)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$
  - $1 \le k \le \text{nums.length}$

---

### Key Idea & Intuition

A naive scan for each window costs $O(k)$, leading to an overall $O(N \cdot k)$ runtime, which triggers a TLE when $N, k \approx 10^5$ ($10^{10}$ operations). A max-heap costs $O(N \log k)$ or $O(N \log N)$ with lazy deletion.

To achieve strict $O(N)$ linear time:
1. We maintain a **Monotonic Decreasing Deque** of array indices.
2. The values corresponding to the indices in the deque are stored in strictly decreasing order:
   $$\text{nums}[q[0]] > \text{nums}[q[1]] > \dots > \text{nums}[q[-1]]$$
3. When considering a new element `nums[i]`:
   - Any older element in the deque whose value is smaller than or equal to `nums[i]` can **never** be the maximum of the current or any future window because `nums[i]` is both **larger** and **younger** (will survive longer). Therefore, we pop them from the back!
4. **Window Expiration:** The front of the deque `q[0]` always holds the maximum candidate. If `q[0] < i - k + 1`, it has drifted out of the sliding window boundary and must be evicted from the front.
5. Once the window index $i \ge k - 1$, `nums[q[0]]` is immediately recorded as the current window maximum in $O(1)$ time.

---

### Solution Approach (Step-by-Step)

1. Initialize `q = deque()` to store indices, and `res = []`.
2. Iterate `i` and `num` through `nums`:
   - **Monotonicity Maintenance:** While `q` is non-empty and `nums[q[-1]] <= num`, `q.pop()`.
   - **Push Current Index:** Append `i` to the back of `q`.
   - **Boundary Eviction:** If `q[0] == i - k`, pop the expired index from the front: `q.popleft()`.
   - **Record Result:** If $i \ge k - 1$, record the current window's maximum: `res.append(nums[q[0]])`.
3. Return `res`.

---

### Visual Algorithm Walkthrough

```
nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3

i=0, num=1:
  q = [0 (val=1)]
  Window not ready yet (< k=3).

i=1, num=3:
  nums[q[-1]] (1) <= 3 -> pop 0
  q = [1 (val=3)]

i=2, num=-1:
  -1 < 3 -> push 2
  q = [1 (val=3), 2 (val=-1)]
  Window [1, 3, -1] complete! Max is nums[q[0]] = 3.
  res = [3]

i=3, num=-3:
  -3 < -1 -> push 3
  q = [1 (val=3), 2 (val=-1), 3 (val=-3)]
  Window [3, -1, -3] complete! Max is nums[q[0]] = 3.
  res = [3, 3]

i=4, num=5:
  nums[q[-1]] (-3) <= 5 -> pop 3
  nums[q[-1]] (-1) <= 5 -> pop 2
  nums[q[-1]] (3)  <= 5 -> pop 1
  q = [4 (val=5)]
  Window [-1, -3, 5] complete! Max is nums[q[0]] = 5.
  res = [3, 3, 5]

i=5, num=3:
  3 < 5 -> push 5
  q = [4 (val=5), 5 (val=3)]
  Window [-3, 5, 3] complete! Max = 5.
  res = [3, 3, 5, 5]

i=6, num=6:
  Pop 5 (val 3) and 4 (val 5) -> q = [6 (val=6)]
  res = [3, 3, 5, 5, 6]

i=7, num=7:
  Pop 6 (val 6) -> q = [7 (val=7)]
  res = [3, 3, 5, 5, 6, 7]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Values
- **Input:** `nums = [1, 3, -1, -3, 5, 3, 6, 7]`, `k = 3`
- **Output:** `[3, 3, 5, 5, 6, 7]`

#### Example 2: Single Element Window ($k = 1$)
- **Input:** `nums = [1]`, `k = 1`
- **Output:** `[1]`

#### Example 3: Strictly Decreasing Array
- **Input:** `nums = [9, 8, 7, 6, 5]`, `k = 3`
- **Trace:**
  - `[9, 8, 7]` -> max = 9
  - `[8, 7, 6]` -> max = 8 (index 0 evicted)
  - `[7, 6, 5]` -> max = 7 (index 1 evicted)
- **Output:** `[9, 8, 7]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q = deque()  # stores indices, values are strictly decreasing
        res: List[int] = []
        
        for i, num in enumerate(nums):
            # 1. Maintain decreasing monotonic order from back
            while q and nums[q[-1]] <= num:
                q.pop()
            q.append(i)
            
            # 2. Evict index that slid out of the window from front
            if q[0] <= i - k:
                q.popleft()
                
            # 3. Record maximum once window reaches size k
            if i >= k - 1:
                res.append(nums[q[0]])
                
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>

class Solution {
public:
    std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k) {
        int n = nums.size();
        std::vector<int> res;
        res.reserve(n - k + 1);
        std::deque<int> dq; // stores indices

        for (int i = 0; i < n; ++i) {
            // Evict expired front element
            if (!dq.empty() && dq.front() <= i - k) {
                dq.pop_front();
            }

            // Maintain decreasing monotonic queue
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }

            dq.push_back(i);

            // Record maximum for window
            if (i >= k - 1) {
                res.push_back(nums[dq.front()]);
            }
        }

        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] res = new int[n - k + 1];
        Deque<Integer> dq = new ArrayDeque<>(); // stores indices

        for (int i = 0; i < n; i++) {
            // Evict expired element from front
            if (!dq.isEmpty() && dq.peekFirst() <= i - k) {
                dq.pollFirst();
            }

            // Maintain monotonic decreasing order
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) {
                dq.pollLast();
            }

            dq.offerLast(i);

            // Record current window max
            if (i >= k - 1) {
                res[i - k + 1] = nums[dq.peekFirst()];
            }
        }

        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Every index is added to the deque exactly once and removed from the deque at most once across the entire traversal. All push, pop, and peek operations take $O(1)$.
- **Space Complexity:** $O(k)$ auxiliary space — At any time, the deque holds at most $k$ indices corresponding to the active sliding window.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Monotonic Decreasing Deque for Sliding Window Extremes. Generalizes to 2D matrices, running maximum/minimum difference calculations, and bounded shortest subarray queries.
- **Trap:** Storing values instead of indices in the deque. Storing indices is essential so you can verify in $O(1)$ whether the maximum candidate has drifted out of the sliding window (`q[0] <= i - k`).