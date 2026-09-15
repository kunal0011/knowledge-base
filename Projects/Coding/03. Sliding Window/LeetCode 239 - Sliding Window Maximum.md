---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 239: Sliding Window Maximum"
tags:
  - leetcode
  - coding
  - sliding-window
  - monotonic-queue
  - deque
  - heap
  - amazon
  - google
---

# LeetCode 239: Sliding Window Maximum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, ByteDance, Bloomberg  
**Difficulty:** Hard  
**Topic:** Sliding Window / Monotonic Queue / Deque  

---

### Problem Statement

You are given an array of integers `nums`, and there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — an array of integers.
  - `k`: `int` — sliding window width ($1 \le k \le \text{nums.length}$).
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — an array of size $n - k + 1$ containing the maximum element for each sliding window of length $k$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$
  - $1 \le k \le \text{nums.length}$

---

### Key Idea & Intuition

A naive approach takes $\mathcal{O}(k)$ time to find the maximum in each window, resulting in $\mathcal{O}(n \cdot k)$ total time, which times out when $n, k \sim 10^5$. Using a max-heap yields $\mathcal{O}(n \log k)$ or $\mathcal{O}(n \log n)$.

To achieve optimal $\mathcal{O}(n)$ time, we use a **Monotonic Decreasing Deque**:
1. The deque stores **indices** of elements in `nums` in strictly decreasing order of their corresponding values: `nums[deq[0]] > nums[deq[1]] > ... > nums[deq[-1]]`.
2. When considering index $i$ with value `nums[i]`:
   - Any index $j$ already in the deque where $\text{nums}[j] \le \text{nums}[i]$ can **never** be the maximum of the current window or any future window because `nums[i]` is both larger (or equal) and appears later (has a longer lifetime). Therefore, we pop from the back of the deque while `nums[deq.back()] <= nums[i]`.
   - Then, append $i$ to the back.
3. **Eviction of Out-of-Bound Indices:**
   - The current window covers indices $[i - k + 1, i]$. If the front index `deq.front() <= i - k`, it has fallen out of the window and must be popped from the front.
4. Once $i \ge k - 1$, the current window's maximum is always at `nums[deq.front()]`.

Because each index enters the deque at most once and is removed at most once, the total amortized cost across the entire array is $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. Initialize an empty double-ended queue `deq` (storing indices) and an output array `result`.
2. Iterate `i` from $0$ to $n - 1$:
   - **Step 2a (Evict Stale Head):** If `deq` is not empty and `deq.front() <= i - k`, remove the front element (`deq.pop_front()`).
   - **Step 2b (Maintain Monotonicity):** While `deq` is not empty and `nums[deq.back()] <= nums[i]`, remove the back element (`deq.pop_back()`).
   - **Step 2c (Insert Current Index):** Push `i` to the back of `deq`.
   - **Step 2d (Collect Maximum):** If $i \ge k - 1$, the front of `deq` contains the index of the window maximum. Append `nums[deq.front()]` to `result`.
3. Return `result`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 3, -1, -3, 5, 3, 6, 7]`, $k = 3$:

```
Index i = 0, val = 1:
  deq: [0 (val: 1)]
  i < k - 1: no output

Index i = 1, val = 3:
  Pop back 0 because nums[0]=1 <= 3
  deq: [1 (val: 3)]
  i < k - 1: no output

Index i = 2, val = -1:
  nums[1]=3 > -1, push 2
  deq: [1 (val: 3), 2 (val: -1)]
  i >= 2: Window [1, 3, -1] -> Max is nums[1] = 3. Result: [3]

Index i = 3, val = -3:
  deq front = 1 > 3 - 3: keep
  nums[2]=-1 > -3, push 3
  deq: [1 (val: 3), 2 (val: -1), 3 (val: -3)]
  Window [3, -1, -3] -> Max is nums[1] = 3. Result: [3, 3]

Index i = 4, val = 5:
  deq front = 1 <= 4 - 3 = 1 -> Pop front 1!
  nums[3]=-3 <= 5 -> pop back 3
  nums[2]=-1 <= 5 -> pop back 2
  push 4
  deq: [4 (val: 5)]
  Window [-1, -3, 5] -> Max is nums[4] = 5. Result: [3, 3, 5]

Index i = 5, val = 3:
  nums[4]=5 > 3, push 5
  deq: [4 (val: 5), 5 (val: 3)]
  Window [-3, 5, 3] -> Max is nums[4] = 5. Result: [3, 3, 5, 5]

Index i = 6, val = 6:
  nums[5]=3 <= 6 -> pop back 5
  nums[4]=5 <= 6 -> pop back 4
  push 6
  deq: [6 (val: 6)]
  Window [5, 3, 6] -> Max is nums[6] = 6. Result: [3, 3, 5, 5, 6]

Index i = 7, val = 7:
  nums[6]=6 <= 7 -> pop back 6
  push 7
  deq: [7 (val: 7)]
  Window [3, 6, 7] -> Max is nums[7] = 7. Result: [3, 3, 5, 5, 6, 7]
```

---

### Solved Examples with Multiple Inputs

#### Example 1 (Standard):
- **Input:** `nums = [1, 3, -1, -3, 5, 3, 6, 7]`, `k = 3`
- **Output:** `[3, 3, 5, 5, 6, 7]`

#### Example 2 (Monotonically Decreasing Array):
- **Input:** `nums = [9, 8, 7, 6, 5]`, `k = 3`
- **Tracing:**
  - $i=0, 1, 2$: Deque holds `[0, 1, 2]`. Front is $0$ ($9$).
  - $i=3$: Deque front $0$ evicted ($0 \le 3-3$). Front is $1$ ($8$).
  - $i=4$: Deque front $1$ evicted ($1 \le 4-3$). Front is $2$ ($7$).
- **Output:** `[9, 8, 7]`

#### Example 3 (Single Element Window):
- **Input:** `nums = [4, -2]`, `k = 1`
- **Output:** `[4, -2]`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums or k == 0:
            return []
        
        deq: deque[int] = deque()  # stores indices, values strictly decreasing
        result: List[int] = []
        
        for i, val in enumerate(nums):
            # 1. Remove indices that are outside the current window
            if deq and deq[0] <= i - k:
                deq.popleft()
            
            # 2. Maintain monotonic decreasing order in deque
            while deq and nums[deq[-1]] <= val:
                deq.pop()
            
            # 3. Add current index
            deq.append(i)
            
            # 4. Once window size reaches k, record the maximum
            if i >= k - 1:
                result.append(nums[deq[0]])
                
        return result
```

#### C++17
```cpp
#include <vector>
#include <deque>

class Solution {
public:
    std::vector<int> maxSlidingWindow(const std::vector<int>& nums, int k) {
        int n = static_cast<int>(nums.size());
        if (n == 0 || k == 0) return {};
        
        std::vector<int> result;
        result.reserve(n - k + 1);
        std::deque<int> deq; // Stores indices
        
        for (int i = 0; i < n; ++i) {
            // 1. Remove out-of-bound index from front
            if (!deq.empty() && deq.front() <= i - k) {
                deq.pop_front();
            }
            
            // 2. Pop smaller elements from back to maintain monotonic decreasing order
            while (!deq.empty() && nums[deq.back()] <= nums[i]) {
                deq.pop_back();
            }
            
            // 3. Push current index
            deq.push_back(i);
            
            // 4. Record maximum when window is of size k
            if (i >= k - 1) {
                result.push_back(nums[deq.front()]);
            }
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || k == 0) {
            return new int[0];
        }
        
        int n = nums.length;
        int[] result = new int[n - k + 1];
        int writeIdx = 0;
        
        // Deque stores array indices
        Deque<Integer> deq = new ArrayDeque<>();
        
        for (int i = 0; i < n; i++) {
            // 1. Remove index outside the sliding window
            if (!deq.isEmpty() && deq.peekFirst() <= i - k) {
                deq.pollFirst();
            }
            
            // 2. Remove indices with smaller or equal values from the tail
            while (!deq.isEmpty() && nums[deq.peekLast()] <= nums[i]) {
                deq.pollLast();
            }
            
            // 3. Add current element's index
            deq.offerLast(i);
            
            // 4. Record the max element when the first window of size k is formed
            if (i >= k - 1) {
                result[writeIdx++] = nums[deq.peekFirst()];
            }
        }
        
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Each element's index is pushed onto the deque exactly once.
  - Each element's index is popped from the deque at most once (either from the back during monotonic maintenance or from the front when falling out of bounds).
  - Hence, the amortized number of operations per element is $\mathcal{O}(1)$, giving total time $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(k)$ auxiliary space
  - The deque holds at most $k$ indices at any point in time.
  - Output space is $\mathcal{O}(n - k + 1)$ which is required for the answer.

---

### Takeaway Pattern & Interview Traps

- **Indices vs. Values in Deque:** Always store **indices**, not values, in the deque. Storing indices lets you immediately determine whether an element has expired (`deq.front() <= i - k`) without needing secondary counters or searches.
- **Strict Inequality `nums[deq.back()] <= nums[i]`:** Using `<=` instead of `<` is critical. If duplicate maximum values appear, discarding earlier occurrences is safe because the newest occurrence will survive longer in the sliding window.
- **Alternative Two-Pass Block Approach ($\mathcal{O}(n)$ time, $\mathcal{O}(n)$ space):**
  - Divide array into blocks of size $k$.
  - Compute `left_max` (prefix max within block) and `right_max` (suffix max within block).
  - Window maximum from $i$ to $i+k-1$ is simply $\max(\text{right\_max}[i], \text{left\_max}[i + k - 1])$.
  - This avoids dynamic deques entirely and is cache-friendly!