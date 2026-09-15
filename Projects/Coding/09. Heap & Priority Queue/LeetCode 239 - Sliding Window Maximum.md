---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 239: Sliding Window Maximum"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - sliding-window
  - monotonic-queue
  - amazon
  - google
---

# LeetCode 239: Sliding Window Maximum

**Target Companies:** Amazon (Top Classic), Google, Meta, Microsoft, Citadel, Bloomberg  
**Difficulty:** Hard  
**Topic:** Monotonic Deque (Linear Time) / Priority Queue (Max-Heap with Lazy Deletion)

---

### Problem Statement

You are given an array of integers `nums`, there is a sliding window of size `k` which is moving from the very left of the array to the very right. You can only see the `k` numbers in the window. Each time the sliding window moves right by one position.

Return *the max sliding window*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 10^5$.
  - `k`: `int`, where $1 \le k \le \text{nums.length}$.
  - $-10^4 \le nums[i] \le 10^4$.
- **Output:**
  - `List[int]`: An array of length $N - k + 1$ containing the maximum element of every consecutive window.
- **Constraints:**
  - The window slides from index $0$ to $N - k$.

---

### Key Idea & Intuition

A brute force search takes $\mathcal{O}(k)$ per window, giving $\mathcal{O}(N \cdot k)$ total, which times out since $N, k \le 10^5$ ($N \cdot k \approx 10^{10}$).

#### Paradigm 1: Optimal Monotonic Decreasing Deque ($\mathcal{O}(N)$ Time, $\mathcal{O}(k)$ Space)
Notice that if an element at index $j$ is smaller than a newer element at index $i$ ($j < i$ and $nums[j] \le nums[i]$), the older element $nums[j]$ can **never** be the maximum for the current window or any future window! The newer element $nums[i]$ is both larger and will stay in the window longer.

Thus, we maintain a **Monotonic Decreasing Deque** of indices:
1. **Maintain Decreasing Order:** Before adding index $i$, pop from the back of the deque all indices $j$ where $nums[j] \le nums[i]$.
2. **Push Current Index:** Append $i$ to the back of the deque.
3. **Evict Out-of-Window Elements:** If the front index is outside the current window ($dq[0] \le i - k$), pop from the front.
4. **Read Window Maximum:** Once $i \ge k - 1$, the front index `dq[0]` holds the maximum value of the current window! Append `nums[dq[0]]` to the result.

Every index is pushed into the deque once and popped at most once $\implies$ strictly $\mathcal{O}(N)$ time!

#### Paradigm 2: Max-Heap with Lazy Deletion ($\mathcal{O}(N \log N)$ Time, $\mathcal{O}(N)$ Space)
- Store `(-nums[i], i)` in a max-heap.
- At each step $i$, push `(-nums[i], i)`.
- When $i \ge k - 1$, lazily pop the heap top while its index is out of bounds (`heap[0][1] <= i - k`).
- The top then holds the valid window maximum.

---

### Solution Approach (Step-by-Step)

#### Monotonic Deque Algorithm:
1. Initialize `dq = deque()` (storing indices) and `result = []`.
2. For each index $i$ from $0$ to $N - 1$:
   - While `dq` is not empty and `nums[dq[-1]] <= nums[i]`:
     - `dq.pop()`
   - `dq.append(i)`
   - If `dq[0] <= i - k`:
     - `dq.popleft()`
   - If $i \ge k - 1$:
     - `result.append(nums[dq[0]])`
3. Return `result`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 3, -1, -3, 5, 3, 6, 7]`, $k = 3$:

```
Indices:   0   1   2   3   4   5   6   7
Values:    1   3  -1  -3   5   3   6   7

i = 0 (1):
  dq = [0] (value 1)

i = 1 (3):
  3 > 1 -> pop 0
  dq = [1] (value 3)

i = 2 (-1):
  -1 < 3 -> append
  dq = [1, 2] (values 3, -1)
  First window complete! Window [1, 3, -1]. Max is nums[dq[0]] = 3.
  Result: [3]

i = 3 (-3):
  -3 < -1 -> append
  dq = [1, 2, 3] (values 3, -1, -3)
  Front index 1 is within [1..3] -> OK.
  Result: [3, 3]

i = 4 (5):
  5 > -3 -> pop 3
  5 > -1 -> pop 2
  5 > 3  -> pop 1
  dq = [4] (value 5)
  Result: [3, 3, 5]

i = 5 (3):
  3 < 5 -> append
  dq = [4, 5] (values 5, 3)
  Result: [3, 3, 5, 5]

i = 6 (6):
  6 > 3 -> pop 5
  6 > 5 -> pop 4
  dq = [6] (value 6)
  Result: [3, 3, 5, 5, 6]

i = 7 (7):
  7 > 6 -> pop 6
  dq = [7] (value 7)
  Result: [3, 3, 5, 5, 6, 7]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Fluctuating Array

- **Input:** `nums = [1,3,-1,-3,5,3,6,7]`, `k = 3`
- **Output:** `[3,3,5,5,6,7]`

#### Example 2: Single Element Array

- **Input:** `nums = [1]`, `k = 1`
- **Output:** `[1]`

#### Example 3: Monotonically Decreasing Array

- **Input:** `nums = [9, 8, 7, 6, 5]`, `k = 3`
- **Tracing:**
  - Window `[9, 8, 7]` $\implies 9$
  - Window `[8, 7, 6]` $\implies 8$
  - Window `[7, 6, 5]` $\implies 7$
- **Output:** `[9, 8, 7]`

---

### Multi-Language Implementations

#### Python 3

##### Optimal $\mathcal{O}(N)$ Monotonic Deque
```python
from collections import deque
from typing import List

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        dq = deque()  # stores indices of candidate maximums
        result = []

        for i, val in enumerate(nums):
            # 1. Remove elements smaller than incoming val from back
            while dq and nums[dq[-1]] <= val:
                dq.pop()
            dq.append(i)

            # 2. Evict front element if out of current window bounds
            if dq[0] <= i - k:
                dq.popleft()

            # 3. Add to result once first full window is established
            if i >= k - 1:
                result.append(nums[dq[0]])

        return result
```

#### C++17

```cpp
#include <vector>
#include <deque>

class Solution {
public:
    std::vector<int> maxSlidingWindow(const std::vector<int>& nums, int k) {
        std::deque<int> dq; // stores indices
        int n = static_cast<int>(nums.size());
        std::vector<int> result;
        result.reserve(n - k + 1);

        for (int i = 0; i < n; ++i) {
            // Remove smaller elements from back
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }
            dq.push_back(i);

            // Remove out-of-bounds indices from front
            if (dq.front() <= i - k) {
                dq.pop_front();
            }

            // Record window maximum
            if (i >= k - 1) {
                result.push_back(nums[dq.front()]);
            }
        }

        return result;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        int[] result = new int[n - k + 1];
        Deque<Integer> dq = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            // Maintain monotonic decreasing order
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) {
                dq.pollLast();
            }
            dq.offerLast(i);

            // Remove elements outside current window
            if (dq.peekFirst() <= i - k) {
                dq.pollFirst();
            }

            // Record maximum
            if (i >= k - 1) {
                result[i - k + 1] = nums[dq.peekFirst()];
            }
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Each index is pushed into the deque once and popped from the deque at most once.
  - Overall time is strictly linear in the size of `nums`.
- **Space Complexity:** $\mathcal{O}(k)$
  - The deque holds at most $k$ indices at any given moment.

---

### Takeaway Pattern & Interview Traps

1. **Storing Indices vs Storing Values:**
   - Storing **indices** in the deque is crucial because it allows instantaneous $\mathcal{O}(1)$ verification of whether an element is still within window bounds (`dq[0] <= i - k`).
2. **Strict Inequality vs Non-Strict:**
   - Using `<=` when popping smaller elements from the back (`nums[dq[-1]] <= val`) removes older duplicates, keeping the deque strictly minimal and saving memory.