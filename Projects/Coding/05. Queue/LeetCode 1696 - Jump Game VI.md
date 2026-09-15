---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1696: Jump Game VI"
tags:
  - leetcode
  - coding
  - queue
  - monotonic-queue
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 1696: Jump Game VI

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming with Monotonic Decreasing Deque Window Optimization  

---

### Problem Statement

You are given a **0-indexed** integer array `nums` and an integer $k$.

You are initially standing at index $0$. In one move, you can jump at most $k$ steps forward without going outside the boundaries of the array. That is, you can jump from index $i$ to any index in the range $[i + 1, \min(n - 1, i + k)]$.

Your score is the sum of all `nums[j]` for each index $j$ you visited in the path.

Return the **maximum score** you can get to reach the last index of the array (index $n - 1$).

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (maximum reachable score)
- **Constraints:**
  - $1 \le \text{nums.length}, k \le 10^5$
  - $-10^4 \le \text{nums}[i] \le 10^4$

---

### Key Idea & Intuition

Let `dp[i]` be the maximum score achievable starting at index $0$ and ending at index $i$.
The transition state is:
$$dp[i] = nums[i] + \max_{\max(0, i - k) \le j < i} (dp[j])$$
with base case $dp[0] = nums[0]$.

A naive evaluation of the inner maximum takes $O(k)$ for each index, resulting in an overall time complexity of $O(N \cdot k)$. When $N, k \le 10^5$, $N \cdot k \approx 10^{10}$, causing a Time Limit Exceeded (TLE).

Notice that as $i$ increments, the lookup range $[i - k, i - 1]$ is a **sliding window of size $k$** over the `dp` array!
We can maintain a **Monotonic Decreasing Deque** of indices:
1. `q[0]` stores the index of the maximum `dp` value within the last $k$ indices $[i - k, i - 1]$.
2. **Boundary Eviction:** If `q[0] < i - k`, pop from the front.
3. **Transition:** Compute `dp[i] = nums[i] + dp[q[0]]`.
4. **Monotonicity Maintenance:** While `q` is non-empty and `dp[q[-1]] <= dp[i]`, pop from the back. Then append `i`.
This reduces the query from $O(k)$ to $O(1)$ amortized time.

---

### Solution Approach (Step-by-Step)

1. Create array `dp` of length $n$ with `dp[0] = nums[0]`.
2. Initialize `q = deque([0])`.
3. For $i$ from $1$ to $n - 1$:
   - Evict expired index from front: if `q[0] < i - k`: `q.popleft()`.
   - Calculate `dp[i] = nums[i] + dp[q[0]]`.
   - While `q` and `dp[q[-1]] <= dp[i]`: `q.pop()`.
   - Append `i` to `q`.
4. Return `dp[n - 1]`.

---

### Visual Algorithm Walkthrough

```
nums = [1, -1, -2, 4, -7, 3], k = 2
dp[0] = nums[0] = 1, q = [0] (stores index 0 with dp[0]=1)

i = 1 (nums[1] = -1):
  q[0] = 0 (1 - 2 = -1 < 0, valid!)
  dp[1] = nums[1] + dp[0] = -1 + 1 = 0
  dp[q[-1]] = 1 > 0 -> push 1
  q = [0 (dp=1), 1 (dp=0)]

i = 2 (nums[2] = -2):
  q[0] = 0 (2 - 2 = 0 <= 0, valid!)
  dp[2] = nums[2] + dp[0] = -2 + 1 = -1
  q = [0 (dp=1), 1 (dp=0), 2 (dp=-1)]

i = 3 (nums[3] = 4):
  Check expiry: q[0] = 0 < 3 - 2 = 1 -> EVICT 0!
  q is now [1 (dp=0), 2 (dp=-1)]
  dp[3] = nums[3] + dp[q[0]] = 4 + dp[1] = 4 + 0 = 4
  Pop back: dp[2] (-1) <= 4 -> pop 2; dp[1] (0) <= 4 -> pop 1
  q = [3 (dp=4)]

i = 4 (nums[4] = -7):
  dp[4] = nums[4] + dp[3] = -7 + 4 = -3
  q = [3 (dp=4), 4 (dp=-3)]

i = 5 (nums[5] = 3):
  q[0] = 3 (5 - 2 = 3 <= 3, valid!)
  dp[5] = nums[5] + dp[3] = 3 + 4 = 7
  q = [5 (dp=7)]

Final Result: dp[5] = 7 (Path: 0 -> 1 -> 3 -> 5, sum = 1 + (-1) + 4 + 3 = 7).
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Optimal Path
- **Input:** `nums = [1, -1, -2, 4, -7, 3]`, `k = 2`
- **Output:** `7`

#### Example 2: Negative Penalty Avoidance
- **Input:** `nums = [10, -5, -2, 4, 0, 3]`, `k = 3`
- **Trace:**
  - Can jump directly from index 0 ($10$) to index 3 ($4$), then to 5 ($3$).
  - Score: $10 + 4 + 3 = 17$.
- **Output:** `17`

#### Example 3: All Negative Steps
- **Input:** `nums = [1, -5, -20, 4, -1, 3, -6, -3]`, `k = 2`
- **Output:** `0`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        
        q = deque([0])  # stores indices, dp values are monotonically decreasing
        
        for i in range(1, n):
            # 1. Evict indices older than i - k
            if q[0] < i - k:
                q.popleft()
                
            # 2. Optimal previous state is at the front of deque
            dp[i] = nums[i] + dp[q[0]]
            
            # 3. Maintain monotonic decreasing order in deque
            while q and dp[q[-1]] <= dp[i]:
                q.pop()
            q.append(i)
            
        return dp[-1]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>

class Solution {
public:
    int maxResult(std::vector<int>& nums, int k) {
        int n = nums.size();
        std::vector<int> dp(n, 0);
        dp[0] = nums[0];

        std::deque<int> dq; // stores indices
        dq.push_back(0);

        for (int i = 1; i < n; ++i) {
            // Evict expired index
            if (!dq.empty() && dq.front() < i - k) {
                dq.pop_front();
            }

            // Transition from best window maximum
            dp[i] = nums[i] + dp[dq.front()];

            // Maintain decreasing monotonic order
            while (!dq.empty() && dp[dq.back()] <= dp[i]) {
                dq.pop_back();
            }
            dq.push_back(i);
        }

        return dp[n - 1];
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int maxResult(int[] nums, int k) {
        int n = nums.length;
        int[] dp = new int[n];
        dp[0] = nums[0];

        Deque<Integer> dq = new ArrayDeque<>();
        dq.offerLast(0);

        for (int i = 1; i < n; i++) {
            // Evict expired index
            if (!dq.isEmpty() && dq.peekFirst() < i - k) {
                dq.pollFirst();
            }

            // DP state transition
            dp[i] = nums[i] + dp[dq.peekFirst()];

            // Maintain decreasing monotonic order
            while (!dq.isEmpty() && dp[dq.peekLast()] <= dp[i]) {
                dq.pollLast();
            }
            dq.offerLast(i);
        }

        return dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Each index from $0$ to $n - 1$ is added to the deque once and popped at most once. DP state evaluation is $O(1)$.
- **Space Complexity:** $O(N)$ for the `dp` array (can be reduced to $O(k)$ in-place) and $O(k)$ for the deque.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Monotonic Queue-Optimized 1D Dynamic Programming. Whenever $dp[i] = \text{cost}[i] + \text{opt}_{j \in [i-k, i-1]}(dp[j])$, apply a monotonic deque to convert an $O(N \cdot k)$ transition to $O(N)$.
- **Trap:** Forgetting that `nums[i]` can be negative. Always store index in deque and compare using `dp[dq.back()] <= dp[i]`.