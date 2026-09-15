---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - queue
  - monotonic-queue
  - prefix-sum
  - google
  - amazon
---

# LeetCode 862: Shortest Subarray with Sum at Least K

**Target Companies:** Google (Signature Hard Classic), Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Prefix Sums with Monotonic Increasing Deque  

---

### Problem Statement

Given an integer array `nums` (which may contain **negative numbers**) and an integer $k$, return the length of the **shortest non-empty subarray** whose sum is at least $k$. If there is no such subarray, return `-1`.

A **subarray** is a contiguous part of an array.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`, `k: int`
- **Output:** `int` (minimum length or `-1`)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^5 \le \text{nums}[i] \le 10^5$
  - $1 \le k \le 10^9$
  - Note: Prefix sums can reach $\pm 10^{10}$, requiring 64-bit integer precision in C++ and Java (`long long` / `long`).

---

### Key Idea & Intuition

When all numbers are positive (e.g., LeetCode 209), standard two-pointer sliding window solves this problem in $O(N)$ because prefix sums are strictly increasing. However, the presence of **negative integers** destroys monotonicity: expanding a window can decrease the sum, and shrinking can increase it.

Let $P$ be the prefix sum array of size $n + 1$, where $P[i] = \sum_{m=0}^{i-1} nums[m]$ with $P[0] = 0$.
The sum of subarray `nums[i ... j-1]` is given by:
$$\text{Sum}(i, j) = P[j] - P[i]$$
We seek to minimize $(j - i)$ subject to $P[j] - P[i] \ge k \iff P[i] \le P[j] - k$.

We maintain a **Monotonic Increasing Deque** of prefix sum indices:
1. **Front Popping (Optimal Subarray Extraction):**
   While $P[j] - P[q[0]] \ge k$:
   - We found a valid subarray ending at $j - 1$ starting at $q[0]$ with length $j - q[0]$.
   - Update `min_len = min(min_len, j - q[0])`.
   - Pop `q[0]` from the front! Why? Any future index $j' > j$ that could form a valid subarray with $q[0]$ will have length $j' - q[0] > j - q[0]$, which is strictly longer and can never be optimal.
2. **Back Popping (Monotonicity Maintenance):**
   While $q$ is non-empty and $P[j] \le P[q[-1]]$:
   - Pop $q[-1]$ from the back! Why? For any future index $j' > j$, $P[j]$ provides both a **smaller or equal prefix sum** (making $P[j'] - P[j] \ge P[j'] - P[q[-1]]$ easier to satisfy $\ge k$) and a **larger index** $j > q[-1]$ (making $j' - j < j' - q[-1]$ strictly shorter). Hence, $q[-1]$ is completely dominated and useless.

---

### Solution Approach (Step-by-Step)

1. Compute prefix sums `P` of length $n + 1$ using 64-bit integers.
2. Initialize `q = deque()`, `min_len = infinity`.
3. Loop index $j$ from $0$ to $n$:
   - **Check Valid Candidates at Front:** While `q` not empty and $P[j] - P[q[0]] \ge k$:
     - `min_len = min(min_len, j - q.popleft())`.
   - **Prune Dominated Candidates at Back:** While `q` not empty and $P[j] \le P[q[-1]]$:
     - `q.pop()`.
   - Append current index $j$ to back of `q`.
4. Return `min_len` if found, else `-1`.

---

### Visual Algorithm Walkthrough

```
nums = [2, -1, 2], k = 3
Prefix array:
Index:   0   1   2   3
P:      [0,  2,  1,  3]

j = 0, P[0] = 0:
  q is empty -> push 0.
  q = [0 (P=0)]

j = 1, P[1] = 2:
  P[1] - P[q[0]] = 2 - 0 = 2 < 3 (not valid)
  P[1] (2) > P[q[-1]] (0) -> push 1.
  q = [0 (P=0), 1 (P=2)]

j = 2, P[2] = 1:
  P[2] - P[q[0]] = 1 - 0 = 1 < 3 (not valid)
  Maintain monotonicity:
    P[2] (1) <= P[q[-1]] (2) -> POP 1! (1 has larger prefix and earlier index)
    P[2] (1) > P[q[-1]] (0) -> stop popping.
  Push 2.
  q = [0 (P=0), 2 (P=1)]

j = 3, P[3] = 3:
  P[3] - P[q[0]] = 3 - 0 = 3 >= k (3 >= 3)! MATCH!
    min_len = min(inf, 3 - 0) = 3
    popleft 0 -> q = [2 (P=1)]
  Check next front:
    P[3] - P[q[0]] = 3 - 1 = 2 < 3 -> stop.
  P[3] (3) > P[q[-1]] (1) -> push 3.
  q = [2 (P=1), 3 (P=3)]

End of loop. Result = 3.
Subarray is nums[0..2] = [2, -1, 2], sum = 3 >= 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Negative Compensation
- **Input:** `nums = [2, -1, 2]`, `k = 3`
- **Output:** `3`

#### Example 2: Negative Prefix Dropping
- **Input:** `nums = [84, -37, 32, 40, 95]`, `k = 167`
- **Trace:**
  - Prefix sums: `[0, 84, 47, 79, 119, 214]`
  - At $j = 5$ ($P[5] = 214$), $214 - 47 = 167 \ge 167$, giving $j - i = 5 - 2 = 3$ (subarray `[32, 40, 95]`).
- **Output:** `3`

#### Example 3: Impossible Target
- **Input:** `nums = [1, 2]`, `k = 4`
- **Prefix:** `[0, 1, 3]`, maximum sum is 3 $< 4$.
- **Output:** `-1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import deque

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Prefix sum array with 64-bit precision
        prefix: List[int] = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
            
        q = deque()  # stores indices of prefix sums in increasing order
        min_len = float('inf')
        
        for j in range(n + 1):
            # 1. Check if candidate at front satisfies the condition
            while q and prefix[j] - prefix[q[0]] >= k:
                min_len = min(min_len, j - q.popleft())
                
            # 2. Maintain increasing monotonic order from back
            while q and prefix[j] <= prefix[q[-1]]:
                q.pop()
                
            q.append(j)
            
        return int(min_len) if min_len != float('inf') else -1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <deque>
#include <algorithm>

class Solution {
public:
    int shortestSubarray(std::vector<int>& nums, int k) {
        int n = nums.size();
        std::vector<long long> prefix(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            prefix[i + 1] = prefix[i] + nums[i];
        }

        std::deque<int> dq; // stores indices
        int minLen = n + 1;

        for (int j = 0; j <= n; ++j) {
            // Front check: valid subarray found
            while (!dq.empty() && prefix[j] - prefix[dq.front()] >= k) {
                minLen = std::min(minLen, j - dq.front());
                dq.pop_front();
            }

            // Back check: maintain strictly increasing prefix sums
            while (!dq.empty() && prefix[j] <= prefix[dq.back()]) {
                dq.pop_back();
            }

            dq.push_back(j);
        }

        return minLen <= n ? minLen : -1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length;
        long[] prefix = new long[n + 1];
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + nums[i];
        }

        Deque<Integer> dq = new ArrayDeque<>();
        int minLen = n + 1;

        for (int j = 0; j <= n; j++) {
            // Check front of deque for valid candidates
            while (!dq.isEmpty() && prefix[j] - prefix[dq.peekFirst()] >= k) {
                minLen = Math.min(minLen, j - dq.pollFirst());
            }

            // Maintain increasing order at back
            while (!dq.isEmpty() && prefix[j] <= prefix[dq.peekLast()]) {
                dq.pollLast();
            }

            dq.offerLast(j);
        }

        return minLen <= n ? minLen : -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Prefix sums are computed in $O(N)$. Each index from $0$ to $n$ is pushed into the deque once and popped from the front or back at most once. Hence, strictly linear time.
- **Space Complexity:** $O(N)$ auxiliary space for prefix sums and the deque.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Monotonic Increasing Deque on Prefix Sums. Whenever negative numbers invalidate standard sliding window monotonicity, transform to prefix sums and prune suboptimal earlier prefix values with a monotonic deque.
- **Trap:** Integer overflow on prefix sums: When $N = 10^5$ and $nums[i] = 10^5$, sum reaches $10^{10} > 2^{31} - 1$. Failing to use `long long` in C++ or `long` in Java causes integer overflow.