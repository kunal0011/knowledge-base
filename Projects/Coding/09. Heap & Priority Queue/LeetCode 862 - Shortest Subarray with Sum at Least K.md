---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
  - monotonic-queue
  - prefix-sum
  - amazon
  - google
---

# LeetCode 862: Shortest Subarray with Sum at Least K

**Target Companies:** Google, Amazon, Meta, Microsoft, Citadel, Bloomberg  
**Difficulty:** Hard  
**Topic:** Monotonic Deque (Linear Time) / Priority Queue (Min-Heap) / Prefix Sums

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return *the length of the shortest non-empty subarray of `nums` with a sum of at least `k`*. If there is no such subarray, return `-1`.

A **subarray** is a contiguous part of an array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]`, where $1 \le \text{nums.length} \le 10^5$. $-10^5 \le nums[i] \le 10^5$.
  - `k`: `int`, where $1 \le k \le 10^9$.
- **Output:**
  - `int`: Minimum length of a subarray with sum $\ge k$, or `-1` if none exists.
- **Constraints:**
  - `nums` contains **negative numbers**. Standard two-pointer / expanding sliding windows fail because subarray sums are non-monotonic.
  - Prefix sums can reach $10^5 \times 10^5 = 10^{10}$, requiring **64-bit integers** (`long long` in C++, `long` in Java).

---

### Key Idea & Intuition

Let $P[i]$ be the prefix sum up to index $i$:
$$P[i] = \sum_{m=0}^{i-1} nums[m], \quad P[0] = 0$$
The sum of subarray $nums[j \dots i-1]$ is given by:
$$\text{Sum}(j \dots i-1) = P[i] - P[j]$$
We seek a pair of indices $(j, i)$ with $j < i$ such that:
$$P[i] - P[j] \ge k \iff P[j] \le P[i] - k$$
and we want to minimize the length $i - j$.

#### Why Standard Sliding Window Fails:
Because $nums[i]$ can be negative, $P[i]$ is not monotonic. Adding an element may decrease the sum, and shrinking the window may increase the sum.

#### Paradigm 1: Optimal Monotonic Increasing Deque ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space)
Maintain a deque of indices $j$ such that their prefix sums $P[j]$ are strictly increasing:
1. **Shrink from Front (Optimal Length Found):**
   - If $P[i] - P[dq[0]] \ge k$, the subarray from $dq[0]$ to $i$ satisfies the condition with length $i - dq[0]$.
   - Can index $dq[0]$ ever yield a shorter valid subarray with future right endpoints $i' > i$?
     - **No!** Any future $i'$ will have distance $i' - dq[0] > i - dq[0]$.
     - Thus, we can safely pop $dq[0]$ permanently via `dq.popleft()`!
2. **Maintain Monotonicity from Back:**
   - Before adding $i$, if $P[i] \le P[dq[-1]]$:
     - For any future right endpoint $i'$, $P[i]$ is smaller than $P[dq[-1]]$ AND index $i$ is closer to $i'$ than $dq[-1]$.
     - Thus, index $i$ dominates $dq[-1]$ in both value and length!
     - Pop $dq[-1]$ from the back: `dq.pop()`.
3. Append $i$ to the deque.

#### Paradigm 2: Min-Heap ($\mathcal{O}(N \log N)$ Time, $\mathcal{O}(N)$ Space)
- Maintain a min-heap of pairs `(P[j], j)`.
- For each $P[i]$, while `P[i] - heap[0].sum >= k`, update $ans = \min(ans, i - heap[0].idx)$ and pop `heap[0]`.
- Push `(P[i], i)`.

---

### Solution Approach (Step-by-Step)

#### Monotonic Deque Algorithm:
1. Compute prefix sums array $P$ of length $n + 1$ with $P[0] = 0$. Use 64-bit integers.
2. Initialize `dq = deque()`, `min_len = infinity`.
3. For $i$ from $0$ to $n$:
   - While `dq` is non-empty and $P[i] - P[dq[0]] \ge k$:
     - `min_len = min(min_len, i - dq.popleft())`
   - While `dq` is non-empty and $P[i] \le P[dq[-1]]$:
     - `dq.pop()`
   - `dq.append(i)`
4. Return `min_len` if `min_len != infinity` else `-1`.

---

### Visual Algorithm Walkthrough

Let `nums = [2, -1, 2]`, $k = 3$:

```
Prefix Sums P:
Index i:   0   1   2   3
Value:     -   2  -1   2
P[i]:      0   2   1   3

i = 0 (P[0] = 0):
  dq = [0]

i = 1 (P[1] = 2):
  P[1] - P[dq[0]] = 2 - 0 = 2 < 3.
  P[1] = 2 > P[0] = 0 -> Maintain increasing order.
  dq = [0, 1]

i = 2 (P[2] = 1):
  P[2] - P[dq[0]] = 1 - 0 = 1 < 3.
  P[2] = 1 <= P[dq[-1]] (1 <= 2) -> Pop 1 from back!
  dq = [0, 2]

i = 3 (P[3] = 3):
  P[3] - P[dq[0]] = 3 - 0 = 3 >= 3!
  Valid! Length = 3 - 0 = 3. min_len = 3.
  Pop 0! dq = [2]
  P[3] - P[dq[0]] = 3 - 1 = 2 < 3.
  Append 3 -> dq = [2, 3]

Output: min_len = 3 (Subarray [2, -1, 2])
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Negative Middle Value

- **Input:** `nums = [2, -1, 2]`, `k = 3`
- **Output:** `3`

#### Example 2: Negative Prefix Excluded

- **Input:** `nums = [1, 2, -1, 2]`, `k = 3`
- **Tracing:**
  - $P = [0, 1, 3, 2, 4]$.
  - At $i = 4$, $P[4] - P[2] = 4 - 1 = 3 \ge 3 \implies$ Length $4 - 1 = 3$ (subarray `[2, -1, 2]`).
- **Output:** `3`

#### Example 3: Impossible Case

- **Input:** `nums = [1, 2]`, `k = 4`
- **Output:** `-1`

---

### Multi-Language Implementations

#### Python 3

```python
from collections import deque
from typing import List

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Prefix sum array (P[i] is sum of first i elements)
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + nums[i]

        dq = deque()  # stores indices of P
        min_len = float('inf')

        for i in range(n + 1):
            # 1. Shrink window from front if condition is satisfied
            while dq and P[i] - P[dq[0]] >= k:
                min_len = min(min_len, i - dq.popleft())

            # 2. Maintain monotonic increasing prefix sums from back
            while dq and P[i] <= P[dq[-1]]:
                dq.pop()

            dq.append(i)

        return min_len if min_len != float('inf') else -1
```

#### C++17

```cpp
#include <vector>
#include <deque>
#include <algorithm>
#include <climits>

class Solution {
public:
    int shortestSubarray(const std::vector<int>& nums, int k) {
        int n = static_cast<int>(nums.size());
        // Use 64-bit integers to prevent overflow with large prefix sums
        std::vector<long long> P(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            P[i + 1] = P[i] + nums[i];
        }

        std::deque<int> dq;
        int min_len = INT_MAX;

        for (int i = 0; i <= n; ++i) {
            // Check front of deque for valid subarrays
            while (!dq.empty() && P[i] - P[dq.front()] >= k) {
                min_len = std::min(min_len, i - dq.front());
                dq.pop_front();
            }

            // Maintain monotonic increasing property
            while (!dq.empty() && P[i] <= P[dq.back()]) {
                dq.pop_back();
            }

            dq.push_back(i);
        }

        return min_len == INT_MAX ? -1 : min_len;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length;
        // 64-bit prefix sum array
        long[] P = new long[n + 1];
        for (int i = 0; i < n; i++) {
            P[i + 1] = P[i] + nums[i];
        }

        Deque<Integer> dq = new ArrayDeque<>();
        int minLen = Integer.MAX_VALUE;

        for (int i = 0; i <= n; i++) {
            // Shrink from front
            while (!dq.isEmpty() && P[i] - P[dq.peekFirst()] >= k) {
                minLen = Math.min(minLen, i - dq.pollFirst());
            }

            // Maintain monotonic increasing prefix sums
            while (!dq.isEmpty() && P[i] <= P[dq.peekLast()]) {
                dq.pollLast();
            }

            dq.offerLast(i);
        }

        return minLen == Integer.MAX_VALUE ? -1 : minLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Computing the prefix sum array takes $\mathcal{O}(N)$ time.
  - In the monotonic deque pass, each index $0 \dots N$ is added to the deque once and popped at most once from the front or the back.
  - Total Time: $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - To store the prefix sum array $P$ and the indices in the deque.

---

### Takeaway Pattern & Interview Traps

1. **64-bit Integer Overflow:**
   - In C++ and Java, summing $10^5$ integers of value $10^5$ yields $10^{10}$, which exceeds `INT_MAX` ($\approx 2 \times 10^9$). Using `long long` / `long` is mandatory.
2. **Why Monotonic Deque Beats Heap:**
   - Both `heappop` and `dq.popleft` can find valid prefixes. However, the deque approach also drops dominated elements from the back (`P[i] <= P[dq[-1]]`), reducing the total time from $\mathcal{O}(N \log N)$ to optimal $\mathcal{O}(N)$.