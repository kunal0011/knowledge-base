---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - sliding-window
  - monotonic-queue
  - prefix-sum
  - deque
  - amazon
  - google
---

# LeetCode 862: Shortest Subarray with Sum at Least K

**Target Companies:** Google, Amazon, Meta, Microsoft, ByteDance  
**Difficulty:** Hard  
**Topic:** Sliding Window / Monotonic Queue / Prefix Sum / Deque  

---

### Problem Statement

Given an integer array `nums` and an integer `k`, return the length of the shortest non-empty subarray of `nums` with a sum of at least `k`. If there is no such subarray, return `-1`.

A **subarray** is a contiguous part of an array.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 10^5$).
  - `k`: `int` ($1 \le k \le 10^9$).
- **Output:**
  - `int` — the minimum length of a contiguous subarray with sum $\ge k$, or `-1` if none exists.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^5 \le \text{nums}[i] \le 10^5$
  - $1 \le k \le 10^9$
  *(Crucially, `nums[i]` may be negative).*

---

### Key Idea & Intuition

Why does standard sliding window (LeetCode 209: Minimum Size Subarray Sum) fail here?
In LeetCode 209, all numbers are strictly positive, so prefix sums are strictly increasing.
Here, **`nums` contains negative numbers**, so prefix sums can fluctuate up and down arbitrarily. Expanding the window may decrease the sum, and shrinking the window may increase the sum.

#### Reduction via Prefix Sums:
Let $P[i] = \sum_{m=0}^{i-1} \text{nums}[m]$ for $0 \le i \le n$, with $P[0] = 0$.
The sum of subarray $\text{nums}[j \dots i - 1]$ is:
$$\text{Sum}(j \dots i - 1) = P[i] - P[j]$$
We want to find a pair of indices $(j, i)$ with $j < i$ such that:
$$P[i] - P[j] \ge k \iff P[j] \le P[i] - k$$
while **minimizing the subarray length $i - j$**.

#### Monotonic Queue Invariants:
We maintain a double-ended queue `deq` storing prefix sum indices $j$ such that the prefix sums $P[\text{deq}[0]], P[\text{deq}[1]], \dots$ are **monotonically strictly increasing**:

1. **Eviction from Front (Optimality Pruning):**
   - While `P[i] - P[deq.front()] >= k`:
     - We found a valid subarray ending at index $i - 1$ of length $i - \text{deq.front()}$.
     - Can this candidate starting index `deq.front()` ever produce a *shorter* valid subarray for any future ending index $i' > i$?
     - **No!** Because for any $i' > i$, the length $i' - \text{deq.front()} > i - \text{deq.front()}$.
     - Therefore, `deq.front()` will never be the optimal starting index again. We can permanently pop it from the front: `deq.pop_front()`.

2. **Eviction from Back (Domination Pruning):**
   - While `P[i] <= P[deq.back()]`:
     - Consider the candidate $j = \text{deq.back()}$.
     - Index $i$ has $P[i] \le P[j]$, and $i > j$.
     - For any future right endpoint $i' > i$, if $P[i'] - P[j] \ge k$, then certainly:
       $$P[i'] - P[i] \ge P[i'] - P[j] \ge k$$
       AND the length using $i$ is $i' - i < i' - j$.
     - Thus, index $i$ **strictly dominates** index $j$ in both prefix sum value and position!
     - Index $j$ can never be part of an optimal solution and can be discarded: `deq.pop_back()`.

Because every index from $0$ to $n$ enters `deq` once and is popped at most once, the amortized time complexity across the entire array is strictly $\mathcal{O}(n)$.

---

### Solution Approach (Step-by-Step)

1. Compute prefix sums `P` of length $n + 1$ with $P[0] = 0$.
2. Initialize `min_len = n + 1` and an empty deque `deq`.
3. Loop $i$ from $0$ to $n$:
   - While `deq` is not empty and $P[i] - P[\text{deq}[0]] \ge k$:
     - `min_len = min(min_len, i - deq.popleft())`
   - While `deq` is not empty and $P[i] \le P[\text{deq}[-1]]$:
     - `deq.pop()`
   - Push $i$ to the back of `deq`.
4. Return `min_len` if `min_len <= n` else `-1`.

---

### Visual Algorithm Walkthrough

For `nums = [2, -1, 2]`, `k = 3`:

```
Prefix sums P:
P[0] = 0
P[1] = 2
P[2] = 1   (2 + -1)
P[3] = 3   (1 + 2)

i = 0 (P[0] = 0):
  deq: [0 (P=0)]

i = 1 (P[1] = 2):
  P[1] - P[0] = 2 - 0 = 2 < 3
  P[1] = 2 > P[0] = 0 -> keep
  deq: [0 (P=0), 1 (P=2)]

i = 2 (P[2] = 1):
  P[2] - P[deq.front()=0] = 1 - 0 = 1 < 3
  Check back: P[deq.back()=1] = 2 >= P[2] = 1
  -> Index 1 is dominated by index 2! Pop back 1.
  deq: [0 (P=0), 2 (P=1)]

i = 3 (P[3] = 3):
  Check front: P[3] - P[deq.front()=0] = 3 - 0 = 3 >= 3!
  -> Valid subarray [0..2], length = 3 - 0 = 3.
  min_len = 3. Pop front 0.
  
  Check front again: P[3] - P[deq.front()=2] = 3 - 1 = 2 < 3. Stop.
  Push 3 to back.
  deq: [2 (P=1), 3 (P=3)]

End of loop.
Result: min_len = 3. Subarray [2, -1, 2].
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1]`, `k = 1`
- **Output:** `1`

#### Example 2:
- **Input:** `nums = [1, 2]`, `k = 4`
- **Output:** `-1` (Max sum is $3 < 4$)

#### Example 3:
- **Input:** `nums = [84, -37, 32, 40, 95]`, `k = 167`
- **Tracing:**
  - $P = [0, 84, 47, 79, 119, 214]$
  - At $i = 5$ ($P[5] = 214$):
    - $P[5] - P[2] = 214 - 47 = 167 \ge 167$, length = $5 - 2 = 3$ (subarray `[32, 40, 95]`).
- **Output:** `3`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import deque
from typing import List

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # 64-bit prefix sum array
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + nums[i]
            
        min_len = n + 1
        deq: deque[int] = deque() # Stores indices of prefix sums
        
        for i in range(n + 1):
            # 1. Pop from front: valid subarray found and front index cannot yield shorter windows
            while deq and P[i] - P[deq[0]] >= k:
                min_len = min(min_len, i - deq.popleft())
                
            # 2. Pop from back: current prefix sum dominates older larger prefix sums
            while deq and P[i] <= P[deq[-1]]:
                deq.pop()
                
            deq.append(i)
            
        return min_len if min_len <= n else -1
```

#### C++17
```cpp
#include <vector>
#include <deque>
#include <algorithm>

class Solution {
public:
    int shortestSubarray(const std::vector<int>& nums, int k) {
        int n = static_cast<int>(nums.size());
        std::vector<long long> P(n + 1, 0);
        for (int i = 0; i < n; ++i) {
            P[i + 1] = P[i] + nums[i];
        }
        
        int min_len = n + 1;
        std::deque<int> deq; // Monotonic deque storing prefix sum indices
        
        for (int i = 0; i <= n; ++i) {
            // Evict from front: satisfied condition cannot be improved by future i
            while (!deq.empty() && P[i] - P[deq.front()] >= k) {
                min_len = std::min(min_len, i - deq.front());
                deq.pop_front();
            }
            
            // Evict from back: maintain strictly increasing prefix sums
            while (!deq.empty() && P[i] <= P[deq.back()]) {
                deq.pop_back();
            }
            
            deq.push_back(i);
        }
        
        return min_len <= n ? min_len : -1;
    }
};
```

#### Java 17
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public int shortestSubarray(int[] nums, int k) {
        int n = nums.length;
        long[] P = new long[n + 1];
        for (int i = 0; i < n; i++) {
            P[i + 1] = P[i] + nums[i];
        }
        
        int minLen = n + 1;
        Deque<Integer> deq = new ArrayDeque<>();
        
        for (int i = 0; i <= n; i++) {
            // Check front
            while (!deq.isEmpty() && P[i] - P[deq.peekFirst()] >= k) {
                minLen = Math.min(minLen, i - deq.pollFirst());
            }
            
            // Maintain monotonicity at back
            while (!deq.isEmpty() && P[i] <= P[deq.peekLast()]) {
                deq.pollLast();
            }
            
            deq.offerLast(i);
        }
        
        return minLen <= n ? minLen : -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Prefix sum construction takes $\mathcal{O}(n)$.
  - Each index $0, 1, \dots, n$ is pushed into the deque exactly once.
  - Each index is popped from the front at most once and from the back at most once.
  - Hence, the total number of deque operations across the whole iteration is bounded by $2(n + 1)$, running in strict $\mathcal{O}(n)$ time.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The prefix sum array `P` and monotonic deque `deq` require $\mathcal{O}(n)$ space.

---

### Takeaway Pattern & Interview Traps

- **LeetCode 209 vs. LeetCode 862:**
  - If $\text{nums}[i] \ge 0$: Use standard Two-Pointer Sliding Window in $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space.
  - If $\text{nums}[i]$ can be negative: Sliding window breaks; you **must** use Prefix Sums with a Monotonic Deque in $\mathcal{O}(n)$ time and $\mathcal{O}(n)$ space.
- **Integer Overflow Trap:** With $n = 10^5$ and $\text{nums}[i] = 10^5$, prefix sums can reach $10^{10}$, exceeding signed 32-bit integer capacity. Always use 64-bit integers (`long long` in C++, `long` in Java).