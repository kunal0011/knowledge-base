---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 313: Super Ugly Number"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - math
  - heap
  - google
  - amazon
  - microsoft
---

# LeetCode 313: Super Ugly Number

**Target Companies:** Google, Amazon, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Math / Heap  

---

### Problem Statement

A **super ugly number** is a positive integer whose prime factors are in the array `primes`.

Given an integer `n` and an array of integers `primes`, return *the $n^{\text{th}}$ **super ugly number***.

The $n^{\text{th}}$ super ugly number is **guaranteed** to fit in a **32-bit** signed integer.

---

### Input & Output Formats & Constraints

- **Input:**
  - An integer `n` ($1 \le n \le 10^5$).
  - An integer array `primes` ($1 \le |primes| \le 100$).
- **Output:** An integer representing the $n^{\text{th}}$ super ugly number.
- **Constraints:**
  - `1 <= n <= 10^5`
  - `1 <= primes.length <= 100`
  - `2 <= primes[i] <= 1000`
  - `primes[i]` is guaranteed to be a **prime number**.
  - All the values of `primes` are **unique** and sorted in **ascending order**.

---

### Key Idea & Intuition

#### $K$-Way Merge Perspective
Every super ugly number $U_i$ (for $i > 1$) is formed by multiplying a previously generated super ugly number by one of the primes in `primes`:
$$U_i = U_m \times primes[j] \quad \text{for some } m < i, \, 0 \le j < k$$

This generalizes **Ugly Number II (LeetCode 264)** from 3 primes ($2, 3, 5$) to $k$ arbitrary primes. Generating the sequence in strictly increasing order is equivalent to merging $k$ sorted streams:
$$\text{Stream } j: \; primes[j] \times dp[0], \, primes[j] \times dp[1], \, primes[j] \times dp[2], \dots$$

#### Pointer Array Invariant
For each prime $primes[j]$, we maintain a pointer $idx[j]$ indicating the index in $dp$ of the next number to be multiplied by $primes[j]$:
1. Candidate for prime $j$: $candidate[j] = primes[j] \times dp[idx[j]]$.
2. The next super ugly number is the minimum candidate across all $k$ streams:
   $$dp[i] = \min_{0 \le j < k} candidate[j]$$
3. **Deduplication Invariant:** To prevent duplicate numbers (for instance, when $2 \times 7 = 14$ and $7 \times 2 = 14$), we must increment $idx[j]$ for **every** prime $j$ that generated $dp[i]$:
   $$\text{if } candidate[j] == dp[i] \implies idx[j] \leftarrow idx[j] + 1$$

---

### Solution Approach (Step-by-Step)

1. **Initialize DP Table and Pointers:**
   - Let $k = |primes|$.
   - Array $dp$ of size $n$, setting $dp[0] = 1$.
   - Array $idx$ of size $k$ initialized to 0.
   - Array $next\_val$ of size $k$ where $next\_val[j] = primes[j] \times dp[0] = primes[j]$.
2. **Iterative Multi-Stream Selection:**
   - For $i$ from 1 to $n - 1$:
     - Find the minimum value in $next\_val$: $min\_val = \min(next\_val)$.
     - $dp[i] = min\_val$.
     - For each $j$ from 0 to $k - 1$:
       - If $next\_val[j] == min\_val$:
         - $idx[j] += 1$.
         - $next\_val[j] = primes[j] \times dp[idx[j]]$.
3. **Return Output:**
   - Return $dp[n - 1]$.

---

### Visual Algorithm Walkthrough

#### Trace for `n = 6`, `primes = [2, 7, 13, 19]`
```
Initial State:
dp = [1, 0, 0, 0, 0, 0]
idx = [0, 0, 0, 0]
next_val = [2*1=2, 7*1=7, 13*1=13, 19*1=19]

Step 1 (i = 1):
- min(next_val) = 2 (j = 0)
- dp[1] = 2
- idx[0] becomes 1 -> next_val[0] = 2 * dp[1] = 2 * 2 = 4
next_val = [4, 7, 13, 19]

Step 2 (i = 2):
- min(next_val) = 4 (j = 0)
- dp[2] = 4
- idx[0] becomes 2 -> next_val[0] = 2 * dp[2] = 2 * 4 = 8
next_val = [8, 7, 13, 19]

Step 3 (i = 3):
- min(next_val) = 7 (j = 1)
- dp[3] = 7
- idx[1] becomes 1 -> next_val[1] = 7 * dp[1] = 7 * 2 = 14
next_val = [8, 14, 13, 19]

Step 4 (i = 4):
- min(next_val) = 8 (j = 0)
- dp[4] = 8
- idx[0] becomes 3 -> next_val[0] = 2 * dp[3] = 2 * 7 = 14
next_val = [14, 14, 13, 19]

Step 5 (i = 5):
- min(next_val) = 13 (j = 2)
- dp[5] = 13
- idx[2] becomes 1 -> next_val[2] = 13 * dp[1] = 13 * 2 = 26

Output dp[5] = 13.
Sequence: [1, 2, 4, 7, 8, 13].
```

---

### Solved Examples with Multiple Inputs

| $n$ | `primes` | First Few Generated Numbers | $n^{\text{th}}$ Number |
|---|---|---|---|
| `12` | `[2, 7, 13, 19]` | `1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32` | `32` |
| `1` | `[2, 3, 5]` | `1` (Base case) | `1` |
| `4` | `[2]` | Powers of 2: `1, 2, 4, 8` | `8` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: list[int]) -> int:
        k: int = len(primes)
        dp: list[int] = [0] * n
        dp[0] = 1
        
        idx: list[int] = [0] * k
        next_val: list[int] = list(primes)
        
        for i in range(1, n):
            min_val: int = min(next_val)
            dp[i] = min_val
            
            # Advance all pointers that produced the minimum to avoid duplicates
            for j in range(k):
                if next_val[j] == min_val:
                    idx[j] += 1
                    next_val[j] = primes[j] * dp[idx[j]]
                    
        return dp[n - 1]
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int nthSuperUglyNumber(int n, const std::vector<int>& primes) {
        int k = static_cast<int>(primes.size());
        std::vector<long long> dp(n);
        dp[0] = 1;

        std::vector<int> idx(k, 0);
        std::vector<long long> next_val(k);
        for (int j = 0; j < k; ++j) {
            next_val[j] = primes[j];
        }

        for (int i = 1; i < n; ++i) {
            long long min_val = next_val[0];
            for (int j = 1; j < k; ++j) {
                if (next_val[j] < min_val) {
                    min_val = next_val[j];
                }
            }

            dp[i] = min_val;

            for (int j = 0; j < k; ++j) {
                if (next_val[j] == min_val) {
                    idx[j]++;
                    next_val[j] = static_cast<long long>(primes[j]) * dp[idx[j]];
                }
            }
        }

        return static_cast<int>(dp[n - 1]);
    }
};
```

#### Java 17
```java
class Solution {
    public int nthSuperUglyNumber(int n, int[] primes) {
        int k = primes.length;
        long[] dp = new long[n];
        dp[0] = 1;

        int[] idx = new int[k];
        long[] nextVal = new long[k];
        for (int j = 0; j < k; j++) {
            nextVal[j] = primes[j];
        }

        for (int i = 1; i < n; i++) {
            long minVal = nextVal[0];
            for (int j = 1; j < k; j++) {
                if (nextVal[j] < minVal) {
                    minVal = nextVal[j];
                }
            }

            dp[i] = minVal;

            for (int j = 0; j < k; j++) {
                if (nextVal[j] == minVal) {
                    idx[j]++;
                    nextVal[j] = (long) primes[j] * dp[idx[j]];
                }
            }
        }

        return (int) dp[n - 1];
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \cdot k)$, where $n$ is the requested sequence index and $k = |primes|$. For each of the $n$ numbers, we check the $k$ prime streams in $\mathcal{O}(k)$ time. With $n = 10^5$ and $k \le 100$, operations are bounded by $10^7$, completing in $< 120$ ms. (Using a Priority Queue yields $\mathcal{O}(n \log k)$, though linear array scan is often faster in practice due to lower constant factors and cache locality when $k \le 100$).
- **Space Complexity:** $\mathcal{O}(n + k)$ auxiliary space to store the $dp$ table of size $n$, plus $idx$ and $next\_val$ arrays of size $k$.

---

### Takeaway Pattern & Interview Traps

1. **64-bit Multiplication Safety:** Even though the final $n^{\text{th}}$ super ugly number fits in a 32-bit signed integer, intermediate values in $next\_val$ can temporarily exceed `INT_MAX` before the search terminates. Using `long long` in C++ and `long` in Java prevents integer overflow.
2. **Deduplication via Independent `if`s:** Never use `else if` when advancing the pointers! If multiple primes yield the same minimum value, all corresponding pointers must be advanced together.
3. **Contrast with Heap-Only Approach:** While a min-heap storing unique values works, it requires a hash set for deduplication which incurs large memory overhead. The multi-pointer DP approach naturally avoids heap and set overhead.