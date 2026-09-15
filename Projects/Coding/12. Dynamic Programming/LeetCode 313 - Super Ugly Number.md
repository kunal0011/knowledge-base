---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 313: Super Ugly Number"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 313: Super Ugly Number

**LeetCode 313 – Super Ugly Number**, focusing strictly on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 313 — Super Ugly Number

### Problem Statement

A **super ugly number** is a positive integer whose prime factors are all in the given array `primes`.

You are given:

* `n`: the index (1-based)
* `primes`: an array of prime numbers

Return the **n-th super ugly number**.

**Note**

* `1` is always considered the first super ugly number.
* `1 <= n <= 10^5`
* `1 <= len(primes) <= 100`

---

## Key Observation

Every super ugly number is formed by:

```
(previous super ugly number) × (one of the primes)
```

This is identical in structure to **Ugly Number II (LeetCode 264)** but generalized to **k primes**.

We want the sequence in **sorted order without duplicates**.

---

## DP State Definition

### DP Array

```
dp[i] = i-th super ugly number (1-indexed)
```

Base case:

```
dp[1] = 1
```

---

### Pointer Array (Critical Insight)

For each prime `primes[j]`, maintain a pointer:

```
idx[j] = index in dp[] whose value is to be multiplied with primes[j]
```

Initially:

```
idx[j] = 1   for all j
```

---

## State Transition

For each `i` from `2` to `n`:

1. **Generate candidates**

```
candidate[j] = dp[idx[j]] * primes[j]
```

2. **Pick the minimum**

```
dp[i] = min(candidate[j]) for all j
```

3. **Advance pointers**  
   For every `j` such that:

```
candidate[j] == dp[i]
```

increment:

```
idx[j] += 1
```

This avoids duplicates.

---

## Why Multiple Pointer Increment Is Required?

If multiple primes generate the same value (e.g., `2×3` and `3×2`), we must **advance all responsible pointers** to prevent repeating the same number.

---

## DP Table Construction (Worked Example)

### Example

```
n = 12
primes = [2, 7, 13, 19]
```

---

### Initialization

```
dp[1] = 1
idx = [1, 1, 1, 1]
```

---

### Iterative DP Filling

| i | dp[i] | candidates (2,7,13,19) | idx after |
| --- | --- | --- | --- |
| 1 | 1 | — | [1,1,1,1] |
| 2 | 2 | 2,7,13,19 | [2,1,1,1] |
| 3 | 4 | 4,7,13,19 | [3,1,1,1] |
| 4 | 7 | 8,7,13,19 | [3,2,1,1] |
| 5 | 8 | 8,14,13,19 | [4,2,1,1] |
| 6 | 13 | 14,14,13,19 | [4,2,2,1] |
| 7 | 14 | 14,14,26,19 | [5,3,2,1] |
| 8 | 16 | 16,28,26,19 | [6,3,2,1] |
| 9 | 19 | 26,28,26,19 | [6,3,2,2] |
| 10 | 26 | 26,28,26,38 | [7,3,3,2] |
| 11 | 28 | 28,28,52,38 | [8,4,3,2] |
| 12 | 32 | 32,49,52,38 | [9,4,3,2] |

---

### Final DP Sequence

```
[1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32]
```

**Answer:** `32`

---

## Python 3 (Typed) — DP Implementation

```python
from typing import List

class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        k = len(primes)

        dp: List[int] = [0] * (n + 1)
        dp[1] = 1

        idx: List[int] = [1] * k

        for i in range(2, n + 1):
            next_val = float('inf')

            # generate candidates
            for j in range(k):
                next_val = min(next_val, dp[idx[j]] * primes[j])

            dp[i] = next_val

            # advance all pointers that produced next_val
            for j in range(k):
                if dp[idx[j]] * primes[j] == next_val:
                    idx[j] += 1

        return dp[n]
```

---

## Time & Space Complexity

### Time

```
O(n × k)
```

Where `k = len(primes)`

### Space

```
O(n + k)
```

---

## Mental Model (Important)

* DP builds the sequence in **sorted order**
* Pointers act as **merge cursors** across `k` sorted sequences:

  ```
  dp × prime1
  dp × prime2
  ...
  dp × primeK
  ```
* This is essentially a **k-way merge using DP**

---

If you want, I can also provide:

* Min-heap solution comparison
* Pointer movement visualization as a tree
* Optimized version discussion
* Dry-run with another input

Just tell me.