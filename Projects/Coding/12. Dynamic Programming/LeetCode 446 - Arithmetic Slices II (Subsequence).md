---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 446: Arithmetic Slices II (Subsequence)"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - hash-map
  - subsequence
  - google
  - amazon
---

# LeetCode 446: Arithmetic Slices II (Subsequence)

**Target Companies:** Google, Amazon, Meta, Apple, ByteDance  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Hash Map / Subsequence Combination / 64-bit Difference  

---

### Problem Statement

Given an integer array `nums`, return the **number of all the arithmetic subsequences** of `nums`.

A sequence of numbers is called arithmetic if it consists of **at least three elements** and if the difference between any two consecutive elements is the same.

- For example, `[1, 3, 5, 7, 9]`, `[7, 7, 7, 7]`, and `[3, -1, -5, -9]` are arithmetic sequences.
- A **subsequence** of an array is a sequence that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

The test cases are generated so that the answer fits in a **32-bit** integer.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of integers.
- **Output:** `int` — Total count of arithmetic subsequences of length $\ge 3$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$
  - The answer fits in a 32-bit signed integer.

---

### Key Idea & Intuition

1. **Subarrays vs. Subsequences:**
   - In LeetCode 413, slices are contiguous subarrays, solvable via simple linear difference checks.
   - Here, subsequences can skip arbitrary elements, so any pair $(j, i)$ with $j < i$ could potentially extend an arithmetic sequence with difference $d = \text{nums}[i] - \text{nums}[j]$.

2. **State Definition (Length $\ge 2$ Foundation):**
   - Let $\text{dp}[i][d]$ be the number of arithmetic subsequences ending at index $i$ with common difference $d$ that have **length $\ge 2$**.
   - Why include length 2?
     - Any two elements $(nums[j], nums[i])$ form a trivial arithmetic progression of length 2.
     - While length 2 sequences are not counted in the final answer (which requires length $\ge 3$), they serve as the indispensable "launchpad" to form valid length 3 sequences when extended by a future element.

3. **Transition & Answer Accumulation:**
   - For every pair $(j, i)$ with $0 \le j < i < n$:
     - Difference $d = \text{nums}[i] - \text{nums}[j]$.
     - Look up how many sequences ending at $j$ had difference $d$: $\text{prev} = \text{dp}[j][d]$.
     - **Extending to Length $\ge 3$:** Each of the $\text{prev}$ sequences ending at $j$ now gains $\text{nums}[i]$, forming $\text{prev}$ valid sequences of length $\ge 3$. We add $\text{prev}$ directly to our global answer:
       $$\text{total} += \text{prev}$$
     - **Updating DP State:** At index $i$, we now have:
       - The $\text{prev}$ sequences extended from $j$.
       - The $+1$ new base pair $(nums[j], nums[i])$ of length 2.
       - Therefore:
         $$\text{dp}[i][d] += \text{prev} + 1$$

4. **32-bit Integer Overflow on Difference:**
   - Elements range from $-2^{31}$ to $2^{31} - 1$.
   - The difference $\text{nums}[i] - \text{nums}[j]$ can reach $(2^{31} - 1) - (-2^{31}) \approx 2^{32} - 1$, which overflows a standard signed 32-bit integer. Differences in C++ and Java must be cast to 64-bit `long long` / `long`.

---

### Solution Approach (Step-by-Step)

1. **Initialize DP:**
   - Create an array `dp` of $n$ hash maps, where `dp[i]` maps common difference $d \to$ count of length $\ge 2$ sequences ending at $i$.
   - Initialize `total = 0`.
2. **Double Loop Over Pairs:**
   - For $i$ from $0$ to $n - 1$:
     - For $j$ from $0$ to $i - 1$:
       - `diff = nums[i] - nums[j]` (using 64-bit integers).
       - `prev_count = dp[j].get(diff, 0)`.
       - `total += prev_count` (all extended sequences reach length $\ge 3$).
       - `dp[i][diff] = dp[i].get(diff, 0) + prev_count + 1`.
3. **Return:**
   - Return `total`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 4, 6, 8]`:

```
i = 0 (2):
  dp[0] = {}

i = 1 (4):
  j = 0 (2): diff = 2. prev = dp[0][2] = 0.
  total += 0
  dp[1][2] += 0 + 1 = 1  (Pair: [2, 4])

i = 2 (6):
  j = 0 (2): diff = 4. prev = dp[0][4] = 0.
    total += 0, dp[2][4] = 1  (Pair: [2, 6])
  j = 1 (4): diff = 2. prev = dp[1][2] = 1.
    total += 1!  (Sequence extended: [2, 4, 6] of length 3)
    dp[2][2] += 1 + 1 = 2  ([2, 4, 6] and [4, 6])

i = 3 (8):
  j = 0 (2): diff = 6. prev = 0. dp[3][6] = 1.
  j = 1 (4): diff = 4. prev = dp[1][4] = 0. dp[3][4] = 1.
  j = 2 (6): diff = 2. prev = dp[2][2] = 2.
    total += 2!  (Extended: [2, 4, 6, 8] and [4, 6, 8])
    dp[3][2] += 2 + 1 = 3

Final Total = 1 + 2 = 3.
Valid sequences: [2, 4, 6], [4, 6, 8], [2, 4, 6, 8].
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | Stepwise Valid Extensions | Result | Explanation |
|---|---|---|---|---|
| **Simple AP** | `[2, 4, 6, 8]` | $i=2: [2,4,6]$, $i=3: [4,6,8], [2,4,6,8]$ | `3` | Exactly 3 subsequences |
| **Identical Elements** | `[7, 7, 7, 7, 7]` | Diff $0$: length 3 $\to \binom{5}{3}=10$, length 4 $\to \binom{5}{4}=5$, length 5 $\to \binom{5}{5}=1$ | `16` | Total $2^5 - 1 - 5 - 10 = 16$ |
| **Short Array** | `[1, 2]` | Length $< 3$, no length 3 possible | `0` | Base condition |
| **No AP of Length $\ge 3$** | `[1, 2, 4, 7]` | Differences are all distinct | `0` | No 3 elements share common difference |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Dict
from collections import defaultdict

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
            
        # dp[i][d] stores count of arithmetic subsequences ending at i with diff d (length >= 2)
        dp: List[Dict[int, int]] = [defaultdict(int) for _ in range(n)]
        total = 0
        
        for i in range(n):
            for j in range(i):
                diff = nums[i] - nums[j]
                prev_count = dp[j][diff]
                
                # Each sequence ending at j can be extended to length >= 3
                total += prev_count
                
                # Update length >= 2 sequences ending at i
                dp[i][diff] += prev_count + 1
                
        return total
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>

class Solution {
public:
    int numberOfArithmeticSlices(std::vector<int>& nums) {
        int n = nums.size();
        if (n < 3) return 0;

        // dp[i] maps difference -> count of sequences of length >= 2
        // Use long long for difference to prevent 32-bit overflow
        std::vector<std::unordered_map<long long, int>> dp(n);
        int total = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                long long diff = static_cast<long long>(nums[i]) - nums[j];
                
                int prev_count = 0;
                auto it = dp[j].find(diff);
                if (it != dp[j].end()) {
                    prev_count = it->second;
                }

                total += prev_count;
                dp[i][diff] += prev_count + 1;
            }
        }

        return total;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int numberOfArithmeticSlices(int[] nums) {
        int n = nums.length;
        if (n < 3) return 0;

        // dp[i] maps difference (Long) -> count of sequences of length >= 2
        Map<Long, Integer>[] dp = new HashMap[n];
        for (int i = 0; i < n; i++) {
            dp[i] = new HashMap<>();
        }

        int total = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                long diff = (long) nums[i] - nums[j];

                int prevCount = dp[j].getOrDefault(diff, 0);

                total += prevCount;
                dp[i].put(diff, dp[i].getOrDefault(diff, 0) + prevCount + 1);
            }
        }

        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^2)$  
  Iterating over all pairs $(j, i)$ takes $\frac{N(N - 1)}{2}$ steps. Hash map lookups and insertions operate in $\mathcal{O}(1)$ average time. For $N \le 1000$, operations $\approx 5 \times 10^5$, executing in $\approx 100$ ms.
- **Space Complexity:** $\mathcal{O}(N^2)$  
  In the worst case, each pair yields a distinct difference, populating up to $\mathcal{O}(N^2)$ hash map entries.

---

### Takeaway Pattern & Interview Traps

1. **The Difference Type Overflow Trap:**
   - If subtracting `nums[i] - nums[j]` in 32-bit signed integers, values like `INT_MAX - INT_MIN` wrap around into undefined/negative behavior. Always cast to `long long` / `long` before subtraction.
2. **Why Accumulate Before Adding 1:**
   - The expression `total += prev_count` counts *only* extensions of existing subsequences of length $\ge 2$ into length $\ge 3$. The subsequent `+ 1` enters the pair $(j, i)$ of length 2 into `dp[i][diff]` for *future* extensions, avoiding false counts of 2-element pairs.