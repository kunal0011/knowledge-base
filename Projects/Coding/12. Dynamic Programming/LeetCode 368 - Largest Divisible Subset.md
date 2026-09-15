---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 368: Largest Divisible Subset"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - sorting
  - math
  - google
  - amazon
  - meta
---

# LeetCode 368: Largest Divisible Subset

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / Longest Increasing Subsequence Variant / Path Reconstruction  

---

### Problem Statement

Given a set of **distinct positive integers** `nums`, return the largest subset `answer` such that every pair of elements `(answer[i], answer[j])` in this subset satisfies:

- `answer[i] % answer[j] == 0`, or
- `answer[j] % answer[i] == 0`

If there are multiple solutions, return any of them.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of distinct positive integers.
- **Output:** `List[int]` — The largest subset satisfying pairwise divisibility.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $1 \le \text{nums}[i] \le 2 \times 10^9$
  - All integers in `nums` are **unique**.

---

### Key Idea & Intuition

1. **Transitivity of Divisibility:**
   - Divisibility is a transitive relation over positive integers:
     $$\text{If } a \mid b \text{ and } b \mid c, \text{ then } a \mid c$$
   - If an array is sorted in ascending order $s_1 < s_2 < \dots < s_k$, to add a new number $x > s_k$ such that all pairwise divisibilities hold, we **only need to check** if $s_k \mid x$ (i.e. $x \pmod{s_k} == 0$).
   - Because $s_k$ is divisible by all previous $s_i$, any number divisible by $s_k$ is automatically divisible by all elements in the chain!

2. **LIS-Style DP Formulation:**
   - Sorting transforms an arbitrary subset problem (which would otherwise resemble NP-hard Maximum Clique) into a variant of **Longest Increasing Subsequence (LIS)**.
   - Let $\text{dp}[i]$ be the size of the largest divisible subset whose largest element is $\text{nums}[i]$ (after sorting).
   - Base case: $\text{dp}[i] = 1$ for all $i$ (each number forms a valid subset of size 1).
   - Recurrence:
     $$\text{dp}[i] = 1 + \max_{\substack{0 \le j < i \\ \text{nums}[i] \% \text{nums}[j] == 0}} \text{dp}[j]$$

3. **Reconstructing the Solution Path:**
   - Maintain an array `parent[i]` initialized to `-1`.
   - When $\text{dp}[j] + 1 > \text{dp}[i]$, update $\text{dp}[i] = \text{dp}[j] + 1$ and record `parent[i] = j`.
   - After computing DP for all elements, find the index `max_idx` with the largest $\text{dp}$ value. Backtrack from `max_idx` using `parent` until `-1` to reconstruct the subset.

---

### Solution Approach (Step-by-Step)

1. **Sort `nums`:**
   - Sort `nums` in ascending order.
2. **Initialize DP and Tracking Arrays:**
   - `dp = [1] * n`, `parent = [-1] * n`.
   - Track `max_len = 1` and `max_idx = 0`.
3. **Double Loop DP:**
   - For $i$ from $0$ to $n - 1$:
     - For $j$ from $0$ to $i - 1$:
       - If `nums[i] % nums[j] == 0` and `dp[j] + 1 > dp[i]`:
         - `dp[i] = dp[j] + 1`
         - `parent[i] = j`
     - If `dp[i] > max_len`:
       - `max_len = dp[i]`
       - `max_idx = i`
4. **Reconstruct Subset:**
   - Backtrack through `parent` starting at `max_idx`. Append each `nums[curr]` to the result, then advance `curr = parent[curr]`.
   - Return the reversed result (or as-is, since any order of the subset is valid).

---

### Visual Algorithm Walkthrough

Suppose `nums = [1, 2, 4, 8]`:

```
Sorted nums: [1, 2, 4, 8]
Indices:      0  1  2  3

i = 0 (val = 1):
  dp[0] = 1, parent[0] = -1

i = 1 (val = 2):
  j = 0: 2 % 1 == 0 -> dp[1] = 1 + dp[0] = 2, parent[1] = 0

i = 2 (val = 4):
  j = 0: 4 % 1 == 0 -> dp[2] = 1 + dp[0] = 2, parent[2] = 0
  j = 1: 4 % 2 == 0 -> dp[2] = 1 + dp[1] = 3, parent[2] = 1

i = 3 (val = 8):
  j = 0: 8 % 1 == 0 -> dp[3] = 2
  j = 1: 8 % 2 == 0 -> dp[3] = 3
  j = 2: 8 % 4 == 0 -> dp[3] = 1 + dp[2] = 4, parent[3] = 2

DP array:     [ 1,  2,  3,  4]
Parent array: [-1,  0,  1,  2]

Max length = 4 at index = 3 (val 8)
Reconstruction: 3 (8) -> 2 (4) -> 1 (2) -> 0 (1) -> -1
Result: [1, 2, 4, 8]
```

---

### Solved Examples with Multiple Inputs

| Case | Input `nums` | Sorted `nums` | Final DP Table | Reconstructed Output | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[1, 2, 3]` | `[1, 2, 3]` | `dp = [1, 2, 2]` | `[1, 2]` (or `[1, 3]`) | Both pairs divide each other |
| **Powers of 2** | `[1, 2, 4, 8]` | `[1, 2, 4, 8]` | `dp = [1, 2, 3, 4]` | `[1, 2, 4, 8]` | Every power divides the next |
| **Primes Only** | `[3, 5, 7, 11]` | `[3, 5, 7, 11]` | `dp = [1, 1, 1, 1]` | `[3]` (any single element) | No prime divides another prime |
| **Branching Divisors** | `[4, 8, 10, 240]` | `[4, 8, 10, 240]` | `4->8->240` (len 3), `10->240` (len 2) | `[4, 8, 240]` | $240 \pmod 8 == 0$ yields larger chain |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        if not nums:
            return []
        
        nums.sort()
        n = len(nums)
        
        dp = [1] * n
        parent = [-1] * n
        
        max_len = 1
        max_idx = 0
        
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        parent[i] = j
            
            if dp[i] > max_len:
                max_len = dp[i]
                max_idx = i
                
        # Reconstruct path
        result = []
        curr = max_idx
        while curr != -1:
            result.append(nums[curr])
            curr = parent[curr]
            
        return result[::-1]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<int> largestDivisibleSubset(std::vector<int>& nums) {
        if (nums.empty()) return {};

        std::sort(nums.begin(), nums.end());
        int n = nums.size();

        std::vector<int> dp(n, 1);
        std::vector<int> parent(n, -1);

        int max_len = 1;
        int max_idx = 0;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (nums[i] % nums[j] == 0) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        parent[i] = j;
                    }
                }
            }
            if (dp[i] > max_len) {
                max_len = dp[i];
                max_idx = i;
            }
        }

        // Reconstruct subset
        std::vector<int> result;
        int curr = max_idx;
        while (curr != -1) {
            result.push_back(nums[curr]);
            curr = parent[curr];
        }

        std::reverse(result.begin(), result.end());
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

class Solution {
    public List<Integer> largestDivisibleSubset(int[] nums) {
        if (nums == null || nums.length == 0) return new ArrayList<>();

        Arrays.sort(nums);
        int n = nums.length;

        int[] dp = new int[n];
        int[] parent = new int[n];
        Arrays.fill(dp, 1);
        Arrays.fill(parent, -1);

        int maxLen = 1;
        int maxIdx = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] % nums[j] == 0) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        parent[i] = j;
                    }
                }
            }
            if (dp[i] > maxLen) {
                maxLen = dp[i];
                maxIdx = i;
            }
        }

        // Reconstruct subset
        List<Integer> result = new ArrayList<>();
        int curr = maxIdx;
        while (curr != -1) {
            result.add(nums[curr]);
            curr = parent[curr];
        }

        Collections.reverse(result);
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^2)$  
  Sorting takes $\mathcal{O}(N \log N)$. The nested loops compare each pair $(i, j)$ where $0 \le j < i < N$, taking $\frac{N(N - 1)}{2}$ steps. For $N \le 1000$, $\approx 5 \times 10^5$ operations, which finishes in $< 10$ milliseconds.
- **Space Complexity:** $\mathcal{O}(N)$  
  Arrays `dp` and `parent` each require $\mathcal{O}(N)$ memory.

---

### Takeaway Pattern & Interview Traps

1. **Sorting Enables Transitivity:**
   - Without sorting, verifying that every pair divides each other requires pairwise validation of all $\binom{K}{2}$ pairs, turning the problem into finding the Maximum Clique in an undirected graph (NP-hard). Sorting enforces linear ordering, making a single modulo check against the largest member sufficient.
2. **Reconstruction vs. Length Only:**
   - Notice that the question asks for the **subset itself**, not just the maximum length. Always include a predecessor/parent pointer array to avoid expensive sub-array copies inside the DP transition.