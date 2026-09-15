---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 646: Maximum Length of Pair Chain"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - greedy
  - interval-scheduling
  - sorting
  - amazon
  - google
---

# LeetCode 646: Maximum Length of Pair Chain

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy Interval Scheduling / Dynamic Programming (LIS on Intervals) / Sorting  

---

### Problem Statement

You are given an array of `n` pairs `pairs` where `pairs[i] = [left_i, right_i]` and `left_i < right_i`.

A pair `p2 = [c, d]` **follows** a pair `p1 = [a, b]` if and only if `b < c`. A chain of pairs can be formed in this fashion.

Return the length longest chain which can be formed.

You do not need to use up all the given intervals. You can select pairs in any order.

---

### Input & Output Formats & Constraints

- **Input:** `pairs: List[List[int]]` — 2D list of integer pairs where $a < b$.
- **Output:** `int` — Maximum possible length of a valid pair chain.
- **Constraints:**
  - $n == \text{pairs.length}$
  - $1 \le n \le 1000$
  - $-1000 \le \text{left}_i < \text{right}_i \le 1000$

---

### Key Idea & Intuition

1. **Equivalence to Interval Scheduling (Greedy Choice Property):**
   - This problem is structurally identical to the classical **Activity Selection / Interval Scheduling Problem** (and LeetCode 435: Non-overlapping Intervals).
   - Each pair $[a, b]$ represents an interval starting at $a$ and ending at $b$.
   - To maximize the number of non-overlapping intervals, we should always greedily select the interval that **finishes earliest** (i.e. has the smallest ending value $b$).
   - An interval that finishes earlier leaves the maximum possible remaining room for future pairs to attach.

2. **Greedy Algorithm ($\mathcal{O}(N \log N)$ Time, $\mathcal{O}(1)$ Space — Optimal):**
   - Sort `pairs` by their second element $b$ in ascending order.
   - Maintain `curr_end` initialized to $-\infty$.
   - For each pair $[a, b]$:
     - If $a > \text{curr\_end}$, we can chain this pair!
     - Increment `chain_len += 1` and update `curr_end = b`.
   - Return `chain_len`.

3. **Dynamic Programming Alternative ($\mathcal{O}(N^2)$ Time — LIS Variant):**
   - Sort `pairs` by their first element $a$.
   - Let $\text{dp}[i]$ be the maximum chain length ending at `pairs[i]`:
     $$\text{dp}[i] = 1 + \max_{\substack{j < i \\ \text{pairs}[j][1] < \text{pairs}[i][0]}} \text{dp}[j]$$
   - While valid for small $N \le 1000$, the Greedy approach is strictly superior in both time and space.

---

### Solution Approach (Step-by-Step)

1. **Sort Pairs by End Coordinate:**
   - Sort `pairs` using key `x[1]` (the right endpoint) ascending.
2. **Greedy Traversal:**
   - Initialize `curr_end = -float('inf')` and `chain_len = 0`.
   - For each pair `[a, b]` in `pairs`:
     - If `a > curr_end`:
       - `chain_len += 1`
       - `curr_end = b`
3. **Return:**
   - Return `chain_len`.

---

### Visual Algorithm Walkthrough

For `pairs = [[1, 2], [7, 8], [4, 5]]`:

```
Step 1: Sort by end coordinate (pairs[i][1]):
  Original: [[1, 2], [7, 8], [4, 5]]
  Sorted:   [[1, 2], [4, 5], [7, 8]]
  Ends:        2       5       8

Step 2: Linear Greedy Scan
  curr_end = -inf, chain_len = 0

  Pair 0: [1, 2]
    1 > -inf -> Valid!
    chain_len = 1, curr_end = 2

  Pair 1: [4, 5]
    4 > 2 -> Valid!
    chain_len = 2, curr_end = 5

  Pair 2: [7, 8]
    7 > 5 -> Valid!
    chain_len = 3, curr_end = 8

Final Result: chain_len = 3.
Chain: [1, 2] -> [4, 5] -> [7, 8].
```

---

### Solved Examples with Multiple Inputs

| Case | `pairs` | Sorted by Second Element | Optimal Chain | Result |
|---|---|---|---|---|
| **Standard** | `[[1,2],[2,3],[3,4]]` | `[[1,2],[2,3],[3,4]]` | `[1,2] -> [3,4]` | `2` |
| **All Disjoint** | `[[1,2],[7,8],[4,5]]` | `[[1,2],[4,5],[7,8]]` | `[1,2] -> [4,5] -> [7,8]` | `3` |
| **Complete Overlap** | `[[1,10],[2,3],[4,5],[6,7]]` | `[[2,3],[4,5],[6,7],[1,10]]` | `[2,3] -> [4,5] -> [6,7]` | `3` |
| **Single Pair** | `[[5, 10]]` | `[[5, 10]]` | `[5, 10]` | `1` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Greedy Optimal)
```python
from typing import List

class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # Sort by right coordinate ascending
        pairs.sort(key=lambda x: x[1])
        
        curr_end = float('-inf')
        chain_len = 0
        
        for a, b in pairs:
            if a > curr_end:
                chain_len += 1
                curr_end = b
                
        return chain_len
```

#### 2. C++ (C++17 / STL — Greedy Optimal)
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int findLongestChain(std::vector<std::vector<int>>& pairs) {
        // Sort by right endpoint
        std::sort(pairs.begin(), pairs.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            return a[1] < b[1];
        });

        int curr_end = INT_MIN;
        int chain_len = 0;

        for (const auto& p : pairs) {
            // Must strictly satisfy a > curr_end (b < c)
            if (chain_len == 0 || p[0] > curr_end) {
                chain_len++;
                curr_end = p[1];
            }
        }

        return chain_len;
    }
};
```

#### 3. Java (Modern, Typed — Greedy Optimal)
```java
import java.util.Arrays;
import java.util.Comparator;

class Solution {
    public int findLongestChain(int[][] pairs) {
        // Sort by right coordinate
        Arrays.sort(pairs, Comparator.comparingInt(a -> a[1]));

        int currEnd = Integer.MIN_VALUE;
        int chainLen = 0;

        for (int[] p : pairs) {
            if (chainLen == 0 || p[0] > currEnd) {
                chainLen++;
                currEnd = p[1];
            }
        }

        return chainLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$  
  Sorting $N$ pairs takes $\mathcal{O}(N \log N)$. The subsequent greedy traversal runs in a single pass of $\mathcal{O}(N)$ time. For $N \le 1000$, finishes in $< 3$ ms.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space (ignoring sorting recursion stack).

---

### Takeaway Pattern & Interview Traps

1. **Strict Inequality ($b < c$):**
   - Notice the condition: pair $[c, d]$ follows $[a, b]$ iff $b < c$.
   - Unlike LeetCode 435 where touching was permitted, here touching is strictly forbidden ($a > \text{curr\_end}$).
2. **Why End-Time Sorting Beats Start-Time Sorting:**
   - If sorted by start time $a$, choosing $[a_1, b_1]$ over $[a_2, b_2]$ does not guarantee an optimal future state because $b_1$ could be arbitrarily large (e.g. $[1, 100]$ vs $[2, 3]$). Sorting by end time $b$ ensures every chosen element terminates as early as possible.