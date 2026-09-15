---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 646: Maximum Length of Pair Chain"
tags:
  - leetcode
  - coding
  - greedy
  - interval-scheduling
  - dynamic-programming
  - sorting
  - amazon
  - google
---

# LeetCode 646: Maximum Length of Pair Chain

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Greedy / Interval Scheduling / Sorting  

---

### Problem Statement

You are given an array of $n$ pairs `pairs` where `pairs[i] = [left_i, right_i]` and `left_i < right_i`.

A pair `p2 = [c, d]` follows a pair `p1 = [a, b]` if and only if `b < c`. A chain of pairs can be formed in this fashion.

Return the **longest chain** which can be formed.

You do not need to use up all the given intervals. You can select pairs in any order.

---

### Input & Output Formats & Constraints

- **Input:**
  - `pairs`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` ($1 \le \text{pairs.length} \le 1000$).
- **Output:**
  - `int` — the maximum number of pairs in a valid chain.
- **Constraints:**
  - $n == \text{pairs.length}$
  - $1 \le n \le 1000$
  - $-1000 \le \text{left}_i < \text{right}_i \le 1000$

---

### Key Idea & Intuition

Each pair `[a, b]` can be viewed as a time interval starting at $a$ and ending at $b$.
The chaining condition `b < c` means interval $[c, d]$ starts strictly after interval $[a, b]$ has completely finished.
Therefore:
> **Finding the longest chain of pairs is isomorphic to the classic Interval Scheduling / Activity Selection Problem: Select the maximum number of non-overlapping intervals.**

#### The Earliest Finish Time Greedy Invariant:
- To maximize the number of pairs we can chain together, we must always pick the pair that ends earliest (smallest $b$).
- An interval that finishes earlier leaves the maximum possible open timeline for subsequent pairs to begin.
- Therefore:
  1. Sort all pairs by their second coordinate `pairs[i][1]` ascending.
  2. Pick the first pair and record its end: `cur_end = pairs[0][1]`, `chain_length = 1`.
  3. Iterate through subsequent pairs `[c, d]`:
     - If $c > cur\_end$:
       - We can attach this pair to our chain: `chain_length += 1`, `cur_end = d`.
  4. Return `chain_length`.

---

### Solution Approach (Step-by-Step)

1. If `pairs` is empty, return 0.
2. Sort `pairs` in ascending order of `pair[1]`.
3. Initialize `chain_len = 1` and `cur_end = pairs[0][1]`.
4. Loop `i` from $1$ to $n - 1$:
   - If `pairs[i][0] > cur_end`:
     - `chain_len += 1`
     - `cur_end = pairs[i][1]`
5. Return `chain_len`.

---

### Visual Algorithm Walkthrough

For `pairs = [[1,2], [2,3], [3,4]]`:

```
1. Sorted by end:
   [1, 2], [2, 3], [3, 4]

Step 1: Pick [1, 2]
  chain_len = 1, cur_end = 2

Step 2: Inspect [2, 3]
  start = 2. Is 2 > cur_end(2)?
  2 > 2 is FALSE (must be strictly greater!).
  Cannot chain [2, 3] after [1, 2]. Skip.

Step 3: Inspect [3, 4]
  start = 3. Is 3 > cur_end(2)?
  3 > 2 is TRUE!
  chain_len = 2, cur_end = 4

Total longest chain = 2 ([1, 2] -> [3, 4]).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `pairs = [[1,2],[2,3],[3,4]]`
- **Output:** `2`

#### Example 2:
- **Input:** `pairs = [[1,2],[7,8],[4,5]]`
- **Tracing:**
  - Sorted: `[1, 2], [4, 5], [7, 8]`
  - Pick [1, 2] $\rightarrow$ pick [4, 5] $\rightarrow$ pick [7, 8].
- **Output:** `3`

#### Example 3 (Negative Coordinates):
- **Input:** `pairs = [[-10,-8],[-6,0],[1,7],[8,9],[-10,-1]]`
- **Tracing:**
  - Sorted by end: `[-10,-8], [-10,-1], [-6,0], [1,7], [8,9]`
  - Pick [-10,-8] $\rightarrow$ pick [-6,0] $\rightarrow$ pick [1,7] $\rightarrow$ pick [8,9].
- **Output:** `4`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # Sort pairs by second coordinate ascending
        pairs.sort(key=lambda x: x[1])
        
        chain_len = 0
        cur_end = float('-inf')
        
        for start, end in pairs:
            if start > cur_end:
                chain_len += 1
                cur_end = end
                
        return chain_len
```

#### C++17
```cpp
#include <vector>
#include <algorithm>
#include <climits>

class Solution {
public:
    int findLongestChain(std::vector<std::vector<int>>& pairs) {
        // Sort by end time
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a[1] < b[1];
        });
        
        int chain_len = 0;
        int cur_end = INT_MIN;
        
        for (const auto& pair : pairs) {
            if (chain_len == 0 || pair[0] > cur_end) {
                chain_len++;
                cur_end = pair[1];
            }
        }
        
        return chain_len;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int findLongestChain(int[][] pairs) {
        // Sort by second element ascending
        Arrays.sort(pairs, (a, b) -> Integer.compare(a[1], b[1]));
        
        int chainLen = 0;
        int curEnd = Integer.MIN_VALUE;
        
        for (int[] pair : pairs) {
            if (chainLen == 0 || pair[0] > curEnd) {
                chainLen++;
                curEnd = pair[1];
            }
        }
        
        return chainLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n \log n)$
  - Sorting the $n$ pairs by second element takes $\mathcal{O}(n \log n)$ time.
  - The linear pass through the sorted list takes $\mathcal{O}(n)$ time.
  - Overall time is $\mathcal{O}(n \log n)$, which takes $< 3\text{ ms}$ for $n = 1000$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Sorting is performed in-place with $\mathcal{O}(1)$ additional memory.

---

### Takeaway Pattern & Interview Traps

- **Greedy Beats DP:** Although this problem can be formulated as Longest Increasing Subsequence DP in $\mathcal{O}(n^2)$ time, sorting by the second element yields an $\mathcal{O}(n \log n)$ greedy solution.
- **Strict Inequality:** The condition is strictly $b < c$, so touching endpoints like $[1, 2]$ and $[2, 3]$ cannot chain together.