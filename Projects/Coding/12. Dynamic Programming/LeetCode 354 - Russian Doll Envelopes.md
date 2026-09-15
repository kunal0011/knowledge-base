---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 354: Russian Doll Envelopes"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - binary-search
  - longest-increasing-subsequence
  - google
  - amazon
  - meta
---

# LeetCode 354: Russian Doll Envelopes

**Target Companies:** Google, Amazon, Meta, Microsoft, ByteDance  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Longest Increasing Subsequence (LIS) / Patience Sorting / Binary Search  

---

### Problem Statement

You are given a 2D array of integers `envelopes` where `envelopes[i] = [wi, hi]` represents the width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are **strictly greater** than the other envelope's width and height.

Return the **maximum number of envelopes** you can Russian doll (i.e., put one inside the other).

**Note:** You cannot rotate an envelope.

---

### Input & Output Formats & Constraints

- **Input:** `envelopes: List[List[int]]` — 2D array where each element is `[w, h]`.
- **Output:** `int` — Maximum number of nested envelopes.
- **Constraints:**
  - $1 \le \text{envelopes.length} \le 10^5$
  - $\text{envelopes}[i].\text{length} == 2$
  - $1 \le w_i, h_i \le 10^5$

---

### Key Idea & Intuition

1. **Reduction to Longest Increasing Subsequence (LIS):**
   - We need a chain of envelopes $(w_1, h_1), (w_2, h_2), \dots, (w_k, h_k)$ such that $w_1 < w_2 < \dots < w_k$ and $h_1 < h_2 < \dots < h_k$.
   - If we sort the envelopes by width $w$ ascending, the width dimension is largely ordered. Can we then simply find the LIS on heights $h$?
   - **Crucial Conflict (Equal Widths):**
     - If two envelopes have identical widths, say $[6, 4]$ and $[6, 7]$, neither can fit into the other ($6 \not< 6$).
     - If we sort height ascending for identical widths, an LIS on height would pick both $[6, 4]$ and $[6, 7]$ because $4 < 7$. This violates the strict width condition!
   - **The Invariant Fix (Sort Width Ascending, Height Descending):**
     - Sort envelopes primarily by $w$ **ascending**.
     - For envelopes with the **same $w$**, sort $h$ **descending**!
     - *Why descending?* Because if heights with the same width are descending, no two envelopes with identical widths can ever be part of a strictly increasing subsequence of heights! The larger height will appear earlier and will never be extended by a smaller or equal height with the same width.
     - Consequently, the problem reduces strictly to 1D LIS on the `height` coordinates.

2. **$O(N \log N)$ Patience Sorting / Binary Search:**
   - Because $N \le 10^5$, an $O(N^2)$ DP will result in Time Limit Exceeded (TLE).
   - We maintain an active tails array `tails`, where `tails[len]` is the smallest tail of all increasing subsequences of length `len + 1`.
   - For each height $h$:
     - Binary search (`bisect_left` / `lower_bound`) to find the first element in `tails` $\ge h$.
     - If no such element exists, append $h$ to `tails`.
     - Otherwise, overwrite that element with $h$.
   - The length of `tails` at the end is the maximum nesting depth.

---

### Solution Approach (Step-by-Step)

1. **Sort Envelopes:**
   - Sort by key: `(w, -h)` in Python, or custom comparator `a[0] < b[0] || (a[0] == b[0] && a[1] > b[1])` in C++ / Java.
2. **Execute LIS on Heights:**
   - Initialize an empty dynamic array `tails`.
   - For each envelope `[w, h]`:
     - Perform binary search to find index `idx` of the first element in `tails` that is $\ge h$.
     - If `idx == len(tails)`, append $h$.
     - Else, replace `tails[idx] = h`.
3. **Return Length:**
   - Return `len(tails)`.

---

### Visual Algorithm Walkthrough

Given `envelopes = [[5, 4], [6, 4], [6, 7], [2, 3]]`.

**Step 1: Custom Sort (`w` asc, `h` desc on tie)**
```
Original: [[5, 4], [6, 4], [6, 7], [2, 3]]
Sorted:   [[2, 3], [5, 4], [6, 7], [6, 4]]
Heights:   [  3,      4,      7,      4   ]
```
Notice for $w = 6$, height $7$ comes before $4$.

**Step 2: Binary Search LIS on Heights**
```
Height = 3:
  tails is empty -> tails = [3]

Height = 4:
  3 < 4, append -> tails = [3, 4]

Height = 7:
  4 < 7, append -> tails = [3, 4, 7]

Height = 4:
  Binary search for 4 finds tails[1] = 4
  Replace tails[1] with 4 -> tails = [3, 4, 7]

Final tails length = 3
Max Russian doll nesting = 3  (Chain: [2, 3] -> [5, 4] -> [6, 7])
```

---

### Solved Examples with Multiple Inputs

| Case | Input `envelopes` | Sorted by `(w asc, h desc)` | Height Sequence | Result | Explanation |
|---|---|---|---|---|---|
| **Standard** | `[[5,4],[6,4],[6,7],[2,3]]` | `[[2,3],[5,4],[6,7],[6,4]]` | `[3, 4, 7, 4]` | `3` | `[2,3] -> [5,4] -> [6,7]` |
| **All Same Width** | `[[1,1],[1,2],[1,3]]` | `[[1,3],[1,2],[1,1]]` | `[3, 2, 1]` | `1` | Cannot nest any since widths are equal |
| **All Same Height** | `[[1,5],[2,5],[3,5]]` | `[[1,5],[2,5],[3,5]]` | `[5, 5, 5]` | `1` | Heights must strictly increase; cannot nest |
| **Strict Diagonal** | `[[1,1],[2,2],[3,3],[4,4]]` | `[[1,1],[2,2],[3,3],[4,4]]` | `[1, 2, 3, 4]` | `4` | All strictly increase in both dimensions |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
import bisect

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        
        # Sort width ascending; if widths match, sort height descending
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        
        # LIS on heights using patience sorting (binary search)
        tails: List[int] = []
        for _, h in envelopes:
            idx = bisect.bisect_left(tails, h)
            if idx == len(tails):
                tails.append(h)
            else:
                tails[idx] = h
                
        return len(tails)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxEnvelopes(std::vector<std::vector<int>>& envelopes) {
        if (envelopes.empty()) return 0;

        // Sort width asc, height desc on ties
        std::sort(envelopes.begin(), envelopes.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
            if (a[0] == b[0]) {
                return a[1] > b[1];
            }
            return a[0] < b[0];
        });

        // Patience sorting / LIS on height
        std::vector<int> tails;
        for (const auto& env : envelopes) {
            int h = env[1];
            auto it = std::lower_bound(tails.begin(), tails.end(), h);
            if (it == tails.end()) {
                tails.push_back(h);
            } else {
                *it = h;
            }
        }

        return static_cast<int>(tails.size());
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public int maxEnvelopes(int[][] envelopes) {
        if (envelopes == null || envelopes.length == 0) return 0;

        // Sort width asc; if width tie, sort height desc
        Arrays.sort(envelopes, (a, b) -> {
            if (a[0] == b[0]) {
                return Integer.compare(b[1], a[1]);
            }
            return Integer.compare(a[0], b[0]);
        });

        // Patience sorting / LIS on height
        int[] tails = new int[envelopes.length];
        int len = 0;

        for (int[] env : envelopes) {
            int h = env[1];
            int left = 0, right = len;
            // Binary search for insertion point
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (tails[mid] < h) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            tails[left] = h;
            if (left == len) {
                len++;
            }
        }

        return len;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \log N)$  
  Sorting $N$ envelopes takes $\mathcal{O}(N \log N)$. Iterating through $N$ envelopes while performing a binary search of cost $\mathcal{O}(\log N)$ on `tails` takes $\mathcal{O}(N \log N)$. Overall time is strictly $\mathcal{O}(N \log N)$, easily handling $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(N)$  
  The `tails` array stores at most $N$ elements in the worst case.

---

### Takeaway Pattern & Interview Traps

1. **The Tie-Breaking Direction Trap:**
   - The single most common failure in this problem is sorting height in *ascending* order when widths match.
   - If widths match and heights ascend, `[3, 3]` and `[3, 4]` would both be picked by standard LIS because $3 < 4$, illegally claiming an envelope can nest inside another of equal width.
   - Sorting matching widths in *descending* height guarantees that between `[3, 4]` and `[3, 3]`, $4$ appears first, so $3$ can never extend it, preventing two envelopes with width $3$ from both being selected.
2. **$O(N^2)$ TLE:**
   - A standard double-loop LIS approach is $O(N^2)$. With $N = 10^5$, $N^2 = 10^{10}$ operations, which will inevitably result in a TLE on LeetCode. Binary search LIS ($\mathcal{O}(N \log N)$) is strictly required.