---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - greedy
  - amazon
  - google
---

# LeetCode 321: Create Maximum Number

**Target Companies:** Google, Amazon, Apple, Meta  
**Difficulty:** Hard  
**Topic:** Monotonic Stack / Greedy / Lexicographical Suffix Merge

---

### Problem Statement

You are given two integer arrays `nums1` and `nums2` of lengths `m` and `n` respectively, representing digits of two numbers. You are also given an integer `k`.

Create the **maximum number of length `k`** from digits of the two arrays. The relative order of the digits from the same array must be preserved.

Return an array of the `k` digits representing the answer.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums1`: `List[int]`, where $m = \text{len}(nums1)$. $0 \le nums1[i] \le 9$.
  - `nums2`: `List[int]`, where $n = \text{len}(nums2)$. $0 \le nums2[j] \le 9$.
  - `k`: `int`, where $1 \le k \le m + n$.
- **Output:**
  - `List[int]`: An array of length $k$ representing the digits of the maximum number.
- **Constraints:**
  - $m = nums1.length$, $n = nums2.length$.
  - $1 \le m, n \le 500$.
  - $0 \le nums1[i], nums2[i] \le 9$.
  - $1 \le k \le m + n$.

---

### Key Idea & Intuition

Creating the maximum number of length $k$ by picking digits from two arrays preserves relative order. This means we are picking a subsequence of length $i$ from `nums1` and a subsequence of length $k - i$ from `nums2`, and merging them to form the lexicographically largest sequence.

The problem decomposes cleanly into three decoupled subproblems:

1. **Subproblem 1: Max Subsequence of Length $t$ from a Single Array (`maxSubsequence`)**
   - Given an array `nums` and target length $t$, find the lexicographically largest subsequence.
   - This is the classic **Monotonic Decreasing Stack** problem:
     - We are allowed to drop at most `drop = len(nums) - t` digits.
     - For each digit: while `drop > 0` and `stack` has a smaller digit on top (`stack[-1] < digit`), pop the stack and decrement `drop`.
     - Push the current digit.
     - Truncate stack to size $t$.

2. **Subproblem 2: Lexicographical Merge of Two Sequences (`merge`)**
   - Given two optimal subsequences $A$ and $B$ (of lengths $i$ and $k - i$), merge them into the largest possible sequence.
   - At each step, compare remaining suffix $A[p_1:]$ with $B[p_2:]$:
     - If $A[p_1:] > B[p_2:]$, pick $A[p_1++]$.
     - Else, pick $B[p_2++]$.
   - *Critical Trap:* When $A[p_1] == B[p_2]$, a simple lookahead is insufficient. You must compare the entire remaining suffixes lexicographically to see which branch exposes a larger digit sooner!

3. **Subproblem 3: Iterate Over Valid Splits**
   - Number of digits $i$ taken from `nums1` must satisfy:
     $$\max(0, k - n) \le i \le \min(k, m)$$
   - For each valid split $i$:
     - Let $A = \text{maxSubsequence}(nums1, i)$
     - Let $B = \text{maxSubsequence}(nums2, k - i)$
     - Let $candidate = \text{merge}(A, B)$
     - Update global best: $best = \max(best, candidate)$.

---

### Solution Approach (Step-by-Step)

1. **Implement `maxSubsequence(nums, t)`:**
   - If $t == 0$, return `[]`.
   - `drop = len(nums) - t`.
   - For `x in nums`:
     - While `drop > 0` and `stack` and `stack[-1] < x`:
       - `stack.pop()`
       - `drop -= 1`
     - `stack.append(x)`
   - Return `stack[:t]`.

2. **Implement `merge(a, b)`:**
   - Use two pointers $p_1, p_2$.
   - While $p_1 < \text{len}(a)$ or $p_2 < \text{len}(b)$:
     - Compare suffixes: `if a[p1:] > b[p2:]: pick a[p1]; p1 += 1`
     - Else: `pick b[p2]; p2 += 1`

3. **Global Search:**
   - Initialize `best = []`.
   - Loop $i$ from $\max(0, k - n)$ to $\min(k, m)$:
     - `cand = merge(maxSubsequence(nums1, i), maxSubsequence(nums2, k - i))`
     - If `cand > best`: `best = cand`.
   - Return `best`.

---

### Visual Algorithm Walkthrough

Let `nums1 = [3, 4, 6, 5]`, `nums2 = [9, 1, 2, 5, 8, 3]`, $k = 5$.

```
Search space for i (digits from nums1):
  m = 4, n = 6, k = 5
  max(0, 5 - 6) = 0 <= i <= min(5, 4) = 4

Consider candidate split i = 2:
  - From nums1 (m=4), pick best subsequence of length 2:
    drop = 4 - 2 = 2.
    Processing [3, 4, 6, 5]:
      3 -> stack=[3]
      4 -> 4 > 3, drop 3 -> stack=[4], drop=1
      6 -> 6 > 4, drop 4 -> stack=[6], drop=0
      5 -> stack=[6, 5]
    A = [6, 5]

  - From nums2 (n=6), pick best subsequence of length 3:
    drop = 6 - 3 = 3.
    Processing [9, 1, 2, 5, 8, 3]:
      stack drops 1, 2 when seeing 5 and 8.
    B = [9, 8, 3]

  - Merging A = [6, 5] and B = [9, 8, 3]:
    Compare [6, 5] vs [9, 8, 3] -> 9 > 6 -> Pick 9 from B.
    Compare [6, 5] vs [8, 3]    -> 8 > 6 -> Pick 8 from B.
    Compare [6, 5] vs [3]       -> 6 > 3 -> Pick 6 from A.
    Compare [5]    vs [3]       -> 5 > 3 -> Pick 5 from A.
    Remaining from B            -> Pick 3.

    Merged candidate: [9, 8, 6, 5, 3]
```

Comparing across all splits $i \in [0, 4]$, `[9, 8, 6, 5, 3]` is indeed the optimal result.

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Disjoint Choice

- **Input:** `nums1 = [3, 4, 6, 5]`, `nums2 = [9, 1, 2, 5, 8, 3]`, `k = 5`
- **Output:** `[9, 8, 6, 5, 3]`

#### Example 2: Suffix Tie-Breaking Requirement

- **Input:** `nums1 = [6, 7]`, `nums2 = [6, 0, 4]`, `k = 5`
- **Execution:**
  - $i=2 \implies A = [6, 7], B = [6, 0, 4]$.
  - At first step: $A[0] = 6, B[0] = 6$.
  - Comparing suffixes: $[6, 7] > [6, 0, 4]$ because at next index $7 > 0$.
  - Therefore, must pick $6$ from $A$ first!
  - Result: `[6, 7, 6, 0, 4]`.
- **Output:** `[6, 7, 6, 0, 4]`

#### Example 3: Full Selection from Single Array

- **Input:** `nums1 = [3, 9]`, `nums2 = [8, 9]`, `k = 3`
- **Output:** `[9, 8, 9]`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        m, n = len(nums1), len(nums2)

        def max_subsequence(nums: List[int], t: int) -> List[int]:
            """Finds the lexicographically largest subsequence of length t."""
            stack = []
            drop = len(nums) - t
            for num in nums:
                while drop > 0 and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:t]

        def merge(a: List[int], b: List[int]) -> List[int]:
            """Merges two sequences greedily by comparing remaining suffixes."""
            res = []
            p1, p2 = 0, 0
            len_a, len_b = len(a), len(b)
            
            while p1 < len_a or p2 < len_b:
                # Suffix comparison: slices in Python handle out-of-bounds gracefully
                if a[p1:] > b[p2:]:
                    res.append(a[p1])
                    p1 += 1
                else:
                    res.append(b[p2])
                    p2 += 1
            return res

        best: List[int] = []
        for i in range(max(0, k - n), min(k, m) + 1):
            sub1 = max_subsequence(nums1, i)
            sub2 = max_subsequence(nums2, k - i)
            candidate = merge(sub1, sub2)
            if not best or candidate > best:
                best = candidate

        return best
```

#### C++17

```cpp
#include <vector>
#include <algorithm>

class Solution {
private:
    std::vector<int> maxSubsequence(const std::vector<int>& nums, int t) {
        std::vector<int> stack;
        int drop = static_cast<int>(nums.size()) - t;
        for (int num : nums) {
            while (drop > 0 && !stack.empty() && stack.back() < num) {
                stack.pop_back();
                drop--;
            }
            stack.push_back(num);
        }
        stack.resize(t);
        return stack;
    }

    // Returns true if suffix of a starting at i is greater than suffix of b starting at j
    bool isGreater(const std::vector<int>& a, int i, const std::vector<int>& b, int j) {
        int n = static_cast<int>(a.size());
        int m = static_cast<int>(b.size());
        while (i < n && j < m) {
            if (a[i] != b[j]) return a[i] > b[j];
            i++;
            j++;
        }
        return (n - i) > (m - j);
    }

    std::vector<int> merge(const std::vector<int>& a, const std::vector<int>& b, int k) {
        std::vector<int> res(k);
        int p1 = 0, p2 = 0;
        for (int r = 0; r < k; ++r) {
            if (isGreater(a, p1, b, p2)) {
                res[r] = a[p1++];
            } else {
                res[r] = b[p2++];
            }
        }
        return res;
    }

public:
    std::vector<int> maxNumber(const std::vector<int>& nums1, const std::vector<int>& nums2, int k) {
        int m = static_cast<int>(nums1.size());
        int n = static_cast<int>(nums2.size());
        std::vector<int> best;

        int start = std::max(0, k - n);
        int end = std::min(k, m);

        for (int i = start; i <= end; ++i) {
            std::vector<int> sub1 = maxSubsequence(nums1, i);
            std::vector<int> sub2 = maxSubsequence(nums2, k - i);
            std::vector<int> candidate = merge(sub1, sub2, k);
            if (best.empty() || candidate > best) {
                best = candidate;
            }
        }

        return best;
    }
};
```

#### Java

```java
import java.util.Arrays;

public class Solution {
    private int[] maxSubsequence(int[] nums, int t) {
        int[] stack = new int[t];
        int top = -1;
        int drop = nums.length - t;

        for (int num : nums) {
            while (drop > 0 && top >= 0 && stack[top] < num) {
                top--;
                drop--;
            }
            if (top < t - 1) {
                stack[++top] = num;
            } else {
                drop--;
            }
        }
        return stack;
    }

    private boolean isGreater(int[] a, int i, int[] b, int j) {
        while (i < a.length && j < b.length) {
            if (a[i] != b[j]) return a[i] > b[j];
            i++;
            j++;
        }
        return (a.length - i) > (b.length - j);
    }

    private int[] merge(int[] a, int[] b, int k) {
        int[] res = new int[k];
        int p1 = 0, p2 = 0;
        for (int r = 0; r < k; r++) {
            if (isGreater(a, p1, b, p2)) {
                res[r] = a[p1++];
            } else {
                res[r] = b[p2++];
            }
        }
        return res;
    }

    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length;
        int n = nums2.length;
        int[] best = null;

        int start = Math.max(0, k - n);
        int end = Math.min(k, m);

        for (int i = start; i <= end; i++) {
            int[] sub1 = maxSubsequence(nums1, i);
            int[] sub2 = maxSubsequence(nums2, k - i);
            int[] cand = merge(sub1, sub2, k);

            if (best == null || isGreater(cand, 0, best, 0)) {
                best = cand;
            }
        }

        return best;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(k \cdot (m + n + k^2))$
  - There are at most $k + 1$ iterations for split $i$.
  - Computing `maxSubsequence` takes $\mathcal{O}(m + n)$ time.
  - Merging two arrays of total length $k$ with suffix comparisons takes $\mathcal{O}(k^2)$ time in the worst case (when many consecutive digits are identical).
  - Given $m, n \le 500$ and $k \le 1000$, total operations are well within the $10^7$ limit and execute in $< 50\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(k)$
  - Stores temporary subsequences of length at most $k$.

---

### Takeaway Pattern & Interview Traps

1. **Greedy Single Array vs Two Arrays:**
   - While a single array uses a pure monotonic stack (like LC 402), two arrays require splitting the quota $k$ and merging the results.
2. **Lexicographical Suffix Comparison Invariant:**
   - If $a[p_1] == b[p_2]$, choosing either arbitrarily breaks correctness. You must evaluate `isGreater(a, p1, b, p2)` across the entire suffix.
3. **Valid Split Boundaries:**
   - Make sure $i$ starts at $\max(0, k - n)$ (since `nums2` can contribute at most $n$ digits) and ends at $\min(k, m)$ (since `nums1` can contribute at most $m$ digits).