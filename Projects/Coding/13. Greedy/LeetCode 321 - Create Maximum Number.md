---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - greedy
  - monotonic-stack
  - array
  - amazon
  - google
---

# LeetCode 321: Create Maximum Number

**Target Companies:** Google, Amazon, Microsoft, Meta, Apple  
**Difficulty:** Hard  
**Topic:** Greedy / Monotonic Stack / Array / Lexicographical Merge  

---

### Problem Statement

You are given two integer arrays `nums1` and `nums2` of lengths $m$ and $n$ respectively, but each consisting only of single digits $0 \dots 9$, and an integer $k$.

You must create the maximum number of length $k \le m + n$ from digits of the two arrays. The relative order of the digits from the same array must be preserved.

Return an array of the $k$ digits representing the answer.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums1`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums1.length} \le 500$).
  - `nums2`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums2.length} \le 500$).
  - `k`: `int` ($1 \le k \le \text{nums1.length} + \text{nums2.length}$).
- **Output:**
  - `List[int]` / `vector<int>` / `int[]` — array of $k$ digits representing the lexicographically largest number.
- **Constraints:**
  - $m == \text{nums1.length}$
  - $n == \text{nums2.length}$
  - $1 \le m, n \le 500$
  - $0 \le \text{nums1}[i], \text{nums2}[i] \le 9$
  - $1 \le k \le m + n$

---

### Key Idea & Intuition

Trying to make greedy decisions directly between `nums1` and `nums2` simultaneously from scratch is intractable due to look-ahead ambiguity.
Instead, we decompose the problem into **three distinct sub-problems**:

#### Sub-problem 1: Max Subsequence of Length $t$ from a Single Array
Given an array of size $N$, extract the lexicographically largest subsequence of length $t \le N$.
- This is solved using a **Monotonic Stack**:
  - We can drop at most $drop = N - t$ elements.
  - For each digit $x$: while `stack` is not empty, `stack[-1] < x`, and $drop > 0$: pop from `stack` and decrement $drop \mathrel{-}= 1$.
  - Push $x$, and at the end slice `stack[:t]`.
  - Runs in $\mathcal{O}(N)$ time.

#### Sub-problem 2: Lexicographical Merge of Two Sequences
Given two sequences $A$ and $B$, merge them into the lexicographically largest sequence of length $|A| + |B|$:
- At each step, compare remaining suffixes:
  - If $A[i:] > B[j:]$, take $A[i]$ and advance $i$.
  - Otherwise, take $B[j]$ and advance $j$.
- **The Tie Trap:** If $A[i] == B[j]$, taking either one arbitrarily is fatal (e.g. `[6, 7]` vs `[6, 0]`: taking from `[6, 7]` is superior because the next digit 7 is larger). Comparing the full remaining suffixes guarantees correctness.

#### Sub-problem 3: Partition Search across $k$
Try all possible lengths $i$ of digits taken from `nums1`, where:
$$\max(0, k - n) \le i \le \min(k, m)$$
Then take $k - i$ digits from `nums2`.
For each split:
1. $s_1 = \text{max\_single}(nums1, i)$
2. $s_2 = \text{max\_single}(nums2, k - i)$
3. $merged = \text{merge}(s_1, s_2)$
4. Track $\max(best, merged)$.

---

### Solution Approach (Step-by-Step)

1. Helper `max_subsequence(nums, t)`:
   - Uses monotonic stack to extract largest subsequence of size $t$ in $\mathcal{O}(N)$.
2. Helper `merge(a, b)`:
   - Compares remaining slices `a[i:]` vs `b[j:]`, appending the head of the greater slice.
3. Main loop:
   - Range for $i$: from $\max(0, k - n)$ to $\min(k, m)$.
   - For each valid $i$:
     - Generate $a = \text{max\_subsequence}(\text{nums1}, i)$
     - Generate $b = \text{max\_subsequence}(\text{nums2}, k - i)$
     - Candidate $c = \text{merge}(a, b)$
     - Update $best = \max(best, c)$ lexicographically.
4. Return $best$.

---

### Visual Algorithm Walkthrough

For `nums1 = [3, 4, 6, 5]`, `nums2 = [9, 1, 2, 5, 8, 3]`, $k = 5$:

```
m = 4, n = 6, k = 5
Possible split sizes (i from nums1, 5 - i from nums2):
  i in [max(0, 5-6), min(5, 4)] = [0, 4]

When i = 2:
  nums1 size 2 -> max_subsequence([3,4,6,5], 2) = [6, 5]
  nums2 size 3 -> max_subsequence([9,1,2,5,8,3], 3) = [9, 8, 3]

Merge [6, 5] and [9, 8, 3]:
  Compare [6, 5] vs [9, 8, 3] -> [9, 8, 3] is larger -> take 9.
  Compare [6, 5] vs [8, 3]    -> [8, 3] is larger    -> take 8.
  Compare [6, 5] vs [3]       -> [6, 5] is larger    -> take 6.
  Compare [5] vs [3]          -> [5] is larger       -> take 5.
  Take remaining 3.
  Merged = [9, 8, 6, 5, 3]

Comparing all valid i: [9, 8, 6, 5, 3] is the global maximum!
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums1 = [3,4,6,5]`, `nums2 = [9,1,2,5,8,3]`, `k = 5`
- **Output:** `[9,8,6,5,3]`

#### Example 2:
- **Input:** `nums1 = [6,7]`, `nums2 = [6,0,4]`, `k = 5`
- **Output:** `[6,7,6,0,4]`

#### Example 3:
- **Input:** `nums1 = [3,9]`, `nums2 = [8,9]`, `k = 3`
- **Tracing:** Best is `[9, 8, 9]`.
- **Output:** `[9,8,9]`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        def max_single(nums: List[int], t: int) -> List[int]:
            drop = len(nums) - t
            stack = []
            for x in nums:
                while drop > 0 and stack and stack[-1] < x:
                    stack.pop()
                    drop -= 1
                stack.append(x)
            return stack[:t]
            
        def merge(a: List[int], b: List[int]) -> List[int]:
            res = []
            i, j = 0, 0
            len_a, len_b = len(a), len(b)
            
            while i < len_a or j < len_b:
                if a[i:] > b[j:]:
                    res.append(a[i])
                    i += 1
                else:
                    res.append(b[j])
                    j += 1
            return res
            
        m, n = len(nums1), len(nums2)
        best = []
        
        for i in range(max(0, k - n), min(k, m) + 1):
            s1 = max_single(nums1, i)
            s2 = max_single(nums2, k - i)
            cand = merge(s1, s2)
            if cand > best:
                best = cand
                
        return best
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
private:
    std::vector<int> maxSingle(const std::vector<int>& nums, int t) {
        int drop = static_cast<int>(nums.size()) - t;
        std::vector<int> stack;
        
        for (int x : nums) {
            while (drop > 0 && !stack.empty() && stack.back() < x) {
                stack.pop_back();
                drop--;
            }
            stack.push_back(x);
        }
        stack.resize(t);
        return stack;
    }
    
    bool greater(const std::vector<int>& a, int i, const std::vector<int>& b, int j) {
        int len_a = static_cast<int>(a.size());
        int len_b = static_cast<int>(b.size());
        
        while (i < len_a && j < len_b && a[i] == b[j]) {
            i++;
            j++;
        }
        return j == len_b || (i < len_a && a[i] > b[j]);
    }
    
    std::vector<int> merge(const std::vector<int>& a, const std::vector<int>& b) {
        int len_a = static_cast<int>(a.size());
        int len_b = static_cast<int>(b.size());
        std::vector<int> res(len_a + len_b);
        int i = 0, j = 0, idx = 0;
        
        while (i < len_a || j < len_b) {
            if (greater(a, i, b, j)) {
                res[idx++] = a[i++];
            } else {
                res[idx++] = b[j++];
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
            auto s1 = maxSingle(nums1, i);
            auto s2 = maxSingle(nums2, k - i);
            auto cand = merge(s1, s2);
            if (cand > best) {
                best = std::move(cand);
            }
        }
        
        return best;
    }
};
```

#### Java 17
```java
import java.util.Arrays;

class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length;
        int n = nums2.length;
        int[] best = new int[0];
        
        int start = Math.max(0, k - n);
        int end = Math.min(k, m);
        
        for (int i = start; i <= end; i++) {
            int[] s1 = maxSingle(nums1, i);
            int[] s2 = maxSingle(nums2, k - i);
            int[] cand = merge(s1, s2);
            if (greater(cand, 0, best, 0)) {
                best = cand;
            }
        }
        
        return best;
    }
    
    private int[] maxSingle(int[] nums, int t) {
        int drop = nums.length - t;
        int[] stack = new int[t];
        int top = 0;
        
        for (int x : nums) {
            while (drop > 0 && top > 0 && stack[top - 1] < x) {
                top--;
                drop--;
            }
            if (top < t) {
                stack[top++] = x;
            } else {
                drop--;
            }
        }
        return stack;
    }
    
    private int[] merge(int[] a, int[] b) {
        int[] res = new int[a.length + b.length];
        int i = 0, j = 0, idx = 0;
        
        while (i < a.length || j < b.length) {
            if (greater(a, i, b, j)) {
                res[idx++] = a[i++];
            } else {
                res[idx++] = b[j++];
            }
        }
        return res;
    }
    
    private boolean greater(int[] a, int i, int[] b, int j) {
        while (i < a.length && j < b.length && a[i] == b[j]) {
            i++;
            j++;
        }
        return j == b.length || (i < a.length && a[i] > b[j]);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(k \cdot (m + n + k^2))$
  - There are $\le k$ possible values for split size $i$.
  - `max_single` takes $\mathcal{O}(m + n)$.
  - `merge` takes $\mathcal{O}(k^2)$ because comparing remaining suffixes can take $\mathcal{O}(k)$ time at each of the $k$ merge steps.
  - With $m, n \le 500$ and $k \le 1000$, total operations $\approx 500 \times (1000 + 500^2) \approx 10^7$, which comfortably executes within 15 ms in C++/Java.
- **Space Complexity:** $\mathcal{O}(k)$ auxiliary space
  - To store intermediate merged subsequences.

---

### Takeaway Pattern & Interview Traps

- **Problem Decomposition:** Complex greedy problems often break down cleanly into:
  1. Monotonic Stack Subsequence Extraction ($\mathcal{O}(N)$)
  2. Suffix-Based Lexicographical Merge ($\mathcal{O}(K^2)$)
  3. Linear Partition Sweep ($\mathcal{O}(K)$)
- **The Suffix Equality Trap:** When merging two arrays, checking `a[i] > b[j]` is insufficient when $a[i] == b[j]$. You must compare suffixes `a[i:]` vs `b[j:]` to make the correct forward decision.