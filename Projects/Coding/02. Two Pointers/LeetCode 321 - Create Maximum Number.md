---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 321: Create Maximum Number"
tags:
  - leetcode
  - coding
  - two-pointers
  - monotonic-stack
  - greedy
  - google
  - amazon
---

# LeetCode 321: Create Maximum Number

**Target Companies:** Google, Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Monotonic Stack / Two Pointers / Greedy Merge  

---

### Problem Statement

You are given two integer arrays `nums1` and `nums2` of lengths $m$ and $n$ respectively, representing the digits of two numbers, and an integer $k$.

You must create a number of length $k$ by selecting digits from `nums1` and `nums2`, preserving the relative order of the digits from the original arrays, such that the resulting number is **as large as possible** (lexicographically maximum).

Return the resulting number as an array of $k$ digits.

---

### Input & Output Formats & Constraints

- **Input:** `nums1: List[int]`, `nums2: List[int]`, `k: int`
- **Output:** `List[int]` (array of length $k$)
- **Constraints:**
  - $m == \text{nums1.length}$, $n == \text{nums2.length}$
  - $1 \le m, n \le 500$
  - $0 \le \text{nums1}[i], \text{nums2}[i] \le 9$
  - $1 \le k \le m + n$

---

### Key Idea & Intuition

The problem decomposes elegantly into three distinct algorithmic phases:

1. **Subproblem 1: Max Subsequence of Length $X$ from an Array (Monotonic Stack)**
   - To find the largest sequence of length $x$ from `nums`, we are allowed to drop at most `len(nums) - x` elements.
   - We maintain a monotonic non-increasing stack. Whenever the incoming element is strictly greater than the top of the stack and our remaining drop budget is $> 0$, we pop from the stack.

2. **Subproblem 2: Merge Two Subsequences (Two Pointers with Lexicographical Lookahead)**
   - Merging two sequences greedily requires choosing the larger digit between pointer $i$ in `seq1` and pointer $j$ in `seq2`.
   - **Crucial Nuance:** When `seq1[i] == seq2[j]`, choosing naively breaks correctness! We must compare the entire remaining suffixes `seq1[i:]` vs. `seq2[j:]` lexicographically and take from whichever array has the larger suffix.

3. **Master Loop: Partition $k$ into $(i, k - i)$**
   - We test every feasible allocation $i \in [\max(0, k - n), \min(k, m)]$.
   - For each partition, extract optimal subsequences of size $i$ and $k - i$, merge them, and maintain the global maximum.

---

### Solution Approach (Step-by-Step)

1. Define `max_subsequence(nums, count)`:
   - Calculate drop budget: `drop = len(nums) - count`.
   - Loop `x` through `nums`: while `drop > 0` and `stack` not empty and `stack[-1] < x`, pop and `drop -= 1`. Append `x`.
   - Return `stack[:count]`.
2. Define `merge(seq1, seq2)`:
   - Maintain pointers `i = 0, j = 0`.
   - While `i < len(seq1)` or `j < len(seq2)`:
     - Compare remaining suffixes: if `seq1[i:] > seq2[j:]`, pick `seq1[i]` and `i += 1`.
     - Else pick `seq2[j]` and `j += 1`.
3. Loop $i$ from $\max(0, k - n)$ to $\min(k, m)$:
   - `sub1 = max_subsequence(nums1, i)`
   - `sub2 = max_subsequence(nums2, k - i)`
   - `candidate = merge(sub1, sub2)`
   - Update `best = max(best, candidate)`.
4. Return `best`.

---

### Visual Algorithm Walkthrough

```
nums1 = [3, 4, 6, 5], nums2 = [9, 1, 2, 5, 8, 3], k = 5

Feasible split: i = 2 from nums1, k - i = 3 from nums2

1. Subsequence 1 (length 2 from nums1):
   nums1 = [3, 4, 6, 5], drop budget = 4 - 2 = 2
   - see 3: stack = [3]
   - see 4: 4 > 3, pop 3 (drop=1), push 4 -> stack = [4]
   - see 6: 6 > 4, pop 4 (drop=0), push 6 -> stack = [6]
   - see 5: drop=0, push 5 -> stack = [6, 5]
   Result: [6, 5]

2. Subsequence 2 (length 3 from nums2):
   nums2 = [9, 1, 2, 5, 8, 3], drop budget = 6 - 3 = 3
   - Result after monotonic stack drops: [9, 8, 3]

3. Two-Pointer Greedy Merge:
   seq1 = [6, 5], seq2 = [9, 8, 3]
   - Compare [6, 5] vs [9, 8, 3] -> 9 > 6 -> pick 9. Res = [9]
   - Compare [6, 5] vs [8, 3]    -> 8 > 6 -> pick 8. Res = [9, 8]
   - Compare [6, 5] vs [3]       -> 6 > 3 -> pick 6. Res = [9, 8, 6]
   - Compare [5] vs [3]          -> 5 > 3 -> pick 5. Res = [9, 8, 6, 5]
   - seq1 empty                  -> pick 3. Res = [9, 8, 6, 5, 3]

Final best array formed: [9, 8, 6, 5, 3]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Cross-Array Merge
- **Input:** `nums1 = [3, 4, 6, 5]`, `nums2 = [9, 1, 2, 5, 8, 3]`, `k = 5`
- **Candidate Splits:**
  - $i = 0$: $[] + [9, 5, 8, 3] \to$ invalid length 4
  - $i = 1$: $[6] + [9, 2, 5, 8] \to [9, 6, 2, 5, 8]$
  - $i = 2$: $[6, 5] + [9, 8, 3] \to [9, 8, 6, 5, 3]$ (Best)
- **Output:** `[9, 8, 6, 5, 3]`

#### Example 2: Duplicate Lookahead Collision
- **Input:** `nums1 = [6, 7]`, `nums2 = [6, 0, 4]`, `k = 5`
- **Trace:**
  - Need all 5 digits: $i = 2$ from nums1, $3$ from nums2.
  - `seq1 = [6, 7]`, `seq2 = [6, 0, 4]`.
  - Both start with 6! Compare `[6, 7]` vs `[6, 0, 4]`: because $7 > 0$, `seq1` is larger.
  - Pick 6 from `seq1`, then 7, then 6, 0, 4 from `seq2`.
- **Output:** `[6, 7, 6, 0, 4]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def maxNumber(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        m, n = len(nums1), len(nums2)
        
        def max_subsequence(nums: List[int], count: int) -> List[int]:
            if count == 0:
                return []
            drop = len(nums) - count
            stack: List[int] = []
            for num in nums:
                while drop > 0 and stack and stack[-1] < num:
                    stack.pop()
                    drop -= 1
                stack.append(num)
            return stack[:count]
            
        def merge(seq1: List[int], seq2: List[int]) -> List[int]:
            res: List[int] = []
            i, j = 0, 0
            n1, n2 = len(seq1), len(seq2)
            while i < n1 or j < n2:
                # Suffix comparison avoids greedily picking the wrong tie
                if seq1[i:] > seq2[j:]:
                    res.append(seq1[i])
                    i += 1
                else:
                    res.append(seq2[j])
                    j += 1
            return res
            
        best: List[int] = []
        # Enumerate valid count i taken from nums1
        for i in range(max(0, k - n), min(k, m) + 1):
            cand1 = max_subsequence(nums1, i)
            cand2 = max_subsequence(nums2, k - i)
            merged = merge(cand1, cand2)
            if merged > best:
                best = merged
                
        return best
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<int> maxNumber(std::vector<int>& nums1, std::vector<int>& nums2, int k) {
        int m = nums1.size(), n = nums2.size();
        std::vector<int> best;

        for (int i = std::max(0, k - n); i <= std::min(k, m); ++i) {
            std::vector<int> seq1 = maxSubsequence(nums1, i);
            std::vector<int> seq2 = maxSubsequence(nums2, k - i);
            std::vector<int> candidate = merge(seq1, seq2);
            if (candidate > best) {
                best = candidate;
            }
        }
        return best;
    }

private:
    std::vector<int> maxSubsequence(const std::vector<int>& nums, int count) {
        if (count == 0) return {};
        int drop = nums.size() - count;
        std::vector<int> stack;
        for (int num : nums) {
            while (drop > 0 && !stack.empty() && stack.back() < num) {
                stack.pop_back();
                drop--;
            }
            stack.push_back(num);
        }
        stack.resize(count);
        return stack;
    }

    bool compareSuffix(const std::vector<int>& seq1, int i,
                       const std::vector<int>& seq2, int j) {
        int n1 = seq1.size(), n2 = seq2.size();
        while (i < n1 && j < n2) {
            if (seq1[i] != seq2[j]) return seq1[i] > seq2[j];
            i++;
            j++;
        }
        return (n1 - i) > (n2 - j);
    }

    std::vector<int> merge(const std::vector<int>& seq1, const std::vector<int>& seq2) {
        std::vector<int> res;
        int i = 0, j = 0;
        int n1 = seq1.size(), n2 = seq2.size();
        while (i < n1 || j < n2) {
            if (compareSuffix(seq1, i, seq2, j)) {
                res.push_back(seq1[i++]);
            } else {
                res.push_back(seq2[j++]);
            }
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int m = nums1.length, n = nums2.length;
        int[] best = new int[k];
        boolean initialized = false;

        for (int i = Math.max(0, k - n); i <= Math.min(k, m); i++) {
            int[] seq1 = maxSubsequence(nums1, i);
            int[] seq2 = maxSubsequence(nums2, k - i);
            int[] candidate = merge(seq1, seq2, k);

            if (!initialized || isGreater(candidate, 0, best, 0)) {
                best = candidate;
                initialized = true;
            }
        }
        return best;
    }

    private int[] maxSubsequence(int[] nums, int count) {
        int[] stack = new int[count];
        int top = -1;
        int drop = nums.length - count;

        for (int num : nums) {
            while (top >= 0 && stack[top] < num && drop > 0) {
                top--;
                drop--;
            }
            if (top + 1 < count) {
                stack[++top] = num;
            } else {
                drop--;
            }
        }
        return stack;
    }

    private boolean isGreater(int[] seq1, int i, int[] seq2, int j) {
        while (i < seq1.length && j < seq2.length) {
            if (seq1[i] != seq2[j]) {
                return seq1[i] > seq2[j];
            }
            i++;
            j++;
        }
        return (seq1.length - i) > (seq2.length - j);
    }

    private int[] merge(int[] seq1, int[] seq2, int k) {
        int[] res = new int[k];
        int i = 0, j = 0;
        for (int r = 0; r < k; r++) {
            if (isGreater(seq1, i, seq2, j)) {
                res[r] = seq1[i++];
            } else {
                res[r] = seq2[j++];
            }
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(k \cdot (m + n)^2)$ in the worst case (or $O(k \cdot (m + n))$ on typical data). For each partition $i \in [0, k]$, generating the monotonic subsequence takes $O(m + n)$, and merging takes $O((m + n)^2)$ due to lexicographical suffix comparison on equal digits. With $m, n \le 500$, $(m + n) \le 1000$, this executes well within the 1-second limit.
- **Space Complexity:** $O(k + m + n)$ to hold the subsequences and merged candidates.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Subproblem Decomposition: Divide into Monotonic Stack (Subsequence selection) + Two-Pointer Lexicographical Merge.
- **Trap:** Greedily advancing either pointer when `seq1[i] == seq2[j]`. If both numbers are `6`, picking arbitrary `6` will lead to suboptimal subsequent choices (e.g. `[6, 7]` vs `[6, 0]`). Lookahead suffix comparison is strictly mandatory.