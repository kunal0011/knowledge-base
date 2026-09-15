---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 334: Increasing Triplet Subsequence"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - patience-sorting
  - amazon
  - google
---

# LeetCode 334: Increasing Triplet Subsequence

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Array / Longest Increasing Subsequence ($k = 3$)  

---

### Problem Statement

Given an integer array `nums`, return `true` if there exists a triple of indices $(i, j, k)$ such that $i < j < k$ and $\text{nums}[i] < \text{nums}[j] < \text{nums}[k]$. If no such indices exist, return `false`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 5 \times 10^5$).
- **Output:**
  - `bool` — `true` if an increasing subsequence of length 3 exists, else `false`.
- **Constraints:**
  - $1 \le \text{nums.length} \le 5 \times 10^5$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$
  - Follow up: Could you implement a solution that runs in $\mathcal{O}(n)$ time complexity and $\mathcal{O}(1)$ space complexity?

---

### Key Idea & Intuition

This problem is equivalent to determining whether the Longest Increasing Subsequence (LIS) has length $\ge 3$.

#### The Greedy Tracking Invariant ($\mathcal{O}(n)$ time, $\mathcal{O}(1)$ space):
Maintain two variables:
- `first`: the smallest value seen so far.
- `second`: the smallest value seen so far that has some valid number before it that is strictly smaller than `second`.

Initially: `first = infinity`, `second = infinity`.

As we iterate through each number $x$ in `nums`:
1. **If $x \le \text{first}$:**
   - Update `first = x`.
2. **Else if $x \le \text{second}$:**
   - Update `second = x` (since $x > \text{first}$, $x$ is a valid second element, and smaller is better).
3. **Else ($x > \text{second}$):**
   - We have found a number strictly greater than `second`!
   - Since `second` is only set when there is an element preceding it that is smaller than it, an increasing triplet $(i, j, k)$ is guaranteed to exist $\rightarrow$ return `true` immediately.

#### The Crucial Interview Question:
*"What if `first` is updated AFTER `second` was set? Doesn't `first` now appear after `second`?"*
**The Invariant Holds:**
- Even if `first` is later updated to an even smaller element that appears *after* `second`, `second` was established using an older value of `first` that appeared *before* `second`.
- Any subsequent number $x > \text{second}$ is automatically greater than that older `first` as well!
- Therefore, the existence of a valid increasing triplet is guaranteed, even if the current `first` index is after `second`.

---

### Solution Approach (Step-by-Step)

1. Initialize `first = float('inf')` and `second = float('inf')`.
2. Loop through each number $x$ in `nums`:
   - If $x \le first$:
     - $first = x$
   - Else if $x \le second$:
     - $second = x$
   - Else:
     - Return `True` (found $x > second > first$).
3. If loop finishes without returning `True`, return `False`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 1, 5, 0, 4, 6]`:

```
Initial: first = inf, second = inf

x = 2:
  2 <= first -> first = 2
  State: first = 2, second = inf

x = 1:
  1 <= first -> first = 1 (A smaller start found!)
  State: first = 1, second = inf

x = 5:
  5 > first(1) and 5 <= second(inf) -> second = 5
  State: first = 1, second = 5 (Pair (1, 5) established)

x = 0:
  0 <= first(1) -> first = 0
  State: first = 0, second = 5
  *Notice: second=5 still remembers there was an element < 5 before it (namely 1)!*

x = 4:
  4 > first(0) and 4 <= second(5) -> second = 4
  State: first = 0, second = 4 (Pair (0, 4) established)

x = 6:
  6 > second(4) -> TRIPLET FOUND! (0 < 4 < 6)
  Return True.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 2, 3, 4, 5]`
- **Tracing:**
  - $x=1 \rightarrow first = 1$
  - $x=2 \rightarrow second = 2$
  - $x=3 > second \rightarrow$ returns `true`
- **Output:** `true`

#### Example 2:
- **Input:** `nums = [5, 4, 3, 2, 1]`
- **Tracing:** Array is strictly decreasing. `second` is never set.
- **Output:** `false`

#### Example 3:
- **Input:** `nums = [2, 1, 5, 0, 4, 6]`
- **Output:** `true`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = float('inf')
        second = float('inf')
        
        for x in nums:
            if x <= first:
                first = x
            elif x <= second:
                second = x
            else:
                # Found x > second > first
                return True
                
        return False
```

#### C++17
```cpp
#include <vector>
#include <climits>

class Solution {
public:
    bool increasingTriplet(const std::vector<int>& nums) {
        int first = INT_MAX;
        int second = INT_MAX;
        
        for (int x : nums) {
            if (x <= first) {
                first = x;
            } else if (x <= second) {
                second = x;
            } else {
                return true;
            }
        }
        
        return false;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean increasingTriplet(int[] nums) {
        int first = Integer.MAX_VALUE;
        int second = Integer.MAX_VALUE;
        
        for (int x : nums) {
            if (x <= first) {
                first = x;
            } else if (x <= second) {
                second = x;
            } else {
                return true;
            }
        }
        
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear scan through the array of length $n$.
  - At each step, at most two scalar comparisons are performed.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only two primitive integer variables `first` and `second` are maintained.

---

### Takeaway Pattern & Interview Traps

- **Patience Sorting / LIS with $K = 3$:** This greedy technique is a specialization of the Patience Sorting algorithm (used in $\mathcal{O}(n \log n)$ LIS), maintaining the minimum tail of active subsequences of length 1 (`first`) and length 2 (`second`).
- **Use `<=` instead of `<`:** If `nums = [1, 1, 1, 1]`, using `<` instead of `<=` would mistakenly assign duplicate values to `first` and `second`, falsely reporting `true`.