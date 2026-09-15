---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 330: Patching Array"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - math
  - amazon
  - google
---

# LeetCode 330: Patching Array

**Target Companies:** Google, Amazon, Microsoft, Meta  
**Difficulty:** Hard  
**Topic:** Greedy / Range Coverage / Math  

---

### Problem Statement

Given a sorted integer array `nums` and an integer `n`, add/patch elements to the array such that any number in the range `[1, n]` inclusive can be formed by the sum of some elements in the array.

Return the **minimum number of patches** required.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — sorted array of positive integers ($1 \le \text{nums.length} \le 1000$).
  - `n`: `int` ($1 \le n \le 2^{31} - 1$).
- **Output:**
  - `int` — minimum number of patches required.
- **Constraints:**
  - $1 \le \text{nums.length} \le 1000$
  - $1 \le \text{nums}[i] \le 10^4$
  - `nums` is sorted in ascending order.
  - $1 \le n \le 2^{31} - 1$

---

### Key Idea & Intuition

Rather than modeling this as an exponential subset-sum dynamic programming problem, recognize that it is a **continuous prefix range coverage problem**.

#### The Continuous Coverage Invariant:
Define `miss` as the **smallest integer in $[1, n]$ that cannot yet be formed**.
Initially, before using any numbers, $\text{miss} = 1$, which means we can form all integers in $[1, 0]$ (the empty set).
Suppose we can currently form every integer in the continuous range $[1, \text{miss} - 1]$.
Now, what happens when we consider the next available number?

- **Case 1: The next number `nums[i] <= miss`:**
  - Since we can already form all numbers in $[1, \text{miss} - 1]$, adding `nums[i]` allows us to form:
    $$[1 + \text{nums}[i], \text{miss} - 1 + \text{nums}[i]]$$
  - Because $\text{nums}[i] \le \text{miss}$, the interval $[1, \text{miss} - 1]$ and the interval $[1 + \text{nums}[i], \text{miss} + \text{nums}[i] - 1]$ **overlap or touch continuously** without any gaps!
  - Therefore, our coverage expands to:
    $$[1, \text{miss} + \text{nums}[i] - 1]$$
  - The new smallest unformed integer becomes $\text{miss} + \text{nums}[i]$. We advance $i \mathrel{+}= 1$.

- **Case 2: The next number `nums[i] > miss` (or array exhausted):**
  - None of the remaining numbers in `nums` can form `miss` (since all future elements are $\ge \text{nums}[i] > \text{miss}$).
  - We **must patch** a new number $x$.
  - To avoid a gap at `miss`, we must choose $x \le \text{miss}$.
  - To **maximally extend** our future coverage greedily, the optimal choice is to pick the largest valid integer:
    $$\text{Patch } x = \text{miss}$$
  - Adding `miss` instantly doubles our coverage to $[1, 2 \cdot \text{miss} - 1]$!
  - We increment `patches += 1` and update $\text{miss} \mathrel{*}= 2$.

We repeat this process until $\text{miss} > n$.

---

### Solution Approach (Step-by-Step)

1. Initialize `miss = 1` (using a 64-bit integer to prevent overflow), `patches = 0`, and `i = 0`.
2. While `miss <= n`:
   - If `i < len(nums)` and `nums[i] <= miss`:
     - `miss += nums[i]`
     - `i += 1`
   - Else:
     - Patch `miss`:
     - `miss += miss`
     - `patches += 1`
3. Return `patches`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 5, 10]`, `n = 20`:

```
Initial: miss = 1, patches = 0, i = 0

Step 1:
  nums[0] = 1 <= miss(1)
  Coverage expands from [1, 0] to [1, 1]
  miss = 1 + 1 = 2, i = 1

Step 2:
  nums[1] = 5 > miss(2)  -> GAP! miss=2 cannot be formed.
  Patch 2:
  Coverage expands to [1, 2 + 2 - 1] = [1, 3]
  miss = 2 + 2 = 4, patches = 1

Step 3:
  nums[1] = 5 > miss(4)  -> GAP! miss=4 cannot be formed.
  Patch 4:
  Coverage expands to [1, 4 + 4 - 1] = [1, 7]
  miss = 4 + 4 = 8, patches = 2

Step 4:
  nums[1] = 5 <= miss(8)
  Coverage expands to [1, 7 + 5] = [1, 12]
  miss = 8 + 5 = 13, i = 2

Step 5:
  nums[2] = 10 <= miss(13)
  Coverage expands to [1, 12 + 10] = [1, 22]
  miss = 13 + 10 = 23, i = 3

miss (23) > n (20) -> Loop terminates!

Total patches added = 2 (patched: 2, 4).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [1, 3]`, `n = 6`
- **Tracing:**
  - $i=0$: `nums[0] = 1 <= 1` $\rightarrow$ miss = 2
  - $i=1$: `nums[1] = 3 > 2` $\rightarrow$ patch 2 $\rightarrow$ miss = 4, patches = 1
  - $i=1$: `nums[1] = 3 <= 4` $\rightarrow$ miss = 7 > 6
- **Output:** `1`

#### Example 2:
- **Input:** `nums = [1, 5, 10]`, `n = 20`
- **Output:** `2`

#### Example 3:
- **Input:** `nums = [1, 2, 2]`, `n = 5`
- **Tracing:**
  - $1 \le 1 \rightarrow \text{miss} = 2$
  - $2 \le 2 \rightarrow \text{miss} = 4$
  - $2 \le 4 \rightarrow \text{miss} = 6 > 5$
  - Patches = 0
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        miss = 1  # Smallest number in [1, n] that cannot be formed
        patches = 0
        i = 0
        m = len(nums)
        
        while miss <= n:
            # If current array element can bridge the gap
            if i < m and nums[i] <= miss:
                miss += nums[i]
                i += 1
            else:
                # Greedily patch `miss` to double our coverage
                miss += miss
                patches += 1
                
        return patches
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int minPatches(const std::vector<int>& nums, int n) {
        long long miss = 1; // 64-bit integer to prevent overflow
        int patches = 0;
        int i = 0;
        int m = static_cast<int>(nums.size());
        
        while (miss <= n) {
            if (i < m && nums[i] <= miss) {
                miss += nums[i];
                i++;
            } else {
                miss += miss;
                patches++;
            }
        }
        
        return patches;
    }
};
```

#### Java 17
```java
class Solution {
    public int minPatches(int[] nums, int n) {
        long miss = 1; // Use long to prevent integer overflow when miss approaches n
        int patches = 0;
        int i = 0;
        int m = nums.length;
        
        while (miss <= n) {
            if (i < m && nums[i] <= miss) {
                miss += nums[i];
                i++;
            } else {
                miss += miss;
                patches++;
            }
        }
        
        return patches;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(m + \log n)$ where $m = \text{nums.length}$
  - In each step, we either advance pointer $i$ in `nums` ($m$ times max) or double `miss` ($\log_2 n$ times max).
  - Since $m \le 1000$ and $\log_2(2^{31}) \approx 31$, the while loop runs at most $1031$ iterations, executing in microseconds.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only scalar registers (`miss`, `patches`, `i`) are utilized.

---

### Takeaway Pattern & Interview Traps

- **Exponential Doubling via Greedy Patching:** When you patch `miss`, you double the reach from $[1, \text{miss} - 1]$ to $[1, 2 \cdot \text{miss} - 1]$. This is why greedy patch algorithms achieve logarithmic guarantees $\mathcal{O}(\log n)$.
- **Integer Overflow Trap:** Since $n$ can be $2^{31} - 1$, when `miss` approaches $n$, doubling `miss += miss` will overflow signed 32-bit integers. **Always declare `miss` as 64-bit integer** (`long long` in C++, `long` in Java).