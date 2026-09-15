---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3191: Minimum Operations to Make Binary Array Elements Equal to One I"
tags:
  - leetcode
  - coding
  - sliding-window
  - greedy
  - bit-manipulation
  - array
  - amazon
  - google
---

# LeetCode 3191: Minimum Operations to Make Binary Array Elements Equal to One I

**Target Companies:** Amazon, Google, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** Sliding Window / Greedy / Bit Manipulation  

---

### Problem Statement

You are given a binary array `nums` consisting of $0$s and $1$s.

You can do the following operation on the array any number of times:
- Choose any **3 consecutive** elements from the array and **flip** all of them.

Flipping an element means changing a $0$ to $1$ and a $1$ to $0$.

Return the **minimum number of operations** required to make all elements in `nums` equal to $1$. If it is impossible, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` — a binary array containing only $0$s and $1$s.
- **Output:**
  - `int` — the minimum operations required to turn all elements into $1$, or `-1` if impossible.
- **Constraints:**
  - $3 \le \text{nums.length} \le 10^5$
  - $0 \le \text{nums}[i] \le 1$

---

### Key Idea & Intuition

#### The Forced Choice Invariant:
Consider the array scanning strictly from left to right at index $i = 0$:
- If `nums[0] == 1`, we must not flip the window starting at index $0$, because flipping would turn `nums[0]` into $0$. No operation starting at $j \ge 1$ can ever reach index $0$.
- If `nums[0] == 0`, we **must** flip the window $[0, 1, 2]$. This is our **only** opportunity to fix `nums[0]`, because all subsequent window operations will start at index $\ge 1$ and will never touch index $0$ again.

By mathematical induction, once indices $0, 1, \dots, i - 1$ are all $1$s:
- If `nums[i] == 0`, we are forced to apply a flip at window $[i, i + 1, i + 2]$.
- If `nums[i] == 1`, we cannot flip at $i$.

We repeat this deterministic greedy choice for every index $i$ from $0$ up to $n - 3$.
When we reach index $n - 2$, no window of length 3 can start at $n - 2$ or $n - 1$.
Therefore:
- If `nums[n - 2] == 1` and `nums[n - 1] == 1`, all elements are successfully $1$, and the total count of flips is minimal and unique.
- If either `nums[n - 2] == 0` or `nums[n - 1] == 0`, it is mathematically impossible to make the array all $1$s, so we return `-1`.

---

### Solution Approach (Step-by-Step)

1. Initialize `operations = 0` and `n = len(nums)`.
2. Iterate `i` from $0$ to $n - 3$:
   - If `nums[i] == 0`:
     - Flip `nums[i + 1] ^= 1`
     - Flip `nums[i + 2] ^= 1`
     - Increment `operations += 1`
3. After the loop, inspect the final two elements:
   - If `nums[n - 2] == 0` or `nums[n - 1] == 0`, return `-1`.
4. Otherwise, return `operations`.

---

### Visual Algorithm Walkthrough

For `nums = [0, 1, 1, 1, 0, 0]`:

```
Initial:      [0, 1, 1, 1, 0, 0]

i = 0: nums[0] == 0 -> Must flip [0, 1, 2]!
  Flip: nums[0]=1, nums[1]=1^1=0, nums[2]=1^1=0
  Array:      [1, 0, 0, 1, 0, 0]
  operations = 1

i = 1: nums[1] == 0 -> Must flip [1, 2, 3]!
  Flip: nums[1]=1, nums[2]=0^1=1, nums[3]=1^1=0
  Array:      [1, 1, 1, 0, 0, 0]
  operations = 2

i = 2: nums[2] == 1 -> No flip needed.
  Array:      [1, 1, 1, 0, 0, 0]

i = 3: nums[3] == 0 -> Must flip [3, 4, 5]!
  Flip: nums[3]=1, nums[4]=0^1=1, nums[5]=0^1=1
  Array:      [1, 1, 1, 1, 1, 1]
  operations = 3

Loop ends (i reaches n - 3 = 3).
Inspect remaining: nums[4] == 1, nums[5] == 1.
All elements are 1!
Result = 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [0, 1, 1, 1, 0, 0]`
- **Output:** `3`

#### Example 2 (Impossible):
- **Input:** `nums = [0, 1, 1, 1]`
- **Tracing:**
  - $i = 0$: `nums[0] == 0` -> flip `[0, 1, 2]` -> array becomes `[1, 0, 0, 1]`, ops = 1.
  - $i = 1$: `nums[1] == 0` -> flip `[1, 2, 3]` -> array becomes `[1, 1, 1, 0]`, ops = 2.
  - Check last two elements: `nums[2] = 1`, but `nums[3] = 0`.
  - Cannot flip further because size requires 3 elements.
- **Output:** `-1`

#### Example 3 (Already All 1s):
- **Input:** `nums = [1, 1, 1]`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        # Greedy sweep up to index n - 3
        for i in range(n - 2):
            if nums[i] == 0:
                nums[i + 1] ^= 1
                nums[i + 2] ^= 1
                operations += 1
                
        # Validate trailing elements that could not be flipped
        if nums[n - 2] == 0 or nums[n - 1] == 0:
            return -1
            
        return operations
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int minOperations(std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        int operations = 0;
        
        for (int i = 0; i <= n - 3; ++i) {
            if (nums[i] == 0) {
                nums[i + 1] ^= 1;
                nums[i + 2] ^= 1;
                ++operations;
            }
        }
        
        if (nums[n - 2] == 0 || nums[n - 1] == 0) {
            return -1;
        }
        
        return operations;
    }
};
```

#### Java 17
```java
class Solution {
    public int minOperations(int[] nums) {
        int n = nums.length;
        int operations = 0;
        
        for (int i = 0; i <= n - 3; i++) {
            if (nums[i] == 0) {
                nums[i + 1] ^= 1;
                nums[i + 2] ^= 1;
                operations++;
            }
        }
        
        if (nums[n - 2] == 0 || nums[n - 1] == 0) {
            return -1;
        }
        
        return operations;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - We traverse the array once from index $0$ to $n - 3$. At each index, we perform $\mathcal{O}(1)$ bitwise XOR operations on the next two elements.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - We modify the input array in-place without allocating any dynamic data structures.

---

### Takeaway Pattern & Interview Traps

- **The Forced Choice Principle:** In any problem where operations affect a contiguous range $[i, i + k - 1]$ and cannot affect earlier elements, scanning unidirectionally turns an exponential decision tree into a deterministic greedy algorithm.
- **Commutativity of Flips:** In binary XOR systems, flipping window A then window B is identical to flipping window B then window A ($A \oplus B = B \oplus A$). Moreover, flipping any window twice is a no-op ($A \oplus A = 0$). Hence, each window is flipped at most once.
- **Generalization to Arbitrary $k$ (LeetCode 995):** When $k$ is large ($k > 3$), flipping in-place takes $\mathcal{O}(k)$ per operation yielding $\mathcal{O}(n \cdot k)$. To achieve $\mathcal{O}(n)$ for arbitrary $k$, maintain a sliding window difference array or deque storing the active flip states.