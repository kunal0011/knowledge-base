---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 55: Jump Game"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - greedy
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 55: Jump Game

**Target Companies:** Amazon, Google, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / 1D Dynamic Programming / Array Reachability  

---

### Problem Statement

You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.

Return `true` if you can reach the **last index**, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of non-negative integers.
- **Output:** `bool` — `true` if last index is reachable, `false` otherwise.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $0 \le \text{nums}[i] \le 10^5$

---

### Key Idea & Intuition

1. **Reachability DP Formulation ($\mathcal{O}(N^2)$ Baseline):**
   - Let $\text{dp}[i]$ be a boolean indicating whether index $i$ is reachable from index $0$.
   - Base Case: $\text{dp}[0] = \text{true}$.
   - Transition:
     $$\text{dp}[i] = \exists j \in [0, i - 1] \text{ such that } \text{dp}[j] \land (j + \text{nums}[j] \ge i)$$
   - This formulation checks all prior indices, taking $\mathcal{O}(N^2)$ time.

2. **Greedy State Compression ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space — Optimal):**
   - Notice that if index $i$ is reachable, every index $k \le i + \text{nums}[i]$ is also reachable!
   - We do not need a boolean flag for every individual index. We only need to maintain a single scalar:
     $$\text{max\_reachable} = \max_{0 \le j \le i} (j + \text{nums}[j])$$
   - As we iterate through $i$ from $0$ to $n - 1$:
     - If $i > \text{max\_reachable}$, we have encountered an unbridgeable barrier (a series of zeros trapping the traversal) $\implies$ return `false`.
     - Update $\text{max\_reachable} = \max(\text{max\_reachable}, \ i + \text{nums}[i])$.
     - If $\text{max\_reachable} \ge n - 1$, the destination is already reachable $\implies$ return `true` immediately.

---

### Solution Approach (Step-by-Step)

1. **Initialize Reachability Bound:**
   - `max_reachable = 0`.
2. **Iterate Through Array:**
   - For index $i$ from $0$ to $n - 1$:
     - If $i > \text{max\_reachable}$:
       - Cannot proceed past current barrier $\implies$ return `False`.
     - Update `max_reachable = max(max_reachable, i + nums[i])`.
     - Early exit: If `max_reachable >= n - 1`, return `True`.
3. **Return:**
   - If the loop finishes, return `True` (for single-element arrays or fully covered bounds).

---

### Visual Algorithm Walkthrough

#### Example 1: `nums = [2, 3, 1, 1, 4]`
```
i = 0 (val = 2):
  max_reachable = max(0, 0 + 2) = 2

i = 1 (val = 3):
  1 <= max_reachable (2) -> valid
  max_reachable = max(2, 1 + 3) = 4
  max_reachable >= n - 1 (4 >= 4) -> Destination reached!
  Returns True!
```

#### Example 2: `nums = [3, 2, 1, 0, 4]`
```
i = 0 (val = 3):
  max_reachable = max(0, 0 + 3) = 3

i = 1 (val = 2):
  max_reachable = max(3, 1 + 2) = 3

i = 2 (val = 1):
  max_reachable = max(3, 2 + 1) = 3

i = 3 (val = 0):
  max_reachable = max(3, 3 + 0) = 3

i = 4 (val = 4):
  i (4) > max_reachable (3)!
  Trapped at index 3 with 0 jump length!
  Returns False!
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | `max_reachable` Evolution | Result | Explanation |
|---|---|---|---|---|
| **Reachable** | `[2, 3, 1, 1, 4]` | $0 \to 2 \to 4$ | `true` | Can jump directly from index 1 to 4 |
| **Trapped by Zero** | `[3, 2, 1, 0, 4]` | $0 \to 3 \to 3 \to 3 \to 3$ | `false` | Cannot jump past zero at index 3 |
| **Single Element** | `[0]` | `max_reachable = 0 >= 0` | `true` | Already at target index 0 |
| **Large Jump Start** | `[5, 0, 0, 0, 0]` | `max_reachable = 5 >= 4` | `true` | Reaches last index in 1 jump |
| **Zero at Start** | `[0, 2, 3]` | `i = 1 > max_reachable = 0` | `false` | Stranded at index 0 |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Greedy $\mathcal{O}(1)$ Space)
```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reachable = 0
        n = len(nums)
        
        for i, jump in enumerate(nums):
            if i > max_reachable:
                return False
            max_reachable = max(max_reachable, i + jump)
            if max_reachable >= n - 1:
                return True
                
        return True
```

#### 2. C++ (C++17 / STL — Greedy $\mathcal{O}(1)$ Space)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    bool canJump(std::vector<int>& nums) {
        int max_reachable = 0;
        int n = nums.size();

        for (int i = 0; i < n; ++i) {
            if (i > max_reachable) {
                return false;
            }
            max_reachable = std::max(max_reachable, i + nums[i]);
            if (max_reachable >= n - 1) {
                return true;
            }
        }

        return true;
    }
};
```

#### 3. Java (Modern, Typed — Greedy $\mathcal{O}(1)$ Space)
```java
class Solution {
    public boolean canJump(int[] nums) {
        int maxReachable = 0;
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            if (i > maxReachable) {
                return false;
            }
            maxReachable = Math.max(maxReachable, i + nums[i]);
            if (maxReachable >= n - 1) {
                return true;
            }
        }

        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  A single pass through the array. At each element, only $\mathcal{O}(1)$ comparisons and updates are performed. For $N = 10^4$, executes in under $2$ ms.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only a single integer accumulator (`max_reachable`) is stored.

---

### Takeaway Pattern & Interview Traps

1. **Jump Game I vs. Jump Game II:**
   - **Jump Game I (LC 55):** Checks **feasibility** (is last index reachable?). Solved by tracking global `max_reachable`.
   - **Jump Game II (LC 45):** Checks **optimality** (minimum jumps to reach last index). Solved by tracking BFS window boundaries `curr_end` and `farthest`.
2. **Single Element Edge Case:**
   - When `nums = [0]`, $n = 1$. The start index is already the destination index. The algorithm must return `true`.