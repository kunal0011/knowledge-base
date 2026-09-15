---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 45: Jump Game II"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - greedy
  - bfs
  - amazon
  - meta
  - google
---

# LeetCode 45: Jump Game II

**Target Companies:** Amazon, Meta, Google, Microsoft, Apple, Uber  
**Difficulty:** Medium  
**Topic:** Greedy / Breadth-First Search Window / Dynamic Programming  

---

### Problem Statement

You are given a **0-indexed** array of integers `nums` of length `n`. You are initially positioned at `nums[0]`.

Each element `nums[i]` represents the maximum length of a forward jump from index `i`. In other words, if you are at `nums[i]`, you can jump to any `nums[i + j]` where:

- $0 \le j \le \text{nums}[i]$ and
- $i + j < n$

Return the **minimum number of jumps** to reach `nums[n - 1]`. The test cases are generated such that you can always reach `nums[n - 1]`.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` — Array of maximum jump lengths from each index.
- **Output:** `int` — Minimum number of jumps to reach index $n - 1$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $0 \le \text{nums}[i] \le 1000$
  - It is guaranteed that you can reach `nums[n - 1]`.

---

### Key Idea & Intuition

1. **DP State & Quadratic Baseline ($\mathcal{O}(N^2)$):**
   - Let $\text{dp}[i]$ be the minimum jumps to reach index $i$.
   - $\text{dp}[0] = 0$.
   - For each index $j < i$, if $j + \text{nums}[j] \ge i$, then:
     $$\text{dp}[i] = \min(\text{dp}[i], \text{dp}[j] + 1)$$
   - This checks all previous indices, yielding an $\mathcal{O}(N^2)$ time complexity.

2. **Greedy BFS Layering ($\mathcal{O}(N)$ Optimal):**
   - Notice that jumps form contiguous **BFS distance layers**:
     - Layer 0: $[0, 0]$ (0 jumps).
     - Layer 1: $[1, \text{nums}[0]]$ (1 jump).
     - Layer 2: Range reachable by any jump from Layer 1 (2 jumps).
   - We do not need to store explicit queue levels. Instead, maintain two boundary pointers:
     - `curr_end`: The boundary of the current jump layer.
     - `farthest`: The maximum reachable index from any position explored so far in the current layer.
   - When our scan index $i$ reaches `curr_end`, we are forced to commit to a new jump:
     $$\text{jumps} += 1$$
     $$\text{curr\_end} = \text{farthest}$$
   - Loop runs up to $n - 2$ because once `curr_end >= n - 1`, the destination is already within reach.

---

### Solution Approach (Step-by-Step)

1. **Base Case:**
   - If `n <= 1`, return `0` (already at the last index).
2. **Initialize State:**
   - `jumps = 0`
   - `curr_end = 0` (furthest index reachable with current count of jumps)
   - `farthest = 0` (furthest index reachable with one additional jump)
3. **Iterate up to $n - 2$:**
   - For $i$ from $0$ to $n - 2$:
     - `farthest = max(farthest, i + nums[i])`.
     - If $i == \text{curr\_end}$:
       - Increment `jumps += 1`.
       - Update `curr_end = farthest`.
       - Early exit: If `curr_end >= n - 1`, break.
4. **Return:**
   - Return `jumps`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 3, 1, 1, 4]`, $n = 5$:

```
Initial: jumps = 0, curr_end = 0, farthest = 0

i = 0 (val = 2):
  farthest = max(0, 0 + 2) = 2
  i == curr_end (0 == 0):
    jumps += 1 (jumps = 1)
    curr_end = farthest = 2
    (Layer 1 covers indices [1, 2])

i = 1 (val = 3):
  farthest = max(2, 1 + 3) = 4
  i (1) != curr_end (2) -> continue

i = 2 (val = 1):
  farthest = max(4, 2 + 1) = 4
  i == curr_end (2 == 2):
    jumps += 1 (jumps = 2)
    curr_end = farthest = 4
    (curr_end >= 4 == n - 1 -> Destination reached!)

Loop terminates at i = n - 2 = 3.
Minimum jumps = 2 (Path: 0 -> 1 -> 4).
```

---

### Solved Examples with Multiple Inputs

| Case | `nums` | BFS Intervals (Layers) | Result | Explanation |
|---|---|---|---|---|
| **Standard** | `[2, 3, 1, 1, 4]` | $0 \to [1, 2] \to [3, 4]$ | `2` | Jump $0 \to 1 \to 4$ |
| **Single Jump Destination** | `[2, 3, 0, 1, 4]` | $0 \to [1, 2] \to [3, 4]$ | `2` | From index 1 jump 3 lands on 4 |
| **Direct Jump** | `[10, 0, 0]` | $0 \to [1, 2]$ | `1` | First jump reaches end directly |
| **Single Element** | `[0]` | Already at target | `0` | 0 jumps needed |
| **Chain of 1s** | `[1, 1, 1, 1]` | Must jump 1 step at a time | `3` | $0 \to 1 \to 2 \to 3$ |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed — Greedy BFS $\mathcal{O}(N)$)
```python
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
            
        jumps = 0
        curr_end = 0
        farthest = 0
        
        # Iterate up to n - 2
        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == curr_end:
                jumps += 1
                curr_end = farthest
                if curr_end >= n - 1:
                    break
                    
        return jumps
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int jump(std::vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0;

        int jumps = 0;
        int curr_end = 0;
        int farthest = 0;

        for (int i = 0; i < n - 1; ++i) {
            farthest = std::max(farthest, i + nums[i]);

            if (i == curr_end) {
                jumps++;
                curr_end = farthest;
                if (curr_end >= n - 1) {
                    break;
                }
            }
        }

        return jumps;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int jump(int[] nums) {
        int n = nums.length;
        if (n <= 1) return 0;

        int jumps = 0;
        int currEnd = 0;
        int farthest = 0;

        for (int i = 0; i < n - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);

            if (i == currEnd) {
                jumps++;
                currEnd = farthest;
                if (currEnd >= n - 1) {
                    break;
                }
            }
        }

        return jumps;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$  
  The array is scanned in a single linear pass from index $0$ to $N - 2$. In each iteration, a constant number of max comparisons and pointer updates are performed.
- **Space Complexity:** $\mathcal{O}(1)$  
  Only three scalar integers (`jumps`, `curr_end`, `farthest`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Loop Bound at $N - 2$:**
   - Why do we iterate only to $N - 2$ (`range(n - 1)`) instead of $N - 1$?
   - If we iterated all the way to $N - 1$, when $i = N - 1 == \text{curr\_end}$, the algorithm would unnecessarily trigger an extra jump even though we have already arrived at the target!
2. **Jump Game I vs. Jump Game II:**
   - **Jump Game I (LeetCode 55):** Decision problem ("Can you reach the last index?"). Requires checking if $i > \text{farthest}$.
   - **Jump Game II (LeetCode 45):** Optimization problem ("What is the minimum number of jumps?"). Guaranteed reachability allows clean BFS interval progression.