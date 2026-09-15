---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 45: Jump Game II"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - bfs
  - amazon
  - google
---

# LeetCode 45: Jump Game II

**Target Companies:** Amazon (Signature Top 5), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Range Expansion / Implicit BFS  

---

### Problem Statement

You are given a 0-indexed array of integers `nums` of length $n$. You are initially positioned at `nums[0]`.

Each element `nums[i]` represents the maximum length of a forward jump from index $i$. In other words, if you are at `nums[i]`, you can jump to any `nums[i + j]` where:
- $0 \le j \le \text{nums}[i]$ and
- $i + j < n$

Return the **minimum number of jumps** to reach `nums[n - 1]`. The test cases are generated such that you can reach `nums[n - 1]`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 10^4$).
- **Output:**
  - `int` — minimum jumps required to reach the last index.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $0 \le \text{nums}[i] \le 1000$
  - It is guaranteed that you can reach `nums[n - 1]`.

---

### Key Idea & Intuition

While dynamic programming can compute minimum jumps in $\mathcal{O}(n^2)$, a **Greedy Frontier Expansion** (implicit Breadth-First Search) solves it in strictly linear $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ auxiliary space.

#### Implicit BFS Invariant:
Think of the problem in terms of BFS layers:
- Level 0: The starting point $[0, 0]$ (0 jumps).
- Level 1: All indices reachable in 1 jump from Level 0: $[1, \text{nums}[0]]$.
- Level 2: All indices reachable in 1 jump from any index in Level 1: $[\text{Level 1 end} + 1, \max_{i \in \text{Level 1}}(i + \text{nums}[i])]$.

At each step, we do not need to decide which exact index to land on. We only need to know **the maximum reach achievable from the current BFS level**:
- `cur_end`: The boundary of the current jump level.
- `farthest`: The furthest index we can reach with one more jump starting from any index $\le cur\_end$.
- As index $i$ progresses from $0$ to $n - 2$:
  - Update `farthest = max(farthest, i + nums[i])`.
  - When $i == cur\_end$:
    - We have exhausted all options at the current jump level.
    - We are forced to transition to the next level: `jumps += 1`, and update `cur_end = farthest`.
    - If `cur_end >= n - 1`, we can stop early!

#### Loop Boundary Caveat ($n - 1$ vs $n - 2$):
We stop our loop at $n - 2$. If we looped all the way to $n - 1$, when $i = n - 1 == cur\_end$, the algorithm would increment `jumps` unnecessarily despite already having arrived at the destination!

---

### Solution Approach (Step-by-Step)

1. If $n \le 1$, return $0$.
2. Initialize `jumps = 0`, `cur_end = 0`, and `farthest = 0`.
3. Loop $i$ from $0$ up to $n - 2$:
   - `farthest = max(farthest, i + nums[i])`
   - If $i == cur\_end$:
     - `jumps += 1`
     - `cur_end = farthest`
     - If `cur_end >= n - 1`: break early.
4. Return `jumps`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 3, 1, 1, 4]`:
$n = 5$, indices $0 \dots 4$. Target is index 4.

```
i = 0 (val = 2):
  farthest = max(0, 0 + 2) = 2
  i == cur_end (0):
    jumps = 1
    cur_end = 2
  Level 1 interval: indices [1, 2]

i = 1 (val = 3):
  farthest = max(2, 1 + 3) = 4
  1 != cur_end (2)

i = 2 (val = 1):
  farthest = max(4, 2 + 1) = 4
  i == cur_end (2):
    jumps = 2
    cur_end = farthest = 4
    cur_end >= 4 (Target reached!)
    break!

Result: jumps = 2 (Path: 0 -> 1 -> 4).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [2, 3, 1, 1, 4]`
- **Output:** `2`

#### Example 2:
- **Input:** `nums = [2, 3, 0, 1, 4]`
- **Output:** `2`

#### Example 3 (Single Element):
- **Input:** `nums = [0]`
- **Tracing:** Already at index 0. Zero jumps needed.
- **Output:** `0`

#### Example 4 (Large Jump from Start):
- **Input:** `nums = [10, 1, 1, 1, 1]`
- **Tracing:** At index 0, $0 + 10 \ge 4$, reached in 1 jump.
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
            
        jumps = 0
        cur_end = 0
        farthest = 0
        
        # Loop up to n - 2 to avoid triggering an extra jump at the final index
        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])
            
            # Reached the edge of the current jump's reach
            if i == cur_end:
                jumps += 1
                cur_end = farthest
                if cur_end >= n - 1:
                    break
                    
        return jumps
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int jump(const std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        if (n <= 1) return 0;
        
        int jumps = 0;
        int cur_end = 0;
        int farthest = 0;
        
        for (int i = 0; i < n - 1; ++i) {
            farthest = std::max(farthest, i + nums[i]);
            
            if (i == cur_end) {
                jumps++;
                cur_end = farthest;
                if (cur_end >= n - 1) {
                    break;
                }
            }
        }
        
        return jumps;
    }
};
```

#### Java 17
```java
class Solution {
    public int jump(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return 0;
        }
        
        int jumps = 0;
        int curEnd = 0;
        int farthest = 0;
        
        for (int i = 0; i < n - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            
            if (i == curEnd) {
                jumps++;
                curEnd = farthest;
                if (curEnd >= n - 1) {
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

- **Time Complexity:** $\mathcal{O}(n)$
  - A single pass through the array from index $0$ to $n - 2$.
  - At each index, constant time arithmetic and maximum comparisons are performed.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only scalar counters (`jumps`, `cur_end`, `farthest`) are stored in registers.

---

### Takeaway Pattern & Interview Traps

- **Implicit BFS over Explicit Queue:** In 1D array interval jumping, an explicit BFS queue is unnecessary. Two pointer integers `cur_end` and `farthest` represent the current BFS level boundary and the next BFS level boundary in $\mathcal{O}(1)$ space.
- **The $n - 1$ Boundary Trap:** Never iterate to $i = n - 1$. Reaching $n - 1$ means you have already arrived at the destination; advancing `cur_end` there falsely adds 1 extra jump.