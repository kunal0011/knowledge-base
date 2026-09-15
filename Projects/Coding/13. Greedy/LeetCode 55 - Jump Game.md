---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 55: Jump Game"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 55: Jump Game

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Array / Reachability Frontier  

---

### Problem Statement

You are given an integer array `nums`. You are initially positioned at the array's **first index**, and each element in the array represents your maximum jump length at that position.

Return `true` if you can reach the last index, or `false` otherwise.

---

### Input & Output Formats & Constraints

- **Input:**
  - `nums`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{nums.length} \le 10^4$).
- **Output:**
  - `bool` — `true` if the last index $n - 1$ is reachable, else `false`.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^4$
  - $0 \le \text{nums}[i] \le 10^5$

---

### Key Idea & Intuition

Rather than simulating individual jumps or testing all path branches via backtracking/DP ($\mathcal{O}(2^n)$ or $\mathcal{O}(n^2)$), observe the continuous reachability property:

#### Approach 1: Forward Greedy Frontier Expansion
Maintain a variable `max_reach` denoting the **furthest index reachable so far**:
- Start at $i = 0$ with `max_reach = 0`.
- As we iterate through each index $i$:
  - If $i > \text{max\_reach}$:
    - We have reached an index that was unreachable from any previous position!
    - We are permanently stuck $\rightarrow$ return `False`.
  - Otherwise, index $i$ is reachable. From index $i$, we can jump up to $i + \text{nums}[i]$.
  - Update the frontier:
    $$\text{max\_reach} = \max(\text{max\_reach}, i + \text{nums}[i])$$
  - If $\text{max\_reach} \ge n - 1$:
    - The last index is already reachable $\rightarrow$ return `True` immediately!

#### Approach 2: Backward Target Shift
Start from the end and work backwards:
- Initialize `target = n - 1`.
- For $i$ from $n - 2$ down to $0$:
  - If $i + \text{nums}[i] \ge target$:
    - Being at index $i$ is sufficient to jump to (or past) `target`.
    - We can shift the goal closer: `target = i`.
- If `target == 0` at the end, the start can reach the destination $\rightarrow$ return `True`.

---

### Solution Approach (Step-by-Step: Forward Greedy)

1. Initialize `max_reach = 0` and $n = \text{len}(nums)$.
2. Iterate $i$ from $0$ to $n - 1$:
   - If $i > max\_reach$:
     - Return `False` (gap detected, cannot step onto $i$).
   - `max_reach = max(max_reach, i + nums[i])`
   - If `max_reach >= n - 1`:
     - Return `True` (destination reachable).
3. Return `True`.

---

### Visual Algorithm Walkthrough

For `nums = [2, 3, 1, 1, 4]`:

```
i = 0 (val = 2):
  i <= max_reach (0 <= 0): OK
  max_reach = max(0, 0 + 2) = 2

i = 1 (val = 3):
  i <= max_reach (1 <= 2): OK
  max_reach = max(2, 1 + 3) = 4
  max_reach (4) >= n - 1 (4) -> REACHED!
  Return True immediately.
```

For `nums = [3, 2, 1, 0, 4]`:

```
i = 0 (val = 3):
  max_reach = max(0, 0 + 3) = 3

i = 1 (val = 2):
  max_reach = max(3, 1 + 2) = 3

i = 2 (val = 1):
  max_reach = max(3, 2 + 1) = 3

i = 3 (val = 0):
  max_reach = max(3, 3 + 0) = 3

i = 4 (val = 4):
  i > max_reach (4 > 3) -> STUCK!
  Cannot reach index 4 from any previous position.
  Return False.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `nums = [2, 3, 1, 1, 4]`
- **Output:** `true`

#### Example 2:
- **Input:** `nums = [3, 2, 1, 0, 4]`
- **Output:** `false`

#### Example 3 (Single Element):
- **Input:** `nums = [0]`
- **Tracing:** Already at the final index (index 0). $0 \ge 0 \rightarrow \text{true}$.
- **Output:** `true`

#### Example 4 (Large Jump at Start):
- **Input:** `nums = [5, 0, 0, 0, 0]`
- **Tracing:** Index 0 can jump to index 5 $\ge 4$.
- **Output:** `true`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0
        n = len(nums)
        
        for i, jump in enumerate(nums):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + jump)
            if max_reach >= n - 1:
                return True
                
        return True
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    bool canJump(const std::vector<int>& nums) {
        int max_reach = 0;
        int n = static_cast<int>(nums.size());
        
        for (int i = 0; i < n; ++i) {
            if (i > max_reach) {
                return false;
            }
            max_reach = std::max(max_reach, i + nums[i]);
            if (max_reach >= n - 1) {
                return true;
            }
        }
        
        return true;
    }
};
```

#### Java 17
```java
class Solution {
    public boolean canJump(int[] nums) {
        int maxReach = 0;
        int n = nums.length;
        
        for (int i = 0; i < n; i++) {
            if (i > maxReach) {
                return false;
            }
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= n - 1) {
                return true;
            }
        }
        
        return true;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single linear pass through the array. Early returns when `max_reach >= n - 1` or when reaching a dead end `i > max_reach`.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Uses only a single integer tracking `max_reach`.

---

### Takeaway Pattern & Interview Traps

- **Forward Frontier vs. Backward Target:**
  - Forward: Tracks the maximum reachable index. Fails if $i > max\_reach$.
  - Backward: Shifts `target` to $i$ whenever $i + nums[i] \ge target$. Succeeds if `target == 0`.
  Both are clean $\mathcal{O}(n)$ time and $\mathcal{O}(1)$ space.
- **Trap: Single Element `[0]`:** Notice that if $n = 1$, the loop checks index $0$: $0 \le 0$ and $0 \ge 0$, returning `true` immediately without getting stuck on 0.