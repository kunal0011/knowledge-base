---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 55: Jump Game"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 55: Jump Game

**LeetCode 55 (Jump Game)**, structured exactly as requested.

---

## LeetCode 55 — Jump Game

### Problem Statement

You are given an integer array `nums`.  
You are initially positioned at index `0`.

Each element `nums[i]` represents the **maximum jump length** you can make from index `i`.

Your goal is to determine whether you can reach the **last index** of the array.

**Return `true` if you can reach the last index, otherwise return `false`.**

---

### Key Observation (Most Important Insight)

You **do not need to try all jump combinations**.

Instead, the problem reduces to this single question:

> **At every index, is it still possible to reach or pass that index from previous jumps?**

If at any index `i`, your maximum reachable position is **less than `i`**, then:

* You are stuck
* The last index is unreachable

This turns the problem into a **reachability check**, not a path enumeration problem.

---

### Greedy Strategy (Core Trick)

Maintain a variable:

```
maxReach = the farthest index you can reach so far
```

Iterate through the array:

1. If the current index `i` is **greater than `maxReach`**, return `False`
2. Otherwise, update:

   ```
   maxReach = max(maxReach, i + nums[i])
   ```
3. If `maxReach` reaches or exceeds the last index, return `True`

Why this is greedy:

* At each step, you **choose the jump that maximizes future reach**
* No backtracking or DP table is needed

---

### Why Greedy Works (Intuition)

* Jump choices **overlap**
* Only the **farthest reachable boundary** matters
* Smaller jumps are irrelevant if a larger jump already covers them

Once a region is reachable, **how you reached it does not matter**.

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0

        for i in range(len(nums)):
            # If current index is not reachable
            if i > max_reach:
                return False

            # Update farthest reachable index
            max_reach = max(max_reach, i + nums[i])

            # Early exit if we can reach the last index
            if max_reach >= len(nums) - 1:
                return True

        return True
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```text
nums = [2, 3, 1, 1, 4]
```

#### Goal

Reach index `4`

---

### Step-by-Step Processing

| Index (`i`) | `nums[i]` | `maxReach` before | Check (`i > maxReach`) | New `maxReach` |
| --- | --- | --- | --- | --- |
| 0 | 2 | 0 | No | max(0, 0+2) = 2 |
| 1 | 3 | 2 | No | max(2, 1+3) = 4 |
| 2 | 1 | 4 | No | max(4, 2+1) = 4 |
| 3 | 1 | 4 | No | max(4, 3+1) = 4 |
| 4 | 4 | 4 | No | max(4, 4+4) = 8 |

---

### Key Moment

At **index 1**, we jump far enough to reach the **last index**:

```
maxReach = 4
```

Since:

```
maxReach >= last index
```

→ **Return `True`**

---

### Failure Example (Why It Can Fail)

#### Input

```text
nums = [3, 2, 1, 0, 4]
```

#### Processing

| Index | `maxReach` |
| --- | --- |
| 0 | 3 |
| 1 | 3 |
| 2 | 3 |
| 3 | 3 |
| 4 | ❌ unreachable |

At index `4`:

```
i (4) > maxReach (3)
```

→ **Return `False`**

---

### Time and Space Complexity

* **Time:** `O(n)`
* **Space:** `O(1)`

Optimal and interview-preferred.

---

### Interview Summary (One-Line Explanation)

> *Track the farthest reachable index while traversing; if you ever reach an index beyond that range, the jump is impossible.*

---

If you want, I can also provide:

* Backward greedy version (right-to-left)
* Visual reach boundary diagram
* Common greedy mistakes
* Comparison with DP approach (and why DP is inferior here)

Just tell me.