---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 45: Jump Game II"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 45: Jump Game II

**LeetCode 45 (Jump Game II)**, aligned with interview expectations and greedy-pattern reasoning.

---

## LeetCode 45 — Jump Game II

### Problem Statement

You are given an integer array `nums`, where `nums[i]` represents the **maximum jump length** from index `i`.

You start at index `0` and want to reach the **last index** in the **minimum number of jumps**.

It is guaranteed that you can reach the last index.

---

## Key Observation (Most Important Insight)

This is **not a DP problem in implementation**, even though it looks like one.

Instead, it is a **Greedy + Range Expansion** problem.

### Core Idea

* From the current range of reachable indices, **choose the jump that maximizes how far you can go next**
* Each jump expands your reachable window as far as possible
* The number of times you expand this window = **minimum jumps**

This is equivalent to a **level-order traversal (BFS)** on jump ranges, but implemented greedily in O(n).

---

## Why Greedy Works Here

At any index `i`, you only care about:

```
farthest reachable index = max(i + nums[i])
```

You do **not** care *which index* you land on — only **how far the next jump can reach**.

Greedy guarantees:

* Minimum number of jumps
* Optimal local choice leads to optimal global solution

---

## Greedy Strategy (Mental Model)

Think in terms of **jump ranges**:

* `current_end` → boundary of the current jump
* `farthest` → farthest index reachable within current jump
* When `i == current_end`, you **must jump**

  * Increase jump count
  * Update `current_end = farthest`

---

## Greedy Algorithm (Step Logic)

1. Initialize:

   * `jumps = 0`
   * `current_end = 0`
   * `farthest = 0`
2. Traverse array from `0` to `n-2`:

   * Update `farthest = max(farthest, i + nums[i])`
   * If `i == current_end`:

     * `jumps += 1`
     * `current_end = farthest`
3. Return `jumps`

---

## Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        current_end = 0
        farthest = 0
        
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                jumps += 1
                current_end = farthest
        
        return jumps
```

---

## Complete Worked Example (Step-by-Step)

### Input

```text
nums = [2, 3, 1, 1, 4]
```

### Goal

Reach index `4` with minimum jumps.

---

### Initialization

```
jumps = 0
current_end = 0
farthest = 0
```

---

### Iteration Details

| i | nums[i] | i + nums[i] | farthest | current\_end | jumps | Action |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 2 | 2 | 0 | 0 | i == current\_end → jump |
|  |  |  |  | → 2 | → 1 | Jump boundary extended |
| 1 | 3 | 4 | 4 | 2 | 1 | continue |
| 2 | 1 | 3 | 4 | 2 | 1 | i == current\_end → jump |
|  |  |  |  | → 4 | → 2 | Jump boundary extended |

Loop stops at `i = n-2 = 3`.

---

### Result

```
Minimum jumps = 2
```

---

## Visual Interpretation (Range Expansion)

```
Jump 1 covers: [0 → 2]
Jump 2 covers: [2 → 4] (end)
```

---

## Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | O(n) |
| Space Complexity | O(1) |

---

## Common Interview Pitfalls

1. **Trying DP first**

   * DP is O(n²) and unnecessary
2. **Tracking actual jump path**

   * Only the jump count matters
3. **Looping till last index**

   * Loop only till `n-2` (last jump handled implicitly)

---

## Pattern Classification (For DSA Rounds)

* **Greedy**
* **Range Expansion**
* **BFS Level Simulation (without queue)**

---

If you want, I can also provide:

* BFS-style explanation mapping to this greedy logic
* Comparison with **Jump Game I**
* Proof of correctness
* Edge-case walkthroughs

Just tell me.