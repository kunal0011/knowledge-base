---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 403: Frog Jump"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 403: Frog Jump

**LeetCode 403 – Frog Jump**, focusing on **state definition, transitions, DP table construction, and a worked example**. This is written in an interview-ready, system-design style rather than trial-and-error backtracking.

---

## LeetCode 403 – Frog Jump

### Problem Statement

A frog is crossing a river by jumping on stones placed at increasing positions.

* The frog starts on stone `0`.
* The **first jump must be exactly `1` unit**.
* If the last jump was `k`, the next jump must be `k-1`, `k`, or `k+1`.
* The frog may only jump **forward**.
* Determine whether the frog can reach the **last stone**.

---

## Key Observation

The frog’s **future options depend on two variables**:

1. **Which stone it is currently on**
2. **What was the size of the last jump**

This naturally leads to a **2D DP state**.

---

## DP State Definition

Let:

```
dp[i][k] = True
```

Meaning:

> The frog **can reach stone `i`** using a **last jump of length `k`**

Where:

* `i` → index of stone
* `k` → jump length used to reach stone `i`

---

## Base Case

* The frog starts at stone `0`
* No jump yet → conceptual jump `0`

```
dp[0][0] = True
```

---

## State Transition

Suppose the frog is at stone `i` with last jump `k`.

From here, it can attempt jumps of size:

```
k - 1, k, k + 1   (must be > 0)
```

Let `next_jump = nk`

If there exists a stone at position:

```
stones[i] + nk
```

Say its index is `j`, then:

```
dp[j][nk] = True
```

---

## DP Table Construction Strategy

### Why we don’t use a full matrix

* Jump sizes can go up to `O(n)`
* Sparse states → use a **map of sets**

### Data Structure

```
dp[i] = set of jump sizes that can land on stone i
```

---

## Algorithm (Step-by-Step)

1. Store stone positions in a hashmap → position → index
2. Initialize `dp[0] = {0}`
3. Iterate over stones from left to right
4. For each reachable jump size, try `k-1, k, k+1`
5. Mark reachable states
6. Check if the last stone has any reachable jump

---

## Example Walkthrough

### Input

```
stones = [0, 1, 3, 5, 6, 8, 12, 17]
```

---

### Stepwise DP Evolution

| Stone | Reachable jump sizes |
| --- | --- |
| 0 | {0} |
| 1 | {1} |
| 3 | {2} |
| 5 | {2} |
| 6 | {1, 3} |
| 8 | {2, 3} |
| 12 | {4} |
| 17 | {5} |

Since the **last stone (17)** has reachable jumps → **answer is `True`**

---

## DP Transition Illustration

At stone `6`, reachable jumps are `{1, 3}`:

* From jump `1` → try `{1, 2}`
* From jump `3` → try `{2, 3, 4}`

Valid next stones found at `8` and `12`.

---

## Python 3 (Typed, Production-Grade)

```python
from typing import List, Dict, Set

class Solution:
    def canCross(self, stones: List[int]) -> bool:
        n = len(stones)
        
        # Quick pruning: first jump must be 1
        if stones[1] != 1:
            return False
        
        # Map stone position to index
        pos_to_index: Dict[int, int] = {pos: i for i, pos in enumerate(stones)}
        
        # dp[i] = set of jump sizes that can land on stone i
        dp: List[Set[int]] = [set() for _ in range(n)]
        dp[0].add(0)
        
        for i in range(n):
            for jump in dp[i]:
                for next_jump in (jump - 1, jump, jump + 1):
                    if next_jump <= 0:
                        continue
                    
                    next_pos = stones[i] + next_jump
                    if next_pos in pos_to_index:
                        next_index = pos_to_index[next_pos]
                        dp[next_index].add(next_jump)
        
        return len(dp[n - 1]) > 0
```

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n²)** worst-case |
| Space | **O(n²)** (sparse in practice) |

---

## Why This DP Works (Interview Insight)

* The **state fully captures the problem constraints**
* Avoids recomputation inherent in naive recursion
* Sparse DP ensures performance
* Easy to reason about correctness and termination

---

## Summary

* **State**: `(stone_index, last_jump)`
* **Transition**: `k → k-1, k, k+1`
* **Structure**: `dp[i] = set(jumps)`
* **Goal**: any reachable jump at last stone

If you want, I can next:

* Convert this to a **top-down memo DP**
* Show **why greedy fails**
* Draw a **state transition graph**
* Compare with **backtracking + memo**

Just tell me.