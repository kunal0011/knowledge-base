---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 174: Dungeon Game"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 174: Dungeon Game

**LeetCode 174 – Dungeon Game**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 174 — Dungeon Game

### Problem Statement (Condensed)

You are given an `m x n` grid `dungeon`, where:

* Each cell contains an integer:

  * **negative** → health loss
  * **positive** → health gain
* The knight starts at `(0,0)` and must reach `(m-1, n-1)`
* He can move **only right or down**
* Health must **never drop to 0 or below**

**Goal:**  
Find the **minimum initial health** required to guarantee survival to the destination.

---

## Key Insight (Why DP from the end)

This is **not** a forward DP problem.

Why?

* At any cell, whether you survive depends on **future cells**, not past.
* We need to know **how much health is required when entering a cell**, not how much we gain after leaving it.

Hence:  
👉 **DP from bottom-right to top-left**

---

## DP State Definition

Let:

```
dp[i][j] = minimum health required to ENTER cell (i, j)
           so that the knight can safely reach the princess
```

Important:

* Health must always be **≥ 1**
* `dp[i][j]` answers:  
  *"What is the minimum HP needed before stepping on (i, j)?"*

---

## State Transition

From `(i, j)` you can go:

* Right → `(i, j+1)`
* Down → `(i+1, j)`

You want the **less demanding path**, so take the minimum.

### Transition Formula

```
min_needed_from_next = min(dp[i+1][j], dp[i][j+1])

dp[i][j] = max(1, min_needed_from_next - dungeon[i][j])
```

### Why `max(1, …)`?

* Even if dungeon gives health, you **cannot start with ≤ 0**
* Minimum allowed health is **1**

---

## Base Case (Destination Cell)

At the princess cell `(m-1, n-1)`:

```
dp[m-1][n-1] = max(1, 1 - dungeon[m-1][n-1])
```

Reason:

* You must leave this cell with at least `1` HP

---

## DP Table Setup (Sentinel Trick)

To avoid bounds checking:

* Create `dp` of size `(m+1) x (n+1)`
* Initialize all values to `∞`
* Set:

  ```
  dp[m][n-1] = 1
  dp[m-1][n] = 1
  ```

This allows uniform transition logic.

---

## Example Walkthrough

### Input

```
dungeon =
[
  [-2, -3,  3],
  [-5, -10, 1],
  [10, 30, -5]
]
```

---

### Step 1: Initialize DP Table

```
dp (initial)

∞    ∞    ∞    ∞
∞    ∞    ∞    ∞
∞    ∞    ∞    ∞
∞    ∞    ∞    ∞

Set:
dp[3][2] = 1
dp[2][3] = 1
```

---

### Step 2: Fill Bottom-Up

#### Cell (2,2) = -5

```
min(1,1) - (-5) = 6
dp[2][2] = 6
```

#### Cell (2,1) = 30

```
min(∞,6) - 30 = -24 → max(1, -24) = 1
dp[2][1] = 1
```

#### Cell (2,0) = 10

```
min(∞,1) - 10 = -9 → 1
dp[2][0] = 1
```

---

#### Cell (1,2) = 1

```
min(6,∞) - 1 = 5
dp[1][2] = 5
```

#### Cell (1,1) = -10

```
min(1,5) - (-10) = 11
dp[1][1] = 11
```

#### Cell (1,0) = -5

```
min(1,11) - (-5) = 6
dp[1][0] = 6
```

---

#### Cell (0,2) = 3

```
min(5,∞) - 3 = 2
dp[0][2] = 2
```

#### Cell (0,1) = -3

```
min(11,2) - (-3) = 5
dp[0][1] = 5
```

#### Cell (0,0) = -2

```
min(6,5) - (-2) = 7
dp[0][0] = 7
```

---

### Final DP Table (Valid Cells)

```
7    5    2
6   11    5
1    1    6
```

---

## Final Answer

```
Minimum Initial Health = dp[0][0] = 7
```

---

## Python 3 Implementation (With Typing)

```python
from typing import List
import math

class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        m, n = len(dungeon), len(dungeon[0])

        # dp with sentinels
        dp = [[math.inf] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = 1
        dp[m - 1][n] = 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
                dp[i][j] = max(1, need)

        return dp[0][0]
```

---

## Complexity Analysis

* **Time:** `O(m * n)`
* **Space:** `O(m * n)`  
  (Can be optimized to `O(n)` with rolling array)

---

## Conceptual Summary

* DP state answers **“minimum health required before entering a cell”**
* Transition is **future-dependent**
* Bottom-up DP ensures correctness
* `max(1, …)` enforces survival constraint

If you want, I can next:

* Convert this to **1D DP**
* Explain **why greedy fails**
* Draw a **conceptual DP dependency graph**