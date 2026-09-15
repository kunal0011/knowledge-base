---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1601: Maximum Number of Achievable Transfer Requests"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 1601: Maximum Number of Achievable Transfer Requests

## LeetCode 1601 — Maximum Number of Achievable Transfer Requests

---

### Problem Statement

You are given `n` buildings and a list of **transfer requests**.  
Each request is represented as `[from, to]`, meaning one employee wants to move from building `from` to building `to`.

A set of requests is **achievable** if, for **every building**, the number of employees leaving equals the number entering (net change = 0).

**Goal:**  
Return the **maximum number of requests** that can be satisfied simultaneously.

**Constraints**

* `1 ≤ n ≤ 20`
* `1 ≤ requests.length ≤ 16`
* `0 ≤ from, to < n`

---

### Key Observations

1. Each request can be either:

   * **taken**
   * **not taken**
2. We are asked to find the **largest subset** of requests such that:

   ```
   for every building i:
   incoming[i] - outgoing[i] == 0
   ```
3. The constraint `requests.length ≤ 16` strongly suggests:

   * **Backtracking / brute force with pruning**
   * Total subsets = `2^16 = 65,536` (manageable)
4. This is **not** a greedy problem:

   * Taking more requests early may block feasibility later.
5. This is a **binary-decision backtracking tree**:

   * At each level, choose or skip the current request.

---

## Core Backtracking Idea

### State

* `index`: current request being considered
* `balance[]`: net employee change per building
* `count`: number of requests chosen so far

### Decision at each request

1. **Skip** the request
2. **Take** the request:

   * `balance[from] -= 1`
   * `balance[to] += 1`

### Validity Check (only at leaves)

* At the end (`index == len(requests)`), check:

  ```
  all(balance[i] == 0)
  ```

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        balance = [0] * n
        self.max_count = 0

        def backtrack(index: int, count: int) -> None:
            if index == len(requests):
                if all(b == 0 for b in balance):
                    self.max_count = max(self.max_count, count)
                return

            # Option 1: skip current request
            backtrack(index + 1, count)

            # Option 2: take current request
            frm, to = requests[index]
            balance[frm] -= 1
            balance[to] += 1

            backtrack(index + 1, count + 1)

            # undo (backtrack)
            balance[frm] += 1
            balance[to] -= 1

        backtrack(0, 0)
        return self.max_count
```

---

## Example Explanation

### Input

```
n = 3
requests = [[0,1],[1,2],[2,0],[1,0]]
```

### Interpretation

* `[0,1]`: one leaves building 0, enters 1
* `[1,2]`: 1 → 2
* `[2,0]`: 2 → 0
* `[1,0]`: 1 → 0

### One valid maximal subset

```
[0,1], [1,2], [2,0]
```

Net effect:

```
building 0: -1 +1 = 0
building 1: +1 -1 = 0
building 2: +1 -1 = 0
```

So answer = `3`

---

## Backtracking Tree Structure (Complete Conceptual Tree)

> Each level represents **one request**  
> Each node branches into **Skip** or **Take**

Let requests be indexed as:

```
R0 = [0,1]
R1 = [1,2]
R2 = [2,0]
```

### Conceptual Binary Tree

```
(index=0, count=0)
                                     /                 \
                                skip R0               take R0
                                 |                     |
                          (i=1, c=0)              (i=1, c=1)
                           /        \               /        \
                      skip R1     take R1      skip R1     take R1
                        |           |             |           |
                 (i=2,c=0)   (i=2,c=1)     (i=2,c=1)   (i=2,c=2)
                   /   \        /   \          /   \        /   \
              skip  take   skip   take     skip   take  skip   take
                |      |      |      |         |      |     |      |
        (i=3,c=0)(1) (1)  (2)     (1)   (2)    (2)   (3)
          ❌      ❌    ❌    ❌       ❌      ❌     ❌    ✅
```

Legend:

* `(i, c)` → `(request index, chosen count)`
* `✅` → all balances zero → valid
* `❌` → invalid (imbalanced buildings)

---

### Key Navigation Insight

* **Every path from root to leaf** corresponds to **one subset** of requests.
* Validation happens **only at leaf nodes**.
* We explore **all possibilities**, but only keep the maximum valid count.

---

## Why This Is Backtracking (Not DP)

* State space depends on **subset choices**, not sequence order.
* Balance vector changes dynamically.
* No overlapping subproblems with identical state `(index, balance[])`.

---

## Complexity Analysis

* **Time Complexity:** `O(2^m * n)` where `m = len(requests)`
* **Space Complexity:** `O(n)` for balance + recursion stack

---

## Pattern Recognition

This problem is a textbook example of:

* **Subset backtracking**
* **Binary decision tree**
* **Constraint validation at leaf nodes**

Closely related to:

* Subset enumeration
* LC 494 (Target Sum)
* LC 473 (Matchsticks to Square)

---

### One-Line Interview Explanation

> “We backtrack over all subsets of requests. For each request, we choose to take or skip it while tracking net employee balance per building. At the end, if all balances are zero, the subset is valid, and we maximize its size.”

If you want next, I can:

* Add **pruning using remaining requests**
* Show **bitmask-based solution**
* Compare this with **flow-based modeling** and explain why it’s overkill