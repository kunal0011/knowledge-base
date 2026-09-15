---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 134: Gas Station"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 134: Gas Station

**134 – Gas Station**, aligned with how greedy problems are expected to be reasoned about in interviews and competitive programming.

---

## 1. Problem Statement

You are given two integer arrays:

* `gas[i]`: the amount of gas available at station `i`
* `cost[i]`: the amount of gas required to travel from station `i` to station `(i + 1) % n`

You start with an **empty tank** at **one** gas station.  
If you can travel around the circuit once in a clockwise direction, return the **starting station index**.  
Otherwise, return `-1`.

**Constraints**

* `n == len(gas) == len(cost)`
* `1 ≤ n ≤ 10⁵`
* `0 ≤ gas[i], cost[i] ≤ 10⁴`

---

## 2. Key Observations (Critical for Greedy)

### Observation 1: Global Feasibility Condition

If the **total gas available** is less than the **total cost**, completing the circuit is **impossible** from any station.

[  
\sum gas < \sum cost \Rightarrow \text{answer = -1}  
]

This is a **necessary and sufficient** condition.

---

### Observation 2: Prefix Deficit Reset Logic

Suppose you start at station `s` and while moving forward, at station `i` your fuel becomes negative.

**Important conclusion**:

> Any station between `s` and `i` (inclusive) **cannot** be a valid starting station.

Why?

* Starting later only gives you **less accumulated gas** before hitting the same deficit point.

Therefore, when a failure occurs, the next possible candidate is **`i + 1`**.

---

### Observation 3: Only One Valid Answer Exists

If a solution exists, it is **unique**.  
Greedy scanning will find it.

---

## 3. Greedy Strategy (Core Trick)

We scan stations once and maintain:

1. `total_balance`: total gas surplus across all stations
2. `current_balance`: gas balance while simulating from a candidate start
3. `start`: current candidate starting index

**Algorithm**

* Traverse from `0` to `n-1`
* Add `gas[i] - cost[i]` to both balances
* If `current_balance < 0`:

  * Reset `current_balance = 0`
  * Set `start = i + 1`
* After traversal:

  * If `total_balance >= 0`, return `start`
  * Else return `-1`

**Why this works**

* We eliminate impossible starts early
* We rely on the global feasibility condition
* Single pass, O(n) time

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_balance: int = 0
        current_balance: int = 0
        start: int = 0

        for i in range(len(gas)):
            diff = gas[i] - cost[i]
            total_balance += diff
            current_balance += diff

            # If we cannot reach next station
            if current_balance < 0:
                start = i + 1
                current_balance = 0

        return start if total_balance >= 0 else -1
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```
gas  = [1, 2, 3, 4, 5]
cost = [3, 4, 5, 1, 2]
```

---

### Step 1: Check Global Feasibility

| Total Gas | Total Cost |
| --- | --- |
| 15 | 15 |

Since `total_gas >= total_cost`, a solution **exists**.

---

### Step 2: Greedy Simulation

| i | gas[i] | cost[i] | diff | current\_balance | total\_balance | start |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 3 | -2 | -2 ❌ | -2 | 1 |
|  |  |  |  | reset → 0 |  |  |
| 1 | 2 | 4 | -2 | -2 ❌ | -4 | 2 |
|  |  |  |  | reset → 0 |  |  |
| 2 | 3 | 5 | -2 | -2 ❌ | -6 | 3 |
|  |  |  |  | reset → 0 |  |  |
| 3 | 4 | 1 | +3 | 3 | -3 | 3 |
| 4 | 5 | 2 | +3 | 6 | 0 | 3 |

---

### Step 3: Final Check

* `total_balance = 0 ≥ 0`
* Valid starting index = **3**

---

### Step 4: Manual Verification from Start = 3

| Station | Gas in Tank | Action |
| --- | --- | --- |
| 3 | 0 + 4 - 1 = 3 | Move |
| 4 | 3 + 5 - 2 = 6 | Move |
| 0 | 6 + 1 - 3 = 4 | Move |
| 1 | 4 + 2 - 4 = 2 | Move |
| 2 | 2 + 3 - 5 = 0 | Complete |

✔ Circuit completed successfully.

---

## 6. Complexity Analysis

* **Time Complexity**: `O(n)`
* **Space Complexity**: `O(1)`

---

## 7. Interview Takeaways

* This problem is **not simulation-heavy**, it is **greedy elimination**
* Resetting the start index is the key insight
* Global sum check converts a local greedy strategy into a correct solution
* Similar greedy pattern appears in:

  * Load balancing
  * Circular scheduling
  * Prefix-sum feasibility problems

If you want, I can also provide:

* Greedy proof using prefix sums
* Visualization using prefix minimum
* Comparison with brute-force and why it TLEs
* Similar problems in the same greedy category