---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 134: Gas Station"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - amazon
  - google
---

# LeetCode 134: Gas Station

**Target Companies:** Amazon (Top Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Prefix Sum / Array  

---

### Problem Statement

There are $n$ gas stations along a circular route, where the amount of gas at the $i$-th station is `gas[i]`.

You have a car with an unlimited gas tank and it costs `cost[i]` of gas to travel from the $i$-th station to its next $(i + 1)$-th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays `gas` and `cost`, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return `-1`. If there exists a solution, it is **guaranteed to be unique**.

---

### Input & Output Formats & Constraints

- **Input:**
  - `gas`: `List[int]` / `vector<int>` / `int[]` ($1 \le \text{gas.length} \le 10^5$).
  - `cost`: `List[int]` / `vector<int>` / `int[]` ($\text{cost.length} == \text{gas.length}$).
- **Output:**
  - `int` — the 0-indexed starting station index, or `-1` if impossible.
- **Constraints:**
  - $n == \text{gas.length} == \text{cost.length}$
  - $1 \le n \le 10^5$
  - $0 \le \text{gas}[i], \text{cost}[i] \le 10^4$

---

### Key Idea & Intuition

Let $\Delta[i] = \text{gas}[i] - \text{cost}[i]$ be the net gain (or loss) of gas upon visiting and leaving station $i$.

#### Fundamental Invariant 1: Global Energy Conservation
If the total gas across the circuit is strictly less than the total cost:
$$\sum_{i=0}^{n-1} \text{gas}[i] < \sum_{i=0}^{n-1} \text{cost}[i] \iff \sum_{i=0}^{n-1} \Delta[i] < 0$$
No starting station can ever complete the circuit because the entire system has a net energy deficit. Hence, return `-1`.

#### Fundamental Invariant 2: The Reset Property
Suppose we start at station $A$ with an empty tank and travel successfully until station $B$, where our tank drops below $0$:
$$\sum_{k=A}^{B} \Delta[k] < 0$$
Can any intermediate station $C$ ($A < C \le B$) be the valid starting station?
**Proof by contradiction:**
- Since our car made it from $A$ to $C$, the fuel accumulated from $A$ to $C - 1$ was non-negative: $\sum_{k=A}^{C-1} \Delta[k] \ge 0$.
- The fuel from $C$ to $B$ is:
  $$\sum_{k=C}^{B} \Delta[k] = \sum_{k=A}^{B} \Delta[k] - \sum_{k=A}^{C-1} \Delta[k]$$
- Because $\sum_{k=A}^{B} \Delta[k] < 0$ and $\sum_{k=A}^{C-1} \Delta[k] \ge 0$, their difference must be strictly negative:
  $$\sum_{k=C}^{B} \Delta[k] \le \sum_{k=A}^{B} \Delta[k] < 0$$
- If we had started at station $C$ with an empty tank ($0$), we would run out of gas even faster before or at $B$!
- **Conclusion:** No station in the range $[A, B]$ can be the starting station. We can greedily reset our candidate start to $B + 1$!

Combining Invariant 1 and Invariant 2 guarantees that if $\sum \text{gas} \ge \sum \text{cost}$, the candidate start index identified after a single linear pass **must** be the unique valid solution. We do not even need a second pass!

---

### Solution Approach (Step-by-Step)

1. Initialize `total_tank = 0`, `curr_tank = 0`, and `start_station = 0`.
2. Loop `i` from $0$ to $n - 1$:
   - Let `delta = gas[i] - cost[i]`.
   - `total_tank += delta`
   - `curr_tank += delta`
   - If `curr_tank < 0`:
     - The journey starting at `start_station` cannot reach past `i`.
     - Reset `start_station = i + 1`.
     - Reset `curr_tank = 0`.
3. If `total_tank >= 0`, return `start_station`.
4. Otherwise, return `-1`.

---

### Visual Algorithm Walkthrough

For `gas = [1, 2, 3, 4, 5]`, `cost = [3, 4, 5, 1, 2]`:
$\Delta = [-2, -2, -2, +3, +3]$

```
Station i = 0: delta = 1 - 3 = -2
  total_tank = -2, curr_tank = -2 < 0!
  -> Reset: start_station = 1, curr_tank = 0

Station i = 1: delta = 2 - 4 = -2
  total_tank = -4, curr_tank = -2 < 0!
  -> Reset: start_station = 2, curr_tank = 0

Station i = 2: delta = 3 - 5 = -2
  total_tank = -6, curr_tank = -2 < 0!
  -> Reset: start_station = 3, curr_tank = 0

Station i = 3: delta = 4 - 1 = +3
  total_tank = -3, curr_tank = 3 >= 0 (OK)

Station i = 4: delta = 5 - 2 = +3
  total_tank = 0, curr_tank = 6 >= 0 (OK)

End of loop:
  total_tank = 0 >= 0 (Valid!)
  start_station = 3

Circuit from station 3:
  At 3: gas = 4, cost to 4 = 1 -> tank = 3
  At 4: gas = 5, cost to 0 = 2 -> tank = 3 + 5 - 2 = 6
  At 0: gas = 1, cost to 1 = 3 -> tank = 6 + 1 - 3 = 4
  At 1: gas = 2, cost to 2 = 4 -> tank = 4 + 2 - 4 = 2
  At 2: gas = 3, cost to 3 = 5 -> tank = 2 + 3 - 5 = 0 (Success!)
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `gas = [1, 2, 3, 4, 5]`, `cost = [3, 4, 5, 1, 2]`
- **Output:** `3`

#### Example 2 (Total Deficit):
- **Input:** `gas = [2, 3, 4]`, `cost = [3, 4, 3]`
- **Tracing:**
  - $\sum gas = 9$, $\sum cost = 10$.
  - Total deficit $= -1 < 0$.
- **Output:** `-1`

#### Example 3 (Single Station):
- **Input:** `gas = [5]`, `cost = [4]`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_tank = 0
        curr_tank = 0
        start_station = 0
        
        for i in range(len(gas)):
            delta = gas[i] - cost[i]
            total_tank += delta
            curr_tank += delta
            
            # If unable to reach the next station, reset starting candidate
            if curr_tank < 0:
                start_station = i + 1
                curr_tank = 0
                
        return start_station if total_tank >= 0 else -1
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    int canCompleteCircuit(const std::vector<int>& gas, const std::vector<int>& cost) {
        int total_tank = 0;
        int curr_tank = 0;
        int start_station = 0;
        int n = static_cast<int>(gas.size());
        
        for (int i = 0; i < n; ++i) {
            int delta = gas[i] - cost[i];
            total_tank += delta;
            curr_tank += delta;
            
            if (curr_tank < 0) {
                start_station = i + 1;
                curr_tank = 0;
            }
        }
        
        return total_tank >= 0 ? start_station : -1;
    }
};
```

#### Java 17
```java
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int totalTank = 0;
        int currTank = 0;
        int startStation = 0;
        
        for (int i = 0; i < gas.length; i++) {
            int delta = gas[i] - cost[i];
            totalTank += delta;
            currTank += delta;
            
            if (currTank < 0) {
                startStation = i + 1;
                currTank = 0;
            }
        }
        
        return totalTank >= 0 ? startStation : -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - A single pass through the $n$ gas stations. Each element is visited exactly once.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - Only three scalar integer variables (`total_tank`, `curr_tank`, `start_station`) are tracked.

---

### Takeaway Pattern & Interview Traps

- **Why a Second Pass is Unnecessary:** Many candidates write an extra loop to simulate circular traversal from `start_station`. This is redundant! If $\sum \text{gas} \ge \sum \text{cost}$ and the car successfully travels from `start_station` to $n - 1$ with `curr_tank >= 0`, the accumulated fuel at $n - 1$ plus the known net surplus of the entire array guarantees the remainder of the circuit $[0, start\_station - 1]$ can also be completed.
- **Starting Station Beyond Index $n - 1$:** If `curr_tank < 0` at index $n - 1$, `start_station` becomes $n$. If `total_tank >= 0`, this case is mathematically impossible unless $n = 0$.