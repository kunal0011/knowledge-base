---
date: "2026-09-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 403: Frog Jump"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - hash-table
  - google
  - amazon
  - meta
  - bloomberg
---

# LeetCode 403: Frog Jump

**Target Companies:** Google, Amazon, Meta, Bloomberg, Microsoft  
**Difficulty:** Hard  
**Topic:** Dynamic Programming / Hash Map / Reachability Graph  

---

### Problem Statement

A frog is crossing a river. The river is divided into some number of units, and at each unit, there may or may not exist a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of `stones` positions (in units) in sorted **ascending order**, determine if the frog can cross the river by landing on the last stone. Initially, the frog is on the first stone (`stones[0] == 0`) and assumes the first jump must be `1` unit.

If the frog's last jump was `k` units, its next jump must be either `k - 1`, `k`, or `k + 1` units. The frog can only jump in the forward direction.

---

### Input & Output Formats & Constraints

- **Input:** `stones: List[int]` — Sorted array of stone coordinates.
- **Output:** `bool` — `true` if the frog can reach `stones[-1]`, `false` otherwise.
- **Constraints:**
  - $2 \le \text{stones.length} \le 2000$
  - $0 \le \text{stones}[i] \le 2^{31} - 1$
  - $\text{stones}[0] == 0$
  - $\text{stones}$ is strictly increasing.

---

### Key Idea & Intuition

1. **State Representation:**
   - Whether the frog can reach stone $j$ depends not only on which stone it is standing on, but also on the **length of the jump that brought it there**.
   - If the last jump was $k$, the subsequent jump options are $k - 1$, $k$, and $k + 1$.
   - Therefore, the state must be 2D: $(\text{stone\_index } i, \text{last\_jump } k)$.

2. **Theoretical Upper Bound on $k$:**
   - On the 1st jump from stone $0$, max jump is $1$.
   - On the 2nd jump, max jump is $2$.
   - By mathematical induction, after visiting at most $i$ stones, the jump length $k \le i$.
   - For $N \le 2000$, the maximum jump cannot exceed $2000$. This prevents state explosion and guarantees $\mathcal{O}(N^2)$ states.

3. **Forward Push DP (Map of Sets):**
   - Store reachable jump sizes for each stone position in a hash map:
     $$\text{dp}[\text{position}] = \{\text{set of incoming jump sizes } k\}$$
   - Base Case: $\text{dp}[0] = \{0\}$.
   - Early Pruning: If $\text{stones}[1] \ne 1$, the frog can never make the required initial jump of length $1 \implies$ immediately return `false`.
   - Forward Transition:
     For each stone position $p$ and each incoming jump $k \in \text{dp}[p]$:
     For candidate next jump $\text{step} \in \{k - 1, k, k + 1\}$:
     If $\text{step} > 0$ and $p + \text{step} \in \text{dp}$:
       Add $\text{step}$ to $\text{dp}[p + \text{step}]$.

---

### Solution Approach (Step-by-Step)

1. **Prune Invalid Initial Step:**
   - If `stones[1] != 1`, return `false`.
2. **Initialize DP Map:**
   - Create a hash map `dp` mapping each stone position in `stones` to an empty set.
   - Insert `dp[0].add(0)`.
3. **Iterate Through Stones:**
   - For each stone position in `stones`:
     - For each jump length `k` in `dp[stone]`:
       - For `step` in `(k - 1, k, k + 1)`:
         - If `step > 0` and `(stone + step)` is a key in `dp`:
           - Add `step` to `dp[stone + step]`.
4. **Evaluate Last Stone:**
   - Return `len(dp[stones[-1]]) > 0`.

---

### Visual Algorithm Walkthrough

`stones = [0, 1, 3, 5, 6, 8, 12, 17]`

```
Index:    0    1    2    3    4    5    6    7
Stone:    0    1    3    5    6    8   12   17

Stone 0:
  jumps = {0}
  try next: step = 0+1 = 1 -> lands on 1.
  dp[1] = {1}

Stone 1:
  jumps = {1}
  try steps: {0 (skip), 1, 2}
  1 + 1 = 2 (no stone at 2)
  1 + 2 = 3 (stone exists!) -> dp[3] = {2}

Stone 3:
  jumps = {2}
  try steps: {1, 2, 3}
  3 + 1 = 4 (no stone)
  3 + 2 = 5 (stone!) -> dp[5] = {2}
  3 + 3 = 6 (stone!) -> dp[6] = {3}

Stone 5:
  jumps = {2}
  try steps: {1, 2, 3}
  5 + 1 = 6 (stone!) -> dp[6] = {1, 3}
  5 + 3 = 8 (stone!) -> dp[8] = {3}

Stone 6:
  jumps = {1, 3}
  from 1 -> try {1, 2} -> 6 + 2 = 8 (stone!) -> dp[8] = {2, 3}
  from 3 -> try {2, 3, 4} -> 6 + 4 = 10 (no), (already checked 8)

Stone 8:
  jumps = {2, 3}
  from 3 -> 8 + 4 = 12 (stone!) -> dp[12] = {4}

Stone 12:
  jumps = {4}
  from 4 -> try {3, 4, 5} -> 12 + 5 = 17 (stone!) -> dp[17] = {5}

Last stone (17) has incoming jump {5}.
Result: True!
```

---

### Solved Examples with Multiple Inputs

| Case | `stones` | Traversal Path | Result | Explanation |
|---|---|---|---|---|
| **Standard Reachable** | `[0,1,3,5,6,8,12,17]` | $0 \xrightarrow{1} 1 \xrightarrow{2} 3 \xrightarrow{2} 5 \xrightarrow{3} 8 \xrightarrow{4} 12 \xrightarrow{5} 17$ | `true` | Valid sequence of jumps |
| **Unreachable Gap** | `[0,1,2,3,4,8,9,11]` | Stuck at stone 4 (max jump from 4 is 3, $4+3=7 < 8$) | `false` | Gap between 4 and 8 cannot be bridged |
| **Invalid Start** | `[0, 2]` | First jump must be 1, but next stone is at 2 | `false` | Early exit `stones[1] != 1` |
| **Minimal Valid** | `[0, 1]` | Jump $1$ reaches destination directly | `true` | Trivially satisfied |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List, Dict, Set

class Solution:
    def canCross(self, stones: List[int]) -> bool:
        # Initial prune: first jump must be exactly 1
        if stones[1] != 1:
            return False
            
        n = len(stones)
        # dp[pos] contains all jump lengths that can land on pos
        dp: Dict[int, Set[int]] = {pos: set() for pos in stones}
        dp[0].add(0)
        
        target = stones[-1]
        
        for pos in stones:
            for jump in dp[pos]:
                for step in (jump - 1, jump, jump + 1):
                    if step > 0:
                        next_pos = pos + step
                        if next_pos == target:
                            return True
                        if next_pos in dp:
                            dp[next_pos].add(step)
                            
        return len(dp[target]) > 0
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_map>
#include <unordered_set>

class Solution {
public:
    bool canCross(std::vector<int>& stones) {
        if (stones[1] != 1) return false;

        int n = stones.size();
        int target = stones.back();

        std::unordered_map<int, std::unordered_set<int>> dp;
        for (int stone : stones) {
            dp[stone] = std::unordered_set<int>();
        }
        dp[0].insert(0);

        for (int pos : stones) {
            for (int jump : dp[pos]) {
                for (int step = jump - 1; step <= jump + 1; ++step) {
                    if (step > 0) {
                        int next_pos = pos + step;
                        if (next_pos == target) {
                            return true;
                        }
                        if (dp.find(next_pos) != dp.end()) {
                            dp[next_pos].insert(step);
                        }
                    }
                }
            }
        }

        return !dp[target].empty();
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

class Solution {
    public boolean canCross(int[] stones) {
        if (stones[1] != 1) return false;

        int n = stones.length;
        int target = stones[n - 1];

        Map<Integer, Set<Integer>> dp = new HashMap<>();
        for (int stone : stones) {
            dp.put(stone, new HashSet<>());
        }
        dp.get(0).add(0);

        for (int pos : stones) {
            Set<Integer> jumps = dp.get(pos);
            for (int jump : jumps) {
                for (int step = jump - 1; step <= jump + 1; step++) {
                    if (step > 0) {
                        int nextPos = pos + step;
                        if (nextPos == target) {
                            return true;
                        }
                        if (dp.containsKey(nextPos)) {
                            dp.get(nextPos).add(step);
                        }
                    }
                }
            }
        }

        return !dp.get(target).isEmpty();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^2)$  
  Because maximum jump size after $i$ stones is at most $i$, the number of possible incoming jump lengths per stone is bounded by $N$. Across all $N$ stones, the total number of $(pos, jump)$ states is at most $\sum_{i=1}^N i = \mathcal{O}(N^2)$. Each state explores $3$ transitions in $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(N^2)$  
  The hash map holds $N$ keys, and the total number of stored jump entries across all sets is at most $\mathcal{O}(N^2)$.

---

### Takeaway Pattern & Interview Traps

1. **Early Return Optimization:**
   - As soon as any transition lands on `stones[-1]`, we can return `true` immediately without finishing iterations on subsequent stones.
2. **First Jump Constraint:**
   - Always verify `stones[1] == 1` before starting the loop. If the second stone is at position $2$ or greater, the frog can never even leave stone $0$.
3. **Sparse vs Dense Coordinates:**
   - Stone coordinates can be as large as $2^{31} - 1$, so an array-indexed DP by position (`dp[pos]`) will cause Out-Of-Memory. A hash map indexed by actual stone coordinates or 2D DP table over stone indices `dp[stone_idx][k]` is mandatory.