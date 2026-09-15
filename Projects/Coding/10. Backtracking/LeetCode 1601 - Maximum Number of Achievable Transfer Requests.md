---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1601: Maximum Number of Achievable Transfer Requests"
tags:
  - leetcode
  - coding
  - backtracking
  - bit-manipulation
  - graph
  - amazon
  - google
---

# LeetCode 1601: Maximum Number of Achievable Transfer Requests

**Target Companies:** Amazon, Google, Uber  
**Difficulty:** Hard  
**Topic:** Backtracking / State-Space Search / Net Balance Invariant  

---

### Problem Statement

We have `n` buildings numbered from `0` to `n - 1`. Each building has a number of employees. It's transfer season, and some employees want to change the building they reside in.

You are given an array `requests` where `requests[i] = [from_i, to_i]` represents an employee's request to transfer from building `from_i` to building `to_i`.

All buildings are full, so a list of requests is achievable only if for each building, the net change in employee count is zero. This means the number of employees leaving is equal to the number of employees entering each building. More formally, let $k$ be the number of requests in a subset. The subset is achievable if and only if:
$$\text{count}(\text{leaves}_b) = \text{count}(\text{enters}_b) \quad \forall b \in [0, n - 1]$$

Return *the maximum number of achievable requests*.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `requests: List[List[int]]`
- **Output:** `int` (maximum size of a valid subset of requests)
- **Constraints:**
  - $1 \le n \le 20$
  - $1 \le \text{requests.length} \le 16$
  - $0 \le \text{from}_i, \text{to}_i < n$

---

### Key Idea & Intuition

- **Constraint Analysis ($M \le 16$):**
  - The number of requests $M \le 16$ is extremely small. The total number of subsets of requests is $2^{16} = 65,536$.
  - This immediately indicates that exhaustive backtracking or bitmask iteration over all $2^M$ subsets is fully viable.
- **The Net Balance Invariant:**
  - Track a balance array `delta` of size $n$, where:
    - When request `[u, v]` is accepted: `delta[u] -= 1` and `delta[v] += 1`.
  - A subset of requests is valid if and only if `delta[i] == 0` for all $0 \le i < n$.
- **Backtracking with Branch-and-Bound Pruning:**
  - For each request index $i$, we have two choices:
    1. **Include** request $i$: decrement `delta[from]`, increment `delta[to]`, increment `count`.
    2. **Exclude** request $i$: skip without modifying `delta`.
  - **Pruning Invariant:** If `count + (total_requests - i) <= max_achieved`, even if we accept every remaining request, we cannot beat our current best answer. Prune this branch immediately!
  - When all requests have been considered ($i == M$), if all elements in `delta` are zero, update `max_achieved = max(max_achieved, count)`.

---

### Solution Approach (Step-by-Step)

1. Initialize `max_achieved = 0` and `delta = [0] * n`.
2. Define recursive function `dfs(idx, count)`:
   - **Pruning:** If `count + (len(requests) - idx) <= max_achieved`, return.
   - **Base Case:** If `idx == len(requests)`:
     - Check if all values in `delta` are $0$.
     - If yes, update `max_achieved = max(max_achieved, count)`.
     - Return.
3. **Choice 1: Include `requests[idx]`:**
   - Let `u, v = requests[idx]`.
   - `delta[u] -= 1; delta[v] += 1`
   - `dfs(idx + 1, count + 1)`
   - `delta[u] += 1; delta[v] -= 1` (backtrack)
4. **Choice 2: Exclude `requests[idx]`:**
   - `dfs(idx + 1, count)`
5. Return `max_achieved`.

---

### Visual Algorithm Walkthrough

Let `n = 3`, `requests = [[0,1], [1,0], [0,1], [1,2], [2,0]]`.

```
Decision Tree (idx=0..4):
                      root (delta=[0,0,0], count=0)
                     /                             \
          Pick [0,1]: delta=[-1,+1,0]           Skip [0,1]: delta=[0,0,0]
                 /            \                             ...
        Pick [1,0]:          Skip [1,0]:
      delta=[0,0,0]         delta=[-1,+1,0]
        count=2
         /     \
     Pick [0,1] Skip [0,1]
    ...          ...
                 |
     Pick [1,2]: delta=[-1, 0, +1]
     Pick [2,0]: delta=[ 0, 0,  0], count = 5!
     All delta == 0 -> max_achieved = 5!
```

Self-loops like `[0,0]` can always be included because `delta[0] -= 1` and `delta[0] += 1` cancel out immediately.

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | `requests` | Achievable Subset Size | Valid Cycle / Transfers |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `5` | `[[0,1],[1,0],[0,1],[1,2],[2,0],[3,4]]` | `5` | `[0,1],[1,0],[0,1],[1,2],[2,0]` (balanced) |
| **Example 2** | `3` | `[[0,0],[1,2],[2,1]]` | `3` | `[0,0]` (self loop) + `[1,2],[2,1]` (swap) |
| **Example 3** | `4` | `[[0,3],[3,1],[1,2],[2,0]]` | `4` | Directed 4-cycle `0->3->1->2->0` |
| **No Moves Possible** | `3` | `[[0,1],[1,2]]` | `0` | Chain cannot close back to `0` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        """
        Finds the maximum number of requests that leave each building's net employee change at 0.
        Uses branch-and-bound backtracking with upper-bound pruning.
        """
        delta = [0] * n
        max_achieved = 0
        m = len(requests)

        def dfs(idx: int, count: int) -> None:
            nonlocal max_achieved
            # Prune: even if we pick all remaining requests, we cannot beat max_achieved
            if count + (m - idx) <= max_achieved:
                return

            if idx == m:
                # Validate net balance
                if all(d == 0 for d in delta):
                    max_achieved = count
                return

            u, v = requests[idx]

            # Option 1: Include this request
            delta[u] -= 1
            delta[v] += 1
            dfs(idx + 1, count + 1)
            delta[u] += 1
            delta[v] -= 1  # backtrack

            # Option 2: Exclude this request
            dfs(idx + 1, count)

        dfs(0, 0)
        return max_achieved
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maximumRequests(int n, const std::vector<std::vector<int>>& requests) {
        std::vector<int> delta(n, 0);
        int max_achieved = 0;
        int m = static_cast<int>(requests.size());

        dfs(0, 0, m, delta, requests, max_achieved);
        return max_achieved;
    }

private:
    void dfs(int idx, int count, int m, std::vector<int>& delta,
             const std::vector<std::vector<int>>& requests, int& max_achieved) {
        // Upper bound pruning
        if (count + (m - idx) <= max_achieved) {
            return;
        }

        if (idx == m) {
            bool all_zero = true;
            for (int val : delta) {
                if (val != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) {
                max_achieved = count;
            }
            return;
        }

        int u = requests[idx][0];
        int v = requests[idx][1];

        // Choice 1: Include request
        delta[u]--;
        delta[v]++;
        dfs(idx + 1, count + 1, m, delta, requests, max_achieved);
        delta[u]++;
        delta[v]--; // backtrack

        // Choice 2: Exclude request
        dfs(idx + 1, count, m, delta, requests, max_achieved);
    }
};
```

#### Java
```java
class Solution {
    private int maxAchieved = 0;

    public int maximumRequests(int n, int[][] requests) {
        int[] delta = new int[n];
        maxAchieved = 0;
        dfs(0, 0, requests, delta);
        return maxAchieved;
    }

    private void dfs(int idx, int count, int[][] requests, int[] delta) {
        int m = requests.length;
        // Upper bound pruning
        if (count + (m - idx) <= maxAchieved) {
            return;
        }

        if (idx == m) {
            boolean allZero = true;
            for (int d : delta) {
                if (d != 0) {
                    allZero = false;
                    break;
                }
            }
            if (allZero) {
                maxAchieved = count;
            }
            return;
        }

        int u = requests[idx][0];
        int v = requests[idx][1];

        // Option 1: Take request
        delta[u]--;
        delta[v]++;
        dfs(idx + 1, count + 1, requests, delta);
        delta[u]++;
        delta[v]--; // backtrack

        // Option 2: Skip request
        dfs(idx + 1, count, requests, delta);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^M \cdot N)$ in the worst case, but significantly reduced in practice due to the branch-and-bound pruning condition `count + (m - idx) <= max_achieved`.
  - For $M \le 16$, $2^{16} = 65,536$.
  - Checking the balance array takes $\mathcal{O}(N)$ where $N \le 20$.
  - Total worst-case operations: $65,536 \times 20 \approx 1.3 \times 10^6 \ll 10^8$, executing in $< 15 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N + M)$ auxiliary space.
  - The recursion stack reaches at most depth $M \le 16$.
  - The `delta` balance array requires $\mathcal{O}(N)$ memory.

---

### Takeaway Pattern & Interview Traps

- **Eulerian Subgraph / Net Flow Equivalence:** An achievable set of transfers corresponds to a union of directed cycles in the employee transfer graph. Instead of trying to detect cycles explicitly, simply maintaining $\text{in-degree} - \text{out-degree} == 0$ for all vertices is both necessary and sufficient.
- **Branch and Bound Pruning:** Whenever searching for a maximum subset, tracking `current + remaining <= best` cuts off huge subtrees that cannot possibly improve the global optimum.
- **Handling Self-Transfers:** If a request has `from == to`, it contributes $+1$ to achievable requests without changing `delta`! Including it is always non-detrimental.