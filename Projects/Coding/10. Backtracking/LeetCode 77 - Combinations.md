---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 77: Combinations"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - combination
  - amazon
  - google
---

# LeetCode 77: Combinations

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Mathematical Pruning / Combinations $\binom{n}{k}$  

---

### Problem Statement

Given two integers `n` and `k`, return *all possible combinations of `k` numbers chosen from the range `[1, n]`*.

You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `n: int`, `k: int`
- **Output:** `List[List[int]]` containing all $\binom{n}{k}$ combinations.
- **Constraints:**
  - $1 \le n \le 20$
  - $1 \le k \le n$

---

### Key Idea & Intuition

- **The Combinatorial Foundation:**
  - The problem asks for all subsets of size $k$ from the set $\{1, 2, \dots, n\}$.
  - The total number of valid outputs is given by the binomial coefficient:
    $$\binom{n}{k} = \frac{n!}{k!(n - k)!}$$
- **Enforcing Strictly Ascending Order:**
  - In a combination, order does not matter: `[1, 2]` is identical to `[2, 1]`.
  - By restricting each subsequent number chosen to be strictly greater than the previous number (`start_num = prev + 1`), we guarantee that each combination is generated in a unique canonical order, avoiding duplicate generation.
- **Aggressive Upper Bound Pruning:**
  - Suppose the current combination already has length `len(path)`.
  - We still need to pick $k - \text{len(path)}$ more numbers.
  - If we are currently considering candidate number $i$, the total remaining numbers available from $i$ to $n$ inclusive is $n - i + 1$.
  - For a valid combination to be possible:
    $$n - i + 1 \ge k - \text{len(path)} \implies i \le n - (k - \text{len(path)}) + 1$$
  - Restricting the loop bound to $n - (k - \text{len(path)}) + 1$ cuts off all branches that would run out of numbers before reaching size $k$.

---

### Solution Approach (Step-by-Step)

1. Initialize `results = []` and `path = []`.
2. Define `backtrack(start)`:
   - **Base Case:** If `len(path) == k`:
     - Append a snapshot `list(path)` to `results`.
     - Return.
   - **Pruned Range:** Loop `i` from `start` to $n - (k - \text{len}(path)) + 1$:
     - **Choose:** `path.append(i)`
     - **Explore:** `backtrack(i + 1)`
     - **Backtrack:** `path.pop()`
3. Call `backtrack(1)` and return `results`.

---

### Visual Algorithm Walkthrough

Let $n = 4, k = 2$.
Pruning bound formula: $i \le 4 - (2 - 0) + 1 = 3$.
At root, $i$ only loops from $1$ to $3$ (number 4 is pruned at root because picking 4 leaves 0 remaining numbers to fill the 2nd slot!).

```
                       backtrack(start=1, path=[])
                   /              |              \
             Pick 1             Pick 2          Pick 3
          path = [1]          path = [2]       path = [3]
          (i <= 4)            (i <= 4)         (i <= 4)
         /   |   \             /    \              |
      Pick 2 3    4         Pick 3   4           Pick 4
        |    |    |           |      |             |
      [1,2] [1,3] [1,4]     [2,3]  [2,4]         [3,4]

Total leaves = 6 combinations.
Number 4 is never picked as the first element!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `n` | `k` | Formula $\binom{n}{k}$ | Combinations Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `4` | `2` | $\binom{4}{2} = 6$ | `[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]` |
| **k == 1** | `4` | `1` | $\binom{4}{1} = 4$ | `[[1], [2], [3], [4]]` |
| **k == n** | `3` | `3` | $\binom{3}{3} = 1$ | `[[1, 2, 3]]` |
| **Max Values** | `20` | `10` | $\binom{20}{10} = 184,756$ | 184,756 combinations |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        """
        Generates all combinations of k numbers from [1, n].
        Uses upper-bound pruning to avoid dead branches.
        """
        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            if len(path) == k:
                results.append(list(path))
                return

            # Pruning bound: only iterate while enough numbers remain to fill k slots
            max_start = n - (k - len(path)) + 1
            for i in range(start, max_start + 1):
                path.append(i)
                backtrack(i + 1)
                path.pop()  # Backtrack

        backtrack(1)
        return results
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> combine(int n, int k) {
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        path.reserve(k);
        backtrack(1, n, k, path, results);
        return results;
    }

private:
    void backtrack(int start, int n, int k,
                   std::vector<int>& path,
                   std::vector<std::vector<int>>& results) {
        if (static_cast<int>(path.size()) == k) {
            results.push_back(path);
            return;
        }

        // Pruning: ensure enough elements remain to reach size k
        int max_start = n - (k - static_cast<int>(path.size())) + 1;
        for (int i = start; i <= max_start; ++i) {
            path.push_back(i);
            backtrack(i + 1, n, k, path, results);
            path.pop_back(); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>(k);
        backtrack(1, n, k, path, results);
        return results;
    }

    private void backtrack(int start, int n, int k, List<Integer> path, List<List<Integer>> results) {
        if (path.size() == k) {
            results.add(new ArrayList<>(path));
            return;
        }

        // Pruning: skip starting numbers where remaining count < required count
        int maxStart = n - (k - path.size()) + 1;
        for (int i = start; i <= maxStart; i++) {
            path.add(i);
            backtrack(i + 1, n, k, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left(\binom{n}{k} \cdot k\right)$.
  - There are exactly $\binom{n}{k}$ combinations generated.
  - At each leaf, copying the path of length $k$ takes $\mathcal{O}(k)$ time.
  - With the upper bound pruning condition $i \le n - (k - \text{len}) + 1$, zero invalid leaf nodes are explored.
- **Space Complexity:** $\mathcal{O}(k)$ auxiliary space for the recursion call stack and `path` vector.

---

### Takeaway Pattern & Interview Traps

- **The Mathematical Pruning Bound:**
  $$i \le n - (k - \text{len(path)}) + 1$$
  Mentioning and implementing this exact bound is the hallmark of a Senior/Staff candidate in combinatorics interviews.
- **Pre-allocating Buffer Capacity:** Reserve vector/list capacity to $k$ (`path.reserve(k)`) to prevent multiple reallocations as elements are pushed.