---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 216: Combination Sum III"
tags:
  - leetcode
  - coding
  - backtracking
  - combination
  - amazon
  - google
---

# LeetCode 216: Combination Sum III

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Combinations / Pruning  

---

### Problem Statement

Find all valid combinations of `k` numbers that sum up to `n` such that the following conditions are true:
- Only numbers `1` through `9` are used.
- Each number is used **at most once**.

Return *a list of all possible valid combinations*. The list must not contain the same combination twice, and the combinations may be returned in any order.

---

### Input & Output Formats & Constraints

- **Input:** `k: int`, `n: int`
- **Output:** `List[List[int]]` containing all unique $k$-element subsets summing to $n$.
- **Constraints:**
  - $2 \le k \le 9$
  - $1 \le n \le 60$

---

### Key Idea & Intuition

- **Fixed Finite Candidate Pool $\{1, 2, \dots, 9\}$:**
  - Because numbers are restricted to digits $1$ through $9$ and each number can be picked at most once, the total number of combinations is bounded by $\binom{9}{k} \le \binom{9}{4} = 126$.
- **Strictly Increasing Selection to Avoid Duplicates:**
  - To prevent permutations of the same subset (e.g., `[1, 2, 4]` vs `[4, 2, 1]`), elements must be chosen in strictly ascending order. When branching from digit `d`, only explore candidates $d + 1, \dots, 9$.
- **Pruning Invariants:**
  1. **Length Pruning:** If `len(path) == k` and `target == 0`, record the combination. If `len(path) == k` and `target != 0`, backtrack immediately.
  2. **Sum Exceeded:** If candidate digit `d > target`, break immediately (since candidates are strictly ascending, subsequent digits will also exceed `target`).
  3. **Insufficient Candidates:** If remaining available numbers $(9 - d + 1) < (k - \text{len}(path))$, we cannot possibly pick $k$ numbers. Prune this branch.

---

### Solution Approach (Step-by-Step)

1. Check trivial out-of-bounds:
   - Minimum possible sum of $k$ distinct digits is $\sum_{i=1}^k i = \frac{k(k+1)}{2}$.
   - Maximum possible sum of $k$ distinct digits is $\sum_{i=10-k}^9 i = \frac{k(19-k)}{2}$.
   - If $n < \text{min\_sum}$ or $n > \text{max\_sum}$, return `[]`.
2. Initialize `results = []` and `path = []`.
3. Define `backtrack(start_digit, remaining_target)`:
   - If `len(path) == k`:
     - If `remaining_target == 0`:
       - Append `list(path)` to `results`.
     - Return.
   - For `digit` from `start_digit` to $9$:
     - If `digit > remaining_target`:
       - Break (cannot form a valid sum with larger numbers).
     - If $(9 - digit + 1) < (k - \text{len}(path))$:
       - Break (not enough remaining digits to fill length $k$).
     - Choose: `path.append(digit)`
     - Explore: `backtrack(digit + 1, remaining_target - digit)`
     - Backtrack: `path.pop()`
4. Call `backtrack(1, n)` and return `results`.

---

### Visual Algorithm Walkthrough

Let $k = 3, n = 7$:

```
                             backtrack(start=1, target=7, path=[])
                   /                  |                  \
              Pick 1                Pick 2             Pick 3
          (target=6, [1])       (target=5, [2])    (target=4, [3])
          /            \              |                  |
       Pick 2        Pick 3         Pick 3             Pick 4
    (target=4, [1,2])  ...       (target=2, [2,3])    (4 > 4, break!)
       /        \                     |
    Pick 3    Pick 4               Pick 4
   (target=1) (target=0)          (4 > 2, break!)
      |          |
  (3 > 1)   len == 3, target == 0!
            Found: [1, 2, 4]

All other branches exceed target or run out of digits.
Final Output: [[1, 2, 4]]
```

---

### Solved Examples with Multiple Inputs

| Test Case | `k` | `n` | Valid Combinations | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `3` | `7` | `1 + 2 + 4 = 7` | `[[1, 2, 4]]` |
| **Multiple Answers** | `3` | `9` | `1+2+6`, `1+3+5`, `2+3+4` | `[[1,2,6],[1,3,5],[2,3,4]]` |
| **Impossible Sum (Too Large)** | `4` | `1` | Min sum is $1+2+3+4 = 10 > 1$ | `[]` |
| **Max Digit Sum** | `9` | `45` | $1+2+3+4+5+6+7+8+9 = 45$ | `[[1,2,3,4,5,6,7,8,9]]` |
| **Impossible Sum (Too Small)** | `2` | `18` | Max sum is $8 + 9 = 17 < 18$ | `[]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        """
        Finds all valid combinations of k numbers from 1 to 9 summing to n.
        Uses ascending branch backtracking with aggressive pruning.
        """
        # Quick boundary checks
        min_sum = k * (k + 1) // 2
        max_sum = k * (19 - k) // 2
        if n < min_sum or n > max_sum:
            return []

        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, target: int) -> None:
            if len(path) == k:
                if target == 0:
                    results.append(list(path))
                return

            for digit in range(start, 10):
                # Prune 1: digit exceeds remaining target
                if digit > target:
                    break
                # Prune 2: remaining available digits cannot fill k slots
                if (10 - digit) < (k - len(path)):
                    break

                path.append(digit)
                backtrack(digit + 1, target - digit)
                path.pop()  # Backtrack

        backtrack(1, n)
        return results
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> combinationSum3(int k, int n) {
        int min_sum = k * (k + 1) / 2;
        int max_sum = k * (19 - k) / 2;
        if (n < min_sum || n > max_sum) return {};

        std::vector<std::vector<int>> results;
        std::vector<int> path;
        backtrack(1, n, k, path, results);
        return results;
    }

private:
    void backtrack(int start, int target, int k,
                   std::vector<int>& path,
                   std::vector<std::vector<int>>& results) {
        if (static_cast<int>(path.size()) == k) {
            if (target == 0) {
                results.push_back(path);
            }
            return;
        }

        for (int digit = start; digit <= 9; ++digit) {
            if (digit > target) break;
            if ((10 - digit) < (k - static_cast<int>(path.size()))) break;

            path.push_back(digit);
            backtrack(digit + 1, target - digit, k, path, results);
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
    public List<List<Integer>> combinationSum3(int k, int n) {
        int minSum = k * (k + 1) / 2;
        int maxSum = k * (19 - k) / 2;
        if (n < minSum || n > maxSum) {
            return new ArrayList<>();
        }

        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(1, n, k, path, results);
        return results;
    }

    private void backtrack(int start, int target, int k,
                          List<Integer> path,
                          List<List<Integer>> results) {
        if (path.size() == k) {
            if (target == 0) {
                results.add(new ArrayList<>(path));
            }
            return;
        }

        for (int digit = start; digit <= 9; digit++) {
            if (digit > target) break;
            if ((10 - digit) < (k - path.size())) break;

            path.add(digit);
            backtrack(digit + 1, target - digit, k, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left(\binom{9}{k} \cdot k\right)$.
  - The maximum number of combinations of $k$ digits from 9 is $\binom{9}{k}$. The largest value is $\binom{9}{4} = \binom{9}{5} = 126$.
  - Copying each valid combination of length $k$ takes $\mathcal{O}(k)$ time.
  - Total operations are at most $126 \times 9 \approx 1134$, running in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(k)$ auxiliary space for the recursion stack and `path` list.

---

### Takeaway Pattern & Interview Traps

- **Ascending Index Invariant:** Enforcing `start = digit + 1` naturally prevents identical combinations with permuted order (such as `[1, 2, 4]` and `[4, 1, 2]`) without requiring hash sets.
- **Strict Break Conditions:** Since candidates are sorted ascending ($1 \dots 9$), the moment `digit > target`, we can `break` out of the loop rather than merely `continue`, cutting off all larger digits.