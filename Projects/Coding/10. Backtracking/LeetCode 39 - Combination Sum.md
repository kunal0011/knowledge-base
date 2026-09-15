---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 39: Combination Sum"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - combination
  - amazon
  - google
---

# LeetCode 39: Combination Sum

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Unbounded Combinations / Pruning  

---

### Problem Statement

Given an array of **distinct** integers `candidates` and a target integer `target`, return a list of all **unique combinations** of `candidates` where the chosen numbers sum to `target`. You may return the combinations in **any order**.

The **same** number may be chosen from `candidates` an **unlimited number of times**. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to `target` is less than `150` combinations for the given input.

---

### Input & Output Formats & Constraints

- **Input:** `candidates: List[int]`, `target: int`
- **Output:** `List[List[int]]`
- **Constraints:**
  - $1 \le \text{candidates.length} \le 30$
  - $2 \le \text{candidates}[i] \le 40$
  - All elements of `candidates` are **distinct**.
  - $1 \le \text{target} \le 40$

---

### Key Idea & Intuition

- **Unbounded Combinations with Monotonic Index Selection:**
  - Because each number can be reused indefinitely, whenever we choose candidate `candidates[i]`, the next recursive step can pick from index `i` onwards (not `i + 1`).
  - To prevent permutations of the same combination (e.g. `[2, 2, 3]` vs `[3, 2, 2]`), we enforce that candidates are chosen in non-decreasing order of their indices. Once we move to index `i + 1`, we never look back at elements before `i + 1`.
- **Aggressive Pruning by Sorting:**
  - Sort `candidates` ascending before starting backtracking.
  - While exploring candidates at index `i`, if `candidates[i] > remaining_target`, we can immediately **break** the loop! Since the array is sorted, every subsequent candidate $j > i$ will also be greater than `remaining_target`.
  - This cuts off massive invalid subtrees and accelerates execution significantly.

---

### Solution Approach (Step-by-Step)

1. **Sort `candidates`** in ascending order.
2. Initialize `results = []` and `path = []`.
3. Define `backtrack(start_idx, remaining)`:
   - If `remaining == 0`:
     - Append a snapshot `list(path)` to `results`.
     - Return.
   - For `i` from `start_idx` to `len(candidates) - 1`:
     - If `candidates[i] > remaining`:
       - **Break** (pruning: all subsequent candidates are even larger).
     - **Choose:** `path.append(candidates[i])`
     - **Explore:** `backtrack(i, remaining - candidates[i])` (re-passing `i` allows reuse).
     - **Backtrack:** `path.pop()`
4. Call `backtrack(0, target)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `candidates = [2, 3, 6, 7]`, `target = 7`.
Sorted array: `[2, 3, 6, 7]`.

```
                             backtrack(start=0, rem=7, [])
                 /              |            \          \
             Pick 2           Pick 3        Pick 6      Pick 7
          rem=5, [2]        rem=4, [3]    rem=1, [6]   rem=0, [7]
          /    \    \          /    \          |          |
        Pick 2 Pick 3 Pick 6  Pick 3 Pick 6   (6>1,brk)  MATCH! [7]
       rem=3   rem=2  (6>5)   rem=1   (6>4)
      [2,2]    [2,3]          [3,3]
      /   \      |              |
   Pick 2 Pick 3 Pick 3      (3>1,brk)
   rem=1  rem=0  (3>2,brk)
  [2,2,2] [2,2,3]
    |       |
 (2>1)   MATCH! [2,2,3]
```

Final combinations: `[[2, 2, 3], [7]]`.

---

### Solved Examples with Multiple Inputs

| Test Case | `candidates` | `target` | Tree Matches | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[2, 3, 6, 7]` | `7` | `2+2+3=7`, `7=7` | `[[2, 2, 3], [7]]` |
| **Reuse Multiple** | `[2, 3, 5]` | `8` | `2+2+2+2`, `2+3+3`, `3+5` | `[[2,2,2,2],[2,3,3],[3,5]]` |
| **No Solution** | `[2]` | `1` | Smallest candidate $> 1$ | `[]` |
| **Single Match** | `[1]` | `2` | `1+1` | `[[1, 1]]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Finds all unique combinations in candidates where numbers sum to target.
        Numbers can be chosen unlimited times.
        """
        candidates.sort()
        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start_idx: int, remaining: int) -> None:
            if remaining == 0:
                results.append(list(path))
                return

            for i in range(start_idx, len(candidates)):
                candidate = candidates[i]
                # Prune: array is sorted, so subsequent candidates will also exceed remaining
                if candidate > remaining:
                    break

                path.append(candidate)
                # i is passed instead of i + 1 to permit unlimited reuse
                backtrack(i, remaining - candidate)
                path.pop()  # Backtrack

        backtrack(0, target)
        return results
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> combinationSum(std::vector<int>& candidates, int target) {
        std::sort(candidates.begin(), candidates.end());
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        backtrack(0, target, candidates, path, results);
        return results;
    }

private:
    void backtrack(int start_idx, int remaining, const std::vector<int>& candidates,
                   std::vector<int>& path, std::vector<std::vector<int>>& results) {
        if (remaining == 0) {
            results.push_back(path);
            return;
        }

        for (size_t i = start_idx; i < candidates.size(); ++i) {
            if (candidates[i] > remaining) {
                break; // Prune
            }

            path.push_back(candidates[i]);
            // Reuse current element: pass i
            backtrack(i, remaining - candidates[i], candidates, path, results);
            path.pop_back(); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(0, target, candidates, path, results);
        return results;
    }

    private void backtrack(int startIdx, int remaining, int[] candidates,
                          List<Integer> path, List<List<Integer>> results) {
        if (remaining == 0) {
            results.add(new ArrayList<>(path));
            return;
        }

        for (int i = startIdx; i < candidates.length; i++) {
            if (candidates[i] > remaining) {
                break; // Prune
            }

            path.add(candidates[i]);
            // Reuse current element: pass i
            backtrack(i, remaining - candidates[i], candidates, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^{T/M})$ loose upper bound, where $N = \text{len(candidates)}$, $T = \text{target}$, and $M = \min(\text{candidates})$.
  - The maximum depth of the recursion tree is $T / M$.
  - With $T \le 40$ and $M \ge 2$, the max depth is $20$.
  - Sorting and early breaking drastically prunes the branches, guaranteeing $< 5 \text{ ms}$ execution for all LeetCode test cases.
- **Space Complexity:** $\mathcal{O}(T / M)$ auxiliary space for the recursion call stack and `path` vector.

---

### Takeaway Pattern & Interview Traps

- **Reuse vs Non-Reuse:** In Combination Sum I, elements can be reused, so recursive call passes `i`. In Combination Sum II, elements cannot be reused, so recursive call passes `i + 1`.
- **Sort and Break Pruning:** Always sort the input before backtracking on sum problems. Changing `if (val > rem) continue` to `break` cuts off entire loops of fruitless candidate evaluations.