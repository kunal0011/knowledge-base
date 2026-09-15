---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 40: Combination Sum II"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - combination
  - amazon
  - google
---

# LeetCode 40: Combination Sum II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Subsets with Duplicates / Level-Wise Pruning  

---

### Problem Statement

Given a collection of candidate numbers (`candidates`) and a target number (`target`), find all unique combinations in `candidates` where the candidate numbers sum to `target`.

Each number in `candidates` may only be used **once** in the combination.

**Note:** The solution set must not contain duplicate combinations.

---

### Input & Output Formats & Constraints

- **Input:** `candidates: List[int]`, `target: int`
- **Output:** `List[List[int]]` containing all unique combinations.
- **Constraints:**
  - $1 \le \text{candidates.length} \le 100$
  - $1 \le \text{candidates}[i] \le 50$
  - $1 \le \text{target} \le 30$

---

### Key Idea & Intuition

- **Key Differences from Combination Sum I (LeetCode 39):**
  1. In LC 39, numbers were distinct and could be reused indefinitely.
  2. In LC 40, numbers **contain duplicates** and each element index can be used **at most once**.
- **The Duplicate Combination Dilemma:**
  - If `candidates = [1, 1, 2]` and `target = 3`:
    - Picking the 1st `1` and `2` gives `[1, 2]`.
    - Picking the 2nd `1` and `2` also gives `[1, 2]`.
  - Storing in a hash set to deduplicate is wasteful and slow.
- **Level-Wise Duplicate Pruning:**
  - First, sort `candidates` ascending so identical numbers are contiguous.
  - In the loop `for i in range(start, len(candidates))`:
    - If `i > start` and `candidates[i] == candidates[i - 1]`: **continue!**
  - **The Invariant:**
    - `i == start`: This is the *first* time we are considering this duplicate value at this depth level. We allow it to be chosen (this allows combinations with duplicate elements, e.g. `[1, 1, 6]`).
    - `i > start`: This value is identical to a candidate we already fully explored at the *same breadth level*. Choosing it again would generate an identical subtree. Skipping it guarantees 100% uniqueness without sets.
- **Pruning via Sorted Order:**
  - Since `candidates` is sorted, if `candidates[i] > remaining`, we can `break` immediately.

---

### Solution Approach (Step-by-Step)

1. **Sort `candidates`** ascending.
2. Initialize `results = []` and `path = []`.
3. Define `backtrack(start, remaining)`:
   - If `remaining == 0`:
     - Append a copy of `path` to `results`.
     - Return.
   - For `i` from `start` to `len(candidates) - 1`:
     - If `candidates[i] > remaining`:
       - **Break** (prune all larger values).
     - If `i > start` and `candidates[i] == candidates[i - 1]`:
       - **Continue** (skip duplicate choice at the current recursion level).
     - **Choose:** `path.append(candidates[i])`
     - **Explore:** `backtrack(i + 1, remaining - candidates[i])` (advance to `i + 1` since elements cannot be reused).
     - **Backtrack:** `path.pop()`
4. Call `backtrack(0, target)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `candidates = [10, 1, 2, 7, 6, 1, 5]`, `target = 8`.
Sorted: `[1, 1, 2, 5, 6, 7, 10]`.

```
Level 0: backtrack(start=0, rem=8, path=[])
├── i=0, val=1 (first '1'):
│   └── Level 1: backtrack(start=1, rem=7, path=[1])
│       ├── i=1, val=1 (second '1' allowed at deeper depth!):
│       │   └── Level 2: backtrack(start=2, rem=6, path=[1, 1])
│       │       ├── i=4, val=6 -> MATCH! [1, 1, 6]
│       ├── i=2, val=2:
│       │   └── Level 2: backtrack(start=3, rem=5, path=[1, 2])
│       │       └── i=3, val=5 -> MATCH! [1, 2, 5]
│       └── i=5, val=7 -> MATCH! [1, 7]
│
├── i=1, val=1 (duplicate '1' at Level 0! i > start && val == prev):
│   SKIPPED! Prunes entire redundant subtree that would re-produce [1, 7], [1, 2, 5], etc.
│
├── i=2, val=2:
│   └── Level 1: backtrack(start=3, rem=6, path=[2])
│       └── i=4, val=6 -> MATCH! [2, 6]
│
└── i=5, val=7: rem=1 (candidates >= 10 break)
```

Final unique results: `[[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]`.

---

### Solved Examples with Multiple Inputs

| Test Case | `candidates` | `target` | Sorted Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[10,1,2,7,6,1,5]` | `8` | `[1,1,2,5,6,7,10]` | `[[1,1,6],[1,2,5],[1,7],[2,6]]` |
| **All Duplicates** | `[2,5,2,1,2]` | `5` | `[1,2,2,2,5]` | `[[1,2,2],[5]]` |
| **No Valid Combination** | `[3, 5]` | `2` | `[3, 5]` | `[]` |
| **Exact Single Element** | `[1]` | `1` | `[1]` | `[[1]]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        Finds all unique combinations summing to target where each candidate is used at most once.
        Prunes horizontal duplicates using sorting and the `i > start` condition.
        """
        candidates.sort()
        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                results.append(list(path))
                return

            for i in range(start, len(candidates)):
                # Prune: array is sorted; subsequent candidates exceed remaining
                if candidates[i] > remaining:
                    break

                # Skip duplicates at the same recursion level
                if i > start and candidates[i] == candidates[i - 1]:
                    continue

                path.append(candidates[i])
                # i + 1 ensures each candidate is used at most once
                backtrack(i + 1, remaining - candidates[i])
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
    std::vector<std::vector<int>> combinationSum2(std::vector<int>& candidates, int target) {
        std::sort(candidates.begin(), candidates.end());
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        backtrack(0, target, candidates, path, results);
        return results;
    }

private:
    void backtrack(int start, int remaining, const std::vector<int>& candidates,
                   std::vector<int>& path, std::vector<std::vector<int>>& results) {
        if (remaining == 0) {
            results.push_back(path);
            return;
        }

        for (size_t i = start; i < candidates.size(); ++i) {
            if (candidates[i] > remaining) {
                break; // Prune all larger numbers
            }

            // Skip duplicate choices at the same recursion depth
            if (i > static_cast<size_t>(start) && candidates[i] == candidates[i - 1]) {
                continue;
            }

            path.push_back(candidates[i]);
            // Move to i + 1 (cannot reuse current candidate)
            backtrack(i + 1, remaining - candidates[i], candidates, path, results);
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
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(0, target, candidates, path, results);
        return results;
    }

    private void backtrack(int start, int remaining, int[] candidates,
                          List<Integer> path, List<List<Integer>> results) {
        if (remaining == 0) {
            results.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < candidates.length; i++) {
            if (candidates[i] > remaining) {
                break; // Prune
            }

            // Skip duplicate elements at the same tree depth
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }

            path.add(candidates[i]);
            // Advance to i + 1
            backtrack(i + 1, remaining - candidates[i], candidates, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^N)$ in the worst case (the power set of $N$ items).
  - Sorting takes $\mathcal{O}(N \log N)$.
  - Duplicate skipping and sum pruning significantly restrict the searched space, keeping practical runtime under $4 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and `path` buffer (excluding the returned list of results).

---

### Takeaway Pattern & Interview Traps

- **The `i > start` Duplicate Filter Invariant:** This is the universal template for handling duplicate elements across Subsets II, Combination Sum II, and Permutations II.
  - `i == start`: Vertical descent in the tree (allows identical values across different positions in the path).
  - `i > start`: Horizontal branching across siblings (forbids choosing the same value twice at the same position).
- **Advance Index to `i + 1`:** Passing `i + 1` ensures each index is used at most once, whereas passing `i` allows infinite reuse (as in LC 39).