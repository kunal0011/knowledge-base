---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 90: Subsets II"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 90: Subsets II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Subsets with Duplicates / Sibling Pruning  

---

### Problem Statement

Given an integer array `nums` that may contain duplicates, return *all possible subsets (the power set)*.

The solution set **must not contain duplicate subsets**. Return the solution in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` containing all unique subsets.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10$
  - $-10 \le \text{nums}[i] \le 10$

---

### Key Idea & Intuition

- **The Duplicate Dilemma:**
  - When `nums` contains duplicate values (e.g. `[1, 2, 2]`), naive power-set generation produces identical subsets (e.g. picking the 1st `'2'` yields `[1, 2]`, and picking the 2nd `'2'` yields `[1, 2]`).
- **The Horizontal Pruning Invariant (`i > start`):**
  - First, sort `nums` in ascending order so duplicate numbers become adjacent.
  - In the recursive function `backtrack(start)`:
    - Loop `i` from `start` to $N - 1$:
      - If `i > start and nums[i] == nums[i - 1]`: **continue!**
  - **First Principles Rationale:**
    - `i == start`: We are making the *first* decision at this recursion depth. Choosing duplicate value `nums[i]` is valid because it represents vertical tree descent (e.g., forming `[2, 2]`).
    - `i > start`: We already explored an entire subtree starting with this exact same value at the *same position* during an earlier iteration of the loop. Exploring it again would generate an identical family of subsets. Skipping it eliminates all duplicates without hash sets.

---

### Solution Approach (Step-by-Step)

1. **Sort `nums`** in ascending order.
2. Initialize `results = []` and `path = []`.
3. Define `backtrack(start)`:
   - **Record current node:** `results.append(list(path))` (every node is a valid subset).
   - Loop `i` from `start` to `len(nums) - 1`:
     - If `i > start` and `nums[i] == nums[i - 1]`:
       - Continue (skip duplicate sibling branch).
     - **Choose:** `path.append(nums[i])`
     - **Explore:** `backtrack(i + 1)`
     - **Backtrack:** `path.pop()`
4. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 2, 2]` (sorted).

```
Level 0: backtrack(start=0, path=[]) -> Record []
├── i=0, num=1: path=[1] -> Record [1]
│   └── Level 1: backtrack(start=1, path=[1])
│       ├── i=1, num=2 (first '2'): path=[1, 2] -> Record [1, 2]
│       │   └── Level 2: backtrack(start=2, path=[1, 2])
│       │       └── i=2, num=2: path=[1, 2, 2] -> Record [1, 2, 2]
│       └── i=2, num=2: SKIPPED! (i > start && nums[2] == nums[1])
│           Prunes duplicate [1, 2]!
│
├── i=1, num=2 (first '2' at Level 0): path=[2] -> Record [2]
│   └── Level 1: backtrack(start=2, path=[2])
│       └── i=2, num=2: path=[2, 2] -> Record [2, 2]
│
└── i=2, num=2: SKIPPED! (i > start && nums[2] == nums[1])
    Prunes duplicate [2]!

Unique Subsets: [], [1], [1, 2], [1, 2, 2], [2], [2, 2].
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | Sorted Input | Unique Subsets | Output Count |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 2, 2]` | `[1, 2, 2]` | `[], [1], [1,2], [1,2,2], [2], [2,2]` | `6` |
| **All Identical** | `[0, 0, 0]` | `[0, 0, 0]` | `[], [0], [0,0], [0,0,0]` | `4` |
| **No Duplicates** | `[4, 1, 0]` | `[0, 1, 4]` | Full power set $2^3$ | `8` |
| **Single Element** | `[1]` | `[1]` | `[], [1]` | `2` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        Returns all unique subsets of nums containing duplicates.
        Uses sorting and the `i > start` horizontal duplicate filter.
        """
        nums.sort()
        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            # Record subset at current node
            results.append(list(path))

            for i in range(start, len(nums)):
                # Skip duplicate choices at the same recursion depth
                if i > start and nums[i] == nums[i - 1]:
                    continue

                path.append(nums[i])
                backtrack(i + 1)
                path.pop()  # Backtrack

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> subsetsWithDup(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        backtrack(0, nums, path, results);
        return results;
    }

private:
    void backtrack(int start, const std::vector<int>& nums,
                   std::vector<int>& path,
                   std::vector<std::vector<int>>& results) {
        results.push_back(path);

        for (size_t i = start; i < nums.size(); ++i) {
            // Prune duplicate sibling branch
            if (i > static_cast<size_t>(start) && nums[i] == nums[i - 1]) {
                continue;
            }

            path.push_back(nums[i]);
            backtrack(i + 1, nums, path, results);
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
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(0, nums, path, results);
        return results;
    }

    private void backtrack(int start, int[] nums, List<Integer> path, List<List<Integer>> results) {
        results.add(new ArrayList<>(path));

        for (int i = start; i < nums.length; i++) {
            // Skip duplicate choices at the current tree level
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            path.add(nums[i]);
            backtrack(i + 1, nums, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot 2^N)$.
  - Sorting takes $\mathcal{O}(N \log N)$.
  - In the worst case (all elements distinct), exactly $2^N$ subsets are generated.
  - For each subset, copying the path takes $\mathcal{O}(N)$ time.
  - For $N \le 10$, $10 \times 1024 \approx 10^4$ operations, completing in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and `path` buffer (excluding the returned subsets list).

---

### Takeaway Pattern & Interview Traps

- **The `i > start` Filter:** Compare `i > start` with `nums[i] == nums[i - 1]`. Never check `i > 0` alone! Checking `i > 0` without `start` would prevent taking duplicate elements deeper in the recursion, erroneously skipping subsets like `[2, 2]`.
- **Node-Collecting Invariant:** In Subsets problems, `results.append(list(path))` happens unconditionally on every invocation of `backtrack()`, not just when `len(path) == n`.