---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 491: Non-decreasing Subsequences"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - hash-table
  - amazon
  - google
---

# LeetCode 491: Non-decreasing Subsequences

**Target Companies:** Amazon, Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Backtracking / Subsequences / Local Frame Deduplication  

---

### Problem Statement

Given an integer array `nums`, return *all the different possible non-decreasing subsequences of the given array with at least two elements*. You may return the answer in **any order**.

A **subsequence** of an array is an array that can be derived from the array by deleting some or no elements without changing the order of the remaining elements.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` containing all unique non-decreasing subsequences of length $\ge 2$.
- **Constraints:**
  - $1 \le \text{nums.length} \le 15$
  - $-100 \le \text{nums}[i] \le 100$

---

### Key Idea & Intuition

- **Why Sorting `nums` is Strictly Prohibited:**
  - In combination and subset problems (e.g. Subsets II, Combination Sum II), sorting is used to bring duplicate values together.
  - **Here, we cannot sort!** The problem demands *subsequences* of the original array. Sorting changes relative element ordering and produces invalid subsequences.
- **Local Level Deduplication:**
  - If we cannot sort, how do we prevent duplicate subsequences when identical values appear (such as `[4, 6, 7, 7]`)?
  - At each recursive call frame, create a **local set** `used_in_this_level`.
  - When iterating $i$ from `start` to $N - 1$:
    - If `nums[i] < path[-1]`: continue (violates non-decreasing constraint).
    - If `nums[i] in used_in_this_level`: continue (violates uniqueness: this exact number was already chosen for this position in this branch).
    - Add `nums[i]` to `used_in_this_level`.
    - **Crucial Invariant:** Notice that `used_in_this_level` is **never restored or cleared during backtracking**! It remains persistent throughout the loop of the current frame to remember all values that were already used as candidates for the current position.
- **Node-Collecting Condition:**
  - Whenever `len(path) >= 2`, append a copy of `path` to `results`. Then, continue searching deeper to form longer subsequences.

---

### Solution Approach (Step-by-Step)

1. Initialize `results = []` and `path = []`.
2. Define `backtrack(start)`:
   - If `len(path) >= 2`:
     - Append `list(path)` to `results`.
   - Initialize a local set (or boolean lookup table) `used_in_this_level = set()`.
   - For `i` from `start` to `len(nums) - 1`:
     - If `path` is not empty and `nums[i] < path[-1]`:
       - Continue.
     - If `nums[i]` in `used_in_this_level`:
       - Continue (skip duplicate at this depth).
     - Record candidate in local set: `used_in_this_level.add(nums[i])`.
     - **Choose:** `path.append(nums[i])`
     - **Explore:** `backtrack(i + 1)`
     - **Backtrack:** `path.pop()`
3. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `nums = [4, 6, 7, 7]`:

```
Level 0: backtrack(start=0, path=[])
└── i=0, num=4: path=[4]
    └── Level 1: backtrack(start=1, path=[4]), used_in_level={}
        ├── i=1, num=6: path=[4, 6] -> Record [4, 6]
        │   └── Level 2: backtrack(start=2, path=[4, 6]), used_in_level={}
        │       ├── i=2, num=7: path=[4, 6, 7] -> Record [4, 6, 7]
        │       │   └── Level 3: backtrack(start=3, path=[4, 6, 7]), used_in_level={}
        │       │       └── i=3, num=7: path=[4, 6, 7, 7] -> Record [4, 6, 7, 7]
        │       └── i=3, num=7: SKIPPED! (7 already in used_in_level at Level 2!)
        │
        ├── i=2, num=7: path=[4, 7] -> Record [4, 7]
        │   └── Level 2: backtrack(start=3, path=[4, 7])
        │       └── i=3, num=7: path=[4, 7, 7] -> Record [4, 7, 7]
        │
        └── i=3, num=7: SKIPPED! (7 already in used_in_level at Level 1!)
```

Notice how both duplicates of `7` at the same tree level are safely skipped without needing to sort the input array.

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | Constraints Checked | Output Size | Output Subsequences |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[4, 6, 7, 7]` | Preserves original order | `8` | `[[4,6],[4,7],[4,6,7],[4,6,7,7],[6,7],[6,7,7],[7,7],[4,7,7]]` |
| **Decreasing Order** | `[4, 3, 2, 1]` | No 2 elements non-decreasing | `0` | `[]` |
| **All Identical** | `[1, 1, 1]` | Non-decreasing ($1 \le 1$) | `3` | `[[1,1], [1,1,1], [1,1]]` $\implies$ unique: `[[1,1],[1,1,1]]` |
| **Two Elements** | `[1, 2]` | Single valid pair | `1` | `[[1, 2]]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        """
        Finds all unique non-decreasing subsequences with length >= 2.
        Uses local call-frame sets for deduplication without altering original array order.
        """
        results: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            if len(path) >= 2:
                results.append(list(path))

            used_in_this_level = set()

            for i in range(start, len(nums)):
                # Non-decreasing constraint
                if path and nums[i] < path[-1]:
                    continue

                # Local frame duplicate check
                if nums[i] in used_in_this_level:
                    continue

                used_in_this_level.add(nums[i])
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()  # Backtrack

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <vector>
#include <unordered_set>

class Solution {
public:
    std::vector<std::vector<int>> findSubsequences(std::vector<int>& nums) {
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        backtrack(0, nums, path, results);
        return results;
    }

private:
    void backtrack(int start, const std::vector<int>& nums,
                   std::vector<int>& path,
                   std::vector<std::vector<int>>& results) {
        if (path.size() >= 2) {
            results.push_back(path);
        }

        // Use boolean array for fast O(1) deduplication mapped from [-100, 100]
        bool used[201] = {false};

        for (size_t i = start; i < nums.size(); ++i) {
            if (!path.empty() && nums[i] < path.back()) {
                continue;
            }

            int mapped_idx = nums[i] + 100;
            if (used[mapped_idx]) {
                continue;
            }

            used[mapped_idx] = true;
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
import java.util.List;

class Solution {
    public List<List<Integer>> findSubsequences(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(0, nums, path, results);
        return results;
    }

    private void backtrack(int start, int[] nums, List<Integer> path, List<List<Integer>> results) {
        if (path.size() >= 2) {
            results.add(new ArrayList<>(path));
        }

        // Numbers range from -100 to 100 -> offset by 100
        boolean[] used = new boolean[201];

        for (int i = start; i < nums.length; i++) {
            if (!path.isEmpty() && nums[i] < path.get(path.size() - 1)) {
                continue;
            }

            int mappedIdx = nums[i] + 100;
            if (used[mappedIdx]) {
                continue;
            }

            used[mappedIdx] = true;
            path.add(nums[i]);
            backtrack(i + 1, nums, path, results);
            path.remove(path.size() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(2^N \cdot N)$.
  - There are $2^N$ possible subsequences in an array of length $N$.
  - For each valid subsequence of length $\le N$, copying into results takes $\mathcal{O}(N)$ time.
  - Since $N \le 15$, $2^{15} \times 15 = 32,768 \times 15 \approx 4.9 \times 10^5$ operations, completing in $< 5 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and `path` buffer (plus local tracking array per stack frame of depth $\le 15$).

---

### Takeaway Pattern & Interview Traps

- **Subsequence vs Subset Ordering:** When the problem says "subsequence", you cannot reorder the array. Do not sort `nums`!
- **Local Set Scoping:** In previous problems, we used `nums[i] == nums[i - 1]` because identical numbers were guaranteed to be adjacent. Without sorting, identical numbers can appear anywhere later in the array. A local set scoped to the current stack frame captures all duplicate values at that depth without polluting other branches.