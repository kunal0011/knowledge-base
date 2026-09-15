---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 47: Permutations II"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - permutation
  - amazon
  - google
---

# LeetCode 47: Permutations II

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Permutations with Duplicates / Sibling Pruning  

---

### Problem Statement

Given a collection of numbers, `nums`, that might contain duplicates, return *all possible unique permutations in **any order***.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` containing all unique permutations.
- **Constraints:**
  - $1 \le \text{nums.length} \le 8$
  - $-10 \le \text{nums}[i] \le 10$

---

### Key Idea & Intuition

- **The Challenge of Duplicates:**
  - When `nums` contains duplicate values (e.g. `[1, 1, 2]`), a standard permutation generator generates identical orderings multiple times (the two `'1'`s swap places, resulting in duplicated leaves).
- **The Canonical Duplicate-Pruning Condition:**
  - First, sort `nums` in non-decreasing order so identical values are adjacent.
  - Track visited elements using a boolean array `used[i]`.
  - In the loop over candidates $i \in [0, N - 1]$:
    1. If `used[i]`: continue.
    2. If `i > 0 and nums[i] == nums[i - 1] and not used[i - 1]`: **continue!**
- **First Principles: Why `not used[i - 1]`?**
  - **Case 1: `used[i - 1] == True`:**
    - `nums[i - 1]` was picked at an earlier position in the current permutation (vertical descent). Choosing `nums[i]` now is valid and necessary to form permutations like `[1, 1, 2]`.
  - **Case 2: `used[i - 1] == False`:**
    - `nums[i - 1]` was picked at the *same position* in a preceding branch, fully explored, and just un-picked (backtracked).
    - Choosing `nums[i]` (which has the same value) at this exact same position will explore an identical subtree of choices. Skipping it when `not used[i - 1]` completely eliminates redundant branches.

---

### Solution Approach (Step-by-Step)

1. **Sort `nums`** in ascending order.
2. Initialize `results = []`, `path = []`, and `used = [False] * len(nums)`.
3. Define `backtrack()`:
   - If `len(path) == len(nums)`:
     - Append a snapshot `list(path)` to `results`.
     - Return.
   - For `i` from `0` to `len(nums) - 1`:
     - If `used[i]`:
       - Continue.
     - If `i > 0` and `nums[i] == nums[i - 1]` and not `used[i - 1]`:
       - Continue (prune horizontal duplicate sibling).
     - **Choose:**
       - `used[i] = True`
       - `path.append(nums[i])`
     - **Explore:**
       - `backtrack()`
     - **Backtrack:**
       - `used[i] = False`
       - `path.pop()`
4. Call `backtrack()` and return `results`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 1, 2]` (sorted). Indices: `0: 1`, `1: 1`, `2: 2`.

```
Level 0: backtrack(path=[])
├── Pick index 0 (val=1): used=[T, F, F], path=[1]
│   ├── Pick index 1 (val=1, used[0] is T -> OK!): used=[T, T, F], path=[1, 1]
│   │   └── Pick index 2 (val=2): used=[T, T, T], path=[1, 1, 2] -> MATCH! [1, 1, 2]
│   └── Pick index 2 (val=2): used=[T, F, T], path=[1, 2]
│       └── Pick index 1 (val=1): used=[T, T, T], path=[1, 2, 1] -> MATCH! [1, 2, 1]
│
├── Pick index 1 (val=1):
│   Notice: nums[1] == nums[0], but used[0] is False!
│   PRUNED! Index 0 already explored all permutations starting with value 1 at this position.
│
└── Pick index 2 (val=2): used=[F, F, T], path=[2]
    ├── Pick index 0 (val=1): used=[T, F, T], path=[2, 1]
    │   └── Pick index 1 (val=1, used[0] is T -> OK!): path=[2, 1, 1] -> MATCH! [2, 1, 1]
    └── Pick index 1 (val=1, used[0] is F): PRUNED!
```

Final unique results: `[[1, 1, 2], [1, 2, 1], [2, 1, 1]]`.

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | Unique Permutation Formula | Count | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 1, 2]` | $\frac{3!}{2!1!} = 3$ | `3` | `[[1,1,2],[1,2,1],[2,1,1]]` |
| **All Duplicates** | `[1, 1, 1]` | $\frac{3!}{3!} = 1$ | `1` | `[[1, 1, 1]]` |
| **Two Pairs** | `[2, 2, 1, 1]` | $\frac{4!}{2!2!} = 6$ | `6` | 6 distinct permutations |
| **All Distinct** | `[1, 2, 3]` | $\frac{3!}{1!1!1!} = 6$ | `6` | 6 standard permutations |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        Generates all unique permutations of nums containing duplicates.
        Uses sorting and the `not used[i - 1]` condition to prune duplicate branches.
        """
        nums.sort()
        results: List[List[int]] = []
        path: List[int] = []
        used = [False] * len(nums)

        def backtrack() -> None:
            if len(path) == len(nums):
                results.append(list(path))
                return

            for i in range(len(nums)):
                if used[i]:
                    continue

                # Skip duplicate choice at the same tree depth
                if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    continue

                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()      # Backtrack
                used[i] = False

        backtrack()
        return results
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<int>> permuteUnique(std::vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        std::vector<std::vector<int>> results;
        std::vector<int> path;
        std::vector<bool> used(nums.size(), false);

        backtrack(nums, used, path, results);
        return results;
    }

private:
    void backtrack(const std::vector<int>& nums, std::vector<bool>& used,
                   std::vector<int>& path, std::vector<std::vector<int>>& results) {
        if (path.size() == nums.size()) {
            results.push_back(path);
            return;
        }

        for (size_t i = 0; i < nums.size(); ++i) {
            if (used[i]) continue;

            // Prune duplicate sibling
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }

            used[i] = true;
            path.push_back(nums[i]);
            backtrack(nums, used, path, results);
            path.pop_back(); // Backtrack
            used[i] = false;
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
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        boolean[] used = new boolean[nums.length];

        backtrack(nums, used, path, results);
        return results;
    }

    private void backtrack(int[] nums, boolean[] used, List<Integer> path, List<List<Integer>> results) {
        if (path.size() == nums.length) {
            results.add(new ArrayList<>(path));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;

            // Prune duplicate element at the same recursion level
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }

            used[i] = true;
            path.add(nums[i]);
            backtrack(nums, used, path, results);
            path.remove(path.size() - 1); // Backtrack
            used[i] = false;
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}\left(N \cdot \frac{N!}{\prod n_i!}\right)$.
  - The number of unique permutations is given by the multinomial coefficient $\frac{N!}{n_1! n_2! \dots n_k!}$.
  - For each unique permutation, copying the array of length $N$ takes $\mathcal{O}(N)$.
  - Pruning ensures that no redundant duplicate subtrees are explored.
  - For $N \le 8$, max combinations $\le 8! = 40,320$, executing in $< 20 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion stack, `path` buffer, and `used` boolean array.

---

### Takeaway Pattern & Interview Traps

- **`used[i - 1]` vs `!used[i - 1]`:**
  - Technically, both conditions work to eliminate duplicates, but `!used[i - 1]` is **drastically faster**!
  - `!used[i - 1]` prunes horizontal branches at the earliest opportunity, whereas `used[i - 1]` prunes at the leaves, resulting in significantly more redundant recursive calls.
- **Sorting is Mandatory:** The skip condition `nums[i] == nums[i - 1]` assumes identical numbers are strictly contiguous. Always sort `nums` before invoking backtracking.