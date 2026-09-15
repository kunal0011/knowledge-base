---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 78: Subsets"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 78: Subsets

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg, Uber  
**Difficulty:** Medium  
**Topic:** Backtracking / Power Set / Bitmasking  

---

### Problem Statement

Given an integer array `nums` of **unique** elements, return *all possible subsets (the power set)*.

The solution set **must not contain duplicate subsets**. Return the solution in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` containing all $2^N$ subsets.
- **Constraints:**
  - $1 \le \text{nums.length} \le 10$
  - $-10 \le \text{nums}[i] \le 10$
  - All numbers of `nums` are **unique**.

---

### Key Idea & Intuition

- **The Power Set Size:**
  - For an array of $N$ distinct elements, each element has exactly two possibilities: either present in the subset or absent.
  - The total number of subsets is strictly:
    $$|\mathcal{P}(nums)| = 2^N$$
- **Node-Collecting DFS (The Combination Pattern):**
  - Unlike permutation problems where solutions exist only at the leaves of the recursion tree, in the power set, **every node in the recursion tree represents a valid subset**!
  - We append a snapshot of `path` to `results` at the very start of each call to `backtrack(start)`.
  - Then, we loop $i$ from `start` to $N - 1$:
    - Append `nums[i]`
    - Recurse on `i + 1`
    - Pop `nums[i]` (backtrack)
- **Alternative Perspective: Bitmasking:**
  - Each integer from $0$ to $2^N - 1$ represents a unique subset. The $j$-th bit of mask $m$ is $1$ if `nums[j]` is included.

---

### Solution Approach (Step-by-Step)

1. Initialize `results = []` and `path = []`.
2. Define `backtrack(start)`:
   - **Record current node:** `results.append(list(path))`.
   - Loop `i` from `start` to `len(nums) - 1`:
     - **Choose:** `path.append(nums[i])`
     - **Explore:** `backtrack(i + 1)`
     - **Backtrack:** `path.pop()`
3. Call `backtrack(0)`.
4. Return `results`.

---

### Visual Algorithm Walkthrough

For `nums = [1, 2, 3]`:

```
                       backtrack(start=0, path=[])
                              Record: []
                 /                 |                 \
           Pick 1 (start=1)   Pick 2 (start=2)   Pick 3 (start=3)
            path = [1]         path = [2]         path = [3]
            Record: [1]        Record: [2]        Record: [3]
            /        \             |
       Pick 2        Pick 3      Pick 3
      path=[1,2]    path=[1,3]  path=[2,3]
     Record:[1,2]  Record:[1,3] Record:[2,3]
          |
       Pick 3
     path=[1,2,3]
    Record:[1,2,3]

Total recorded subsets = 2^3 = 8:
[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3].
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | Power Set Size $2^N$ | Subsets Generated | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1, 2, 3]` | $2^3 = 8$ | `[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]` | 8 subsets |
| **Single Element** | `[0]` | $2^1 = 2$ | `[], [0]` | `[[], [0]]` |
| **Two Elements** | `[1, 2]` | $2^2 = 4$ | `[], [1], [2], [1, 2]` | `[[], [1], [2], [1,2]]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Generates all subsets (power set) of unique numbers.
        Uses node-collecting DFS backtracking.
        """
        results: List[List[int]] = []
        path: List[int] = []
        n = len(nums)

        def backtrack(start: int) -> None:
            # Every node in the recursion tree represents a valid subset
            results.append(list(path))

            for i in range(start, n):
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()  # Backtrack

        backtrack(0)
        return results

    def subsetsBitmask(self, nums: List[int]) -> List[List[int]]:
        """
        Alternative: Bit manipulation approach.
        Generates all 2^n subsets via integer masks.
        """
        n = len(nums)
        total_subsets = 1 << n
        results = []

        for mask in range(total_subsets):
            subset = []
            for j in range(n):
                if mask & (1 << j):
                    subset.append(nums[j])
            results.append(subset)

        return results
```

#### C++17
```cpp
#include <vector>

class Solution {
public:
    std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
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
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack(0, nums, path, results);
        return results;
    }

    private void backtrack(int start, int[] nums, List<Integer> path, List<List<Integer>> results) {
        results.add(new ArrayList<>(path));

        for (int i = start; i < nums.length; i++) {
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
  - The recursion generates exactly $2^N$ nodes.
  - At each node, copying the current `path` of average length $N / 2$ takes $\mathcal{O}(N)$ time.
  - For $N \le 10$, $10 \times 2^{10} = 10,240$ operations, which finishes in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and `path` vector (excluding the $2^N$ output subsets).

---

### Takeaway Pattern & Interview Traps

- **Node-Collecting vs Leaf-Collecting:** Notice there is **no base case condition** like `if len(path) == n: return` before adding to results. The subset is added unconditionally on entry to the function, capturing subsets of all lengths $0$ to $N$.
- **Index Advancement (`i + 1`):** Ensure the recursive call advances the search range to `i + 1` (not `start + 1`), guaranteeing that elements are only selected in strictly increasing order of their indices.