---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 46: Permutations"
tags:
  - leetcode
  - coding
  - backtracking
  - array
  - permutation
  - amazon
  - google
---

# LeetCode 46: Permutations

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg, Uber  
**Difficulty:** Medium  
**Topic:** Backtracking / Permutations / In-Place Swapping  

---

### Problem Statement

Given an array `nums` of distinct integers, return *all possible permutations*. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[List[int]]` containing all $N!$ permutations.
- **Constraints:**
  - $1 \le \text{nums.length} \le 6$
  - $-10 \le \text{nums}[i] \le 10$
  - All integers of `nums` are **unique**.

---

### Key Idea & Intuition

- **Permutations vs Combinations:**
  - Combinations choose a subset where order does not matter.
  - Permutations rearrange all $N$ elements where every ordering is distinct.
- **State-Space Tree:**
  - At depth $0$, we can choose any of the $N$ numbers for the 1st position.
  - At depth $1$, we choose any of the remaining $N - 1$ numbers for the 2nd position.
  - At depth $k$, there are $N - k$ candidate choices.
  - Total leaves $= N \times (N - 1) \times \dots \times 1 = N!$.
- **Two Backtracking Paradigms:**
  1. **Auxiliary `used` Boolean Array:**
     - Maintain `path = []` and `used = [False] * N`.
     - At each step, iterate through all $i \in [0, N-1]$. If `not used[i]`: mark `used[i] = True`, push `nums[i]`, recurse, unmark, and pop.
  2. **In-Place Swap (Optimal $\mathcal{O}(1)$ Auxiliary Space):**
     - Maintain current position `first` from $0$ to $N - 1$.
     - Any element in `nums[first ... N-1]` can be swapped into position `first`.
     - Swap `nums[first]` with `nums[i]`, recurse on `first + 1`, and swap back!
     - When `first == N`, snapshot `list(nums)`. No extra `used` array or `path` buffer required.

---

### Solution Approach (Step-by-Step)

#### Using In-Place Swapping:
1. Initialize `results = []`.
2. Define `backtrack(first)`:
   - If `first == len(nums)`:
     - Append a snapshot `list(nums)` to `results`.
     - Return.
   - For `i` from `first` to `len(nums) - 1`:
     - **Choose:** Swap `nums[first]` with `nums[i]`.
     - **Explore:** Recurse `backtrack(first + 1)`.
     - **Backtrack:** Swap `nums[first]` with `nums[i]` back.
3. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

Let `nums = [1, 2, 3]`:

```
                           backtrack(first = 0)
                 /                  |                  \
           Swap(0, 0)          Swap(0, 1)          Swap(0, 2)
           nums=[1,2,3]        nums=[2,1,3]        nums=[3,2,1]
             first=1             first=1             first=1
            /        \          /        \          /        \
       Swap(1,1)   Swap(1,2)  Swap(1,1) Swap(1,2)  Swap(1,1) Swap(1,2)
        [1,2,3]     [1,3,2]    [2,1,3]   [2,3,1]    [3,2,1]   [3,1,2]
        first=2     first=2    first=2   first=2    first=2   first=2
           |           |          |         |          |         |
        Swap(2,2)   Swap(2,2)  Swap(2,2) Swap(2,2)  Swap(2,2) Swap(2,2)
           |           |          |         |          |         |
        [1,2,3]     [1,3,2]    [2,1,3]   [2,3,1]    [3,2,1]   [3,1,2]
```

Every leaf at depth $N=3$ represents one of the $3! = 6$ unique permutations.

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | $N!$ Count | Permutations |
| :--- | :--- | :--- | :--- |
| **Standard** | `[1, 2, 3]` | $3! = 6$ | `[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,2,1],[3,1,2]]` |
| **Two Elements** | `[0, 1]` | $2! = 2$ | `[[0, 1], [1, 0]]` |
| **Single Element** | `[1]` | $1! = 1$ | `[[1]]` |
| **Max Size** | `[1, 2, 3, 4, 5, 6]` | $6! = 720$ | 720 distinct permutations |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        Generates all permutations of distinct integers using in-place swapping.
        O(1) auxiliary space beyond the output list and recursion stack.
        """
        results: List[List[int]] = []
        n = len(nums)

        def backtrack(first: int) -> None:
            if first == n:
                results.append(list(nums))
                return

            for i in range(first, n):
                # Place nums[i] at position `first`
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                # Backtrack: restore original array order
                nums[first], nums[i] = nums[i], nums[first]

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <vector>
#include <utility>

class Solution {
public:
    std::vector<std::vector<int>> permute(std::vector<int>& nums) {
        std::vector<std::vector<int>> results;
        backtrack(0, nums, results);
        return results;
    }

private:
    void backtrack(int first, std::vector<int>& nums, std::vector<std::vector<int>>& results) {
        if (first == static_cast<int>(nums.size())) {
            results.push_back(nums);
            return;
        }

        for (int i = first; i < static_cast<int>(nums.size()); ++i) {
            std::swap(nums[first], nums[i]);
            backtrack(first + 1, nums, results);
            std::swap(nums[first], nums[i]); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        backtrack(0, nums, results);
        return results;
    }

    private void backtrack(int first, int[] nums, List<List<Integer>> results) {
        if (first == nums.length) {
            List<Integer> current = new ArrayList<>(nums.length);
            for (int val : nums) {
                current.add(val);
            }
            results.add(current);
            return;
        }

        for (int i = first; i < nums.length; i++) {
            swap(nums, first, i);
            backtrack(first + 1, nums, results);
            swap(nums, first, i); // Backtrack
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot N!)$.
  - There are $N!$ permutations generated at the leaves.
  - At each leaf, copying the array of length $N$ takes $\mathcal{O}(N)$ time.
  - Total number of recursive calls is $\sum_{k=1}^N \frac{N!}{(N-k)!} = \mathcal{O}(N!)$.
  - For $N \le 6$, $6 \times 720 = 4,320$ operations, which finishes in $< 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack (tree depth $N$). In-place swapping eliminates all auxiliary tracking arrays.

---

### Takeaway Pattern & Interview Traps

- **In-Place Swapping:** Swapping `nums[first]` with `nums[i]` is the canonical memory-optimal technique for generating permutations when elements are all distinct.
- **Backtrack Swap Order:** Failing to swap back (`nums[first], nums[i] = nums[i], nums[first]`) alters the array state for subsequent iterations of the loop, corrupting the generation order and producing duplicate permutations.