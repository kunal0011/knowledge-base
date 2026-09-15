---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 77: Combinations"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 77: Combinations

## LeetCode 77 — Combinations

### Problem Statement

Given two integers `n` and `k`, return **all possible combinations** of `k` numbers chosen from the range `[1, n]`.

You may return the answer in **any order**.

**Constraints**

* `1 ≤ n ≤ 20`
* `1 ≤ k ≤ n`

**Example**

* Input: `n = 4`, `k = 2`
* Output:

```
[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

---

### Key Observations

1. Order **does not matter**:

   * `[1,2]` and `[2,1]` are the same combination.
2. To avoid duplicates, we must enforce **increasing order** while building combinations.
3. Each element can be used **at most once**, so the next recursive call must start from the next number.
4. This is a classic **“choose / don’t choose”** backtracking problem with **combinatorial pruning**.
5. If the remaining numbers are insufficient to reach size `k`, the branch can be pruned early.

---

### Approach (Backtracking)

* Maintain:

  * `start`: the smallest number we are allowed to pick next
  * `path`: current combination being built
* Recursive steps:

  * If `len(path) == k`, record the combination.
  * Iterate from `start` to `n`:

    * Choose `i`
    * Recurse with `start = i + 1`
    * Backtrack (remove `i`)

---

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            # Base case: k elements chosen
            if len(path) == k:
                result.append(path.copy())
                return

            # Pruning:
            # Remaining numbers = n - start + 1
            # Needed numbers = k - len(path)
            for i in range(start, n + 1):
                if n - i + 1 < k - len(path):
                    break

                path.append(i)          # choose
                backtrack(i + 1)        # explore
                path.pop()              # un-choose

        backtrack(1)
        return result
```

---

### Example Walkthrough (`n = 4`, `k = 2`)

1. Start with `path = []`
2. Choose `1` → `[1]`

   * Choose `2` → `[1,2]` ✓
   * Choose `3` → `[1,3]` ✓
   * Choose `4` → `[1,4]` ✓
3. Backtrack, choose `2` → `[2]`

   * Choose `3` → `[2,3]` ✓
   * Choose `4` → `[2,4]` ✓
4. Backtrack, choose `3` → `[3]`

   * Choose `4` → `[3,4]` ✓
5. Choose `4` alone → cannot reach size `2` → pruned

**Final Output**

```
[[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
```

---

### Backtracking Tree Structure (`n = 4`, `k = 2`)

![https://i.ytimg.com/vi/ZtKpUF8gHDQ/hq720.jpg?rs=AOn4CLDAa22FwU-0hFA6rnjd6CZNXBVtMQ&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com](https://i.ytimg.com/vi/ZtKpUF8gHDQ/hq720.jpg?rs=AOn4CLDAa22FwU-0hFA6rnjd6CZNXBVtMQ&sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&utm_source=chatgpt.com)

![https://miro.medium.com/max/2058/1%2AMs439ltPzgkpqh_kFh9aGA.jpeg?utm_source=chatgpt.com](https://miro.medium.com/max/2058/1%2AMs439ltPzgkpqh_kFh9aGA.jpeg?utm_source=chatgpt.com)

![https://user-images.githubusercontent.com/114635550/257113025-26bc83f5-d6b9-423e-8b1c-523343151a0c.png?utm_source=chatgpt.com](https://user-images.githubusercontent.com/114635550/257113025-26bc83f5-d6b9-423e-8b1c-523343151a0c.png?utm_source=chatgpt.com)

**Conceptual Tree**

```
[]
        ------------------------------------------------
        |               |               |             |
       [1]             [2]             [3]           [4]
     /  |   \          /   \              |
 [1,2][1,3][1,4]   [2,3][2,4]          [3,4]
```

**Tree Characteristics**

* Depth = `k`
* Each level selects the **next larger number**
* No duplicates due to increasing order constraint
* Branches terminate early when remaining elements are insufficient

---

### Complexity Analysis

* **Time Complexity:**  
  `O(C(n, k) * k)`  
  (`k` for copying each valid combination)
* **Space Complexity:**  
  `O(k)` recursion stack (excluding result storage)

---

### Pattern Recognition

This problem exemplifies the **Combinations Backtracking Pattern**:

* Monotonic choice (`start` pointer)
* Fixed-length selection
* Pruning based on remaining capacity