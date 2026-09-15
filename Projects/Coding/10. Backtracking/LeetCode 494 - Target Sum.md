---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 494: Target Sum"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 494: Target Sum

## LeetCode 494 — Target Sum

---

## Problem Statement

You are given an integer array `nums` and an integer `target`.

You want to build an expression by assigning **either `'+'` or `'-'`** in front of each number in `nums`, and then evaluate the expression.

Return the **number of different expressions** that evaluate to `target`.

---

### Example

```text
Input: nums = [1,1,1,1,1], target = 3
Output: 5
```

---

## Key Observations

1. **Each number has exactly two choices**:

   * Add it (`+nums[i]`)
   * Subtract it (`-nums[i]`)
2. The problem asks for a **count**, not the actual expressions.
3. The decision process forms a **binary tree**:

   * Each level corresponds to one index in `nums`
   * Left branch = `+`
   * Right branch = `-`
4. This is fundamentally a **backtracking / DFS counting problem**.
5. No pruning is possible in the pure backtracking version because:

   * Negative numbers are allowed
   * Intermediate sums can exceed the target and still recover later

---

## Backtracking Approach (DFS)

### State Definition

At index `i`, we track:

* `current_sum`: the sum built so far

### Recurrence

At each index:

* Try `+nums[i]`
* Try `-nums[i]`

### Base Case

* If `i == len(nums)`:

  * If `current_sum == target` → count 1
  * Else → count 0

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        count: int = 0

        def backtrack(index: int, current_sum: int) -> None:
            nonlocal count

            if index == len(nums):
                if current_sum == target:
                    count += 1
                return

            # Choose +
            backtrack(index + 1, current_sum + nums[index])

            # Choose -
            backtrack(index + 1, current_sum - nums[index])

        backtrack(0, 0)
        return count
```

---

## Example Explanation

### Input

```text
nums = [1,1,1,1,1]
target = 3
```

We need expressions that evaluate to `3`.

Valid expressions include:

```
+1 +1 +1 +1 -1 = 3
+1 +1 +1 -1 +1 = 3
+1 +1 -1 +1 +1 = 3
+1 -1 +1 +1 +1 = 3
-1 +1 +1 +1 +1 = 3
```

Total ways = **5**

---

## Backtracking Tree Structure (Complete Conceptual Tree)

Each level corresponds to choosing `+` or `-` for one number.

For `nums = [1,1,1]`, target is irrelevant for structure.

```
(0)
                          /-------------------------------\
                       +1                                  -1
                      (1)                                 (-1)
               /--------------\                     /--------------\
            +1                 -1                +1                 -1
           (2)                (0)                (0)                (-2)
        /-------\           /-------\          /-------\           /-------\
     +1          -1      +1          -1     +1          -1      +1          -1
    (3)         (1)     (1)         (-1)    (1)         (-1)   (-1)         (-3)
```

### How to Read This Tree

* Root starts at sum `0`
* Each edge adds or subtracts the current number
* Leaf nodes represent final sums
* We **count how many leaves equal `target`**

---

## Key Insight About Tree Growth

For `n` numbers:

* Total leaf nodes = `2ⁿ`
* Tree is **full binary**
* No duplicate paths
* No pruning in basic backtracking

---

## Complexity Analysis

### Time Complexity

```
O(2ⁿ)
```

Every number creates two branches.

### Space Complexity

```
O(n)
```

Depth of recursion stack.

---

## Important Pattern Recognition

This problem represents the **“Binary Choice Backtracking”** pattern:

* Each step has exactly two decisions
* No ordering constraints
* Count solutions, not generate them

This pattern also appears in:

* Subset generation
* Binary expression enumeration
* Knapsack-style decision trees

---

## Interview One-Liner

> “At each index we either add or subtract the number, forming a binary decision tree of size `2ⁿ`. We count how many root-to-leaf paths evaluate to the target.”

---

If you want next, I can:

* Convert this to the **subset-sum / DP transformation**
* Show **memoized DFS** to optimize overlapping subproblems
* Draw the **DP table interpretation** corresponding to this tree