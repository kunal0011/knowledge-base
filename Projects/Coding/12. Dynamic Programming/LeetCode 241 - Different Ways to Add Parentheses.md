---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 241: Different Ways to Add Parentheses"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 241: Different Ways to Add Parentheses

**LeetCode 241 – Different Ways to Add Parentheses**, with **formal state definition, transition, DP table construction, and a worked example**.

---

## Problem Statement (LC 241)

Given a string expression consisting of digits and binary operators (`+`, `-`, `*`), return **all possible results** from computing the expression by adding parentheses in all possible ways.

**Example**

```text
Input:  "2*3-4*5"
Output: [-34, -14, -10, -10, 10]
```

Order does not matter.

---

## Key Observation

* The expression is **fixed order**, but **parentheses change evaluation order**
* Each operator can act as the **last operation**
* The problem exhibits:

  * **Optimal substructure**
  * **Overlapping subproblems**

Hence, **Dynamic Programming over intervals** (or memoized recursion) is ideal.

---

## Step 1: Tokenization (Preprocessing)

Convert the string into:

* `nums[]`: list of integers
* `ops[]`: list of operators

Example:

```text
"2*3-4*5"

nums = [2, 3, 4, 5]
ops  = ['*', '-', '*']
```

Let:

* `n = len(nums)`

---

## Step 2: DP State Definition

### DP State

```
dp[i][j] = all possible results from evaluating
           the subexpression using nums[i] to nums[j]
```

* `i`, `j` are indices in `nums`
* `0 ≤ i ≤ j < n`
* Each `dp[i][j]` is a **list of integers**, not a single value

---

### Base Case

```
dp[i][i] = [nums[i]]
```

A single number evaluates to itself.

---

## Step 3: State Transition

To compute `dp[i][j]` where `i < j`:

* Try **every operator k** between `i` and `j`
* Operator `ops[k]` splits the expression into:

  * Left: `dp[i][k]`
  * Right: `dp[k+1][j]`

### Transition Formula

```
dp[i][j] = for each k in [i, j-1]:
              for each a in dp[i][k]:
                  for each b in dp[k+1][j]:
                      apply ops[k] on (a, b)
```

### Operator Application

```
if ops[k] == '+': a + b
if ops[k] == '-': a - b
if ops[k] == '*': a * b
```

---

## Step 4: DP Table Construction Order

We fill the DP table by **increasing interval length**.

```
length = 1 → base cases
length = 2 → dp[i][i+1]
length = 3 → dp[i][i+2]
...
length = n
```

---

## Step 5: Worked Example

### Expression

```text
2 * 3 - 4 * 5
nums = [2, 3, 4, 5]
ops  = ['*', '-', '*']
```

---

### Base Cases (length = 1)

```
dp[0][0] = [2]
dp[1][1] = [3]
dp[2][2] = [4]
dp[3][3] = [5]
```

---

### Length = 2

```
dp[0][1]: 2 * 3 = [6]
dp[1][2]: 3 - 4 = [-1]
dp[2][3]: 4 * 5 = [20]
```

---

### Length = 3

#### dp[0][2] → "2\*3-4"

Split options:

1. k = 0 → (2) \* (3-4)

   ```
   2 * (-1) = -2
   ```
2. k = 1 → (2\*3) - (4)

   ```
   6 - 4 = 2
   ```

```
dp[0][2] = [-2, 2]
```

---

#### dp[1][3] → "3-4\*5"

Split options:

1. k = 1 → (3) - (4\*5)

   ```
   3 - 20 = -17
   ```
2. k = 2 → (3-4) \* (5)

   ```
   -1 * 5 = -5
   ```

```
dp[1][3] = [-17, -5]
```

---

### Length = 4 (Full Expression)

#### dp[0][3] → "2*3-4*5"

Split options:

1. k = 0 → (2) \* (3-4\*5)

   ```
   2 * (-17) = -34
   2 * (-5)  = -10
   ```
2. k = 1 → (2*3) - (4*5)

   ```
   6 - 20 = -14
   ```
3. k = 2 → (2\*3-4) \* (5)

   ```
   -2 * 5 = -10
    2 * 5 = 10
   ```

```
dp[0][3] = [-34, -10, -14, -10, 10]
```

---

## Step 6: Final Answer

```
return dp[0][n-1]
```

---

## Python 3 DP Implementation (Typing Included)

```python
from typing import List
import operator

class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        nums = []
        ops = []

        # Tokenize
        num = 0
        for ch in expression:
            if ch.isdigit():
                num = num * 10 + int(ch)
            else:
                nums.append(num)
                ops.append(ch)
                num = 0
        nums.append(num)

        n = len(nums)
        dp = [[[] for _ in range(n)] for _ in range(n)]

        # Base case
        for i in range(n):
            dp[i][i] = [nums[i]]

        # Operator mapping
        op_map = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul
        }

        # Interval DP
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                for k in range(i, j):
                    for a in dp[i][k]:
                        for b in dp[k + 1][j]:
                            dp[i][j].append(op_map[ops[k]](a, b))

        return dp[0][n - 1]
```

---

## Time & Space Complexity

### Time

* Worst case exponential due to result combinations
* Roughly: **Catalan-like growth**

### Space

* DP table storing lists → **O(n³)** in practice

---

## Summary

| Component | Description |
| --- | --- |
| DP Type | Interval DP |
| State | `dp[i][j] = all results from nums[i..j]` |
| Transition | Split at every operator |
| Base Case | Single number |
| Output | `dp[0][n-1]` |

---

If you want, I can also provide:

* Memoized recursion vs DP comparison
* Expression tree visualization
* Why greedy or stack-based parsing does **not** work here
* Pattern mapping to other interval DP problems (Burst Balloons, Matrix Chain Multiplication)

Just tell me.