---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 553: Optimal Division"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 553: Optimal Division

**LeetCode 553 – Optimal Division**, including **state definition, transitions, DP table construction, and a worked example**.

---

## LeetCode 553 – Optimal Division

### Problem Statement

You are given an array `nums` of positive integers.  
Insert division operators `/` and parentheses to **maximize the result** of the expression.

* You must keep the original order of numbers.
* Division is real-number division.

Return the expression string that yields the **maximum value**.

---

## Key Observation (Why DP is Applicable)

This is an **interval DP** problem:

* You are partitioning an array into left and right subexpressions.
* Each partition affects the result due to **division’s non-associativity**.
* For each subarray, you must know:

  * The **maximum value** achievable
  * The **minimum value** achievable  
    (critical because division by a smaller number increases the result)

Hence, **both min and max values must be tracked**.

---

## DP State Definition

Let:

```
dp[i][j] = (max_value, min_value)
```

Where:

* `i` = start index
* `j` = end index
* `dp[i][j].max_value` = maximum value obtainable from nums[i..j]
* `dp[i][j].min_value` = minimum value obtainable from nums[i..j]

---

## Base Case

For a single number:

```
dp[i][i].max = nums[i]
dp[i][i].min = nums[i]
```

Because no division is possible.

---

## State Transition

For interval `[i, j]`, split at position `k`:

```
(nums[i..k]) / (nums[k+1..j])
```

To **maximize** the result:

```
max = left.max / right.min
```

To **minimize** the result:

```
min = left.min / right.max
```

### Transition Formula

For all `k ∈ [i, j-1]`:

```
dp[i][j].max = max(
    dp[i][k].max / dp[k+1][j].min
)

dp[i][j].min = min(
    dp[i][k].min / dp[k+1][j].max
)
```

---

## DP Table Construction Order

We fill the DP table by **increasing subarray length**:

```
length = 1 → n
```

This ensures smaller subproblems are already computed.

---

## Example Walkthrough

### Input

```text
nums = [1000, 100, 10, 2]
```

### Step 1: Base Cases

| i | j | max | min |
| --- | --- | --- | --- |
| 0 | 0 | 1000 | 1000 |
| 1 | 1 | 100 | 100 |
| 2 | 2 | 10 | 10 |
| 3 | 3 | 2 | 2 |

---

### Step 2: Length = 2

| Interval | Expression | max | min |
| --- | --- | --- | --- |
| [0,1] | 1000/100 | 10 | 10 |
| [1,2] | 100/10 | 10 | 10 |
| [2,3] | 10/2 | 5 | 5 |

---

### Step 3: Length = 3

#### Interval [0,2]

Splits:

* k=0 → `1000 / (100/10) = 100`
* k=1 → `(1000/100) / 10 = 1`

```
max = 100
min = 1
```

#### Interval [1,3]

Splits:

* k=1 → `100 / (10/2) = 20`
* k=2 → `(100/10) / 2 = 5`

```
max = 20
min = 5
```

---

### Step 4: Length = 4 (Final)

#### Interval [0,3]

Splits:

* k=0 → `1000 / (100/10/2) = 200`
* k=1 → `(1000/100) / (10/2) = 2`
* k=2 → `(1000/100/10) / 2 = 0.5`

```
max = 200
min = 0.5
```

---

## Final Result

Maximum value is achieved by:

```
1000 / (100 / 10 / 2)
```

---

## Python 3 DP Implementation (with typing)

```python
from typing import List, Tuple

class Solution:
    def optimalDivision(self, nums: List[int]) -> str:
        n = len(nums)

        # dp[i][j] = (max_value, min_value)
        dp: List[List[Tuple[float, float]]] = [
            [(0.0, 0.0) for _ in range(n)] for _ in range(n)
        ]

        # expr[i][j] stores expression for max value
        expr = [["" for _ in range(n)] for _ in range(n)]

        for i in range(n):
            dp[i][i] = (nums[i], nums[i])
            expr[i][i] = str(nums[i])

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                max_val = float("-inf")
                min_val = float("inf")
                max_expr = ""

                for k in range(i, j):
                    left_max, left_min = dp[i][k]
                    right_max, right_min = dp[k + 1][j]

                    # maximize
                    val = left_max / right_min
                    if val > max_val:
                        max_val = val
                        right_part = expr[k + 1][j]
                        if k + 1 < j:
                            right_part = f"({right_part})"
                        max_expr = f"{expr[i][k]}/{right_part}"

                    # minimize
                    min_val = min(min_val, left_min / right_max)

                dp[i][j] = (max_val, min_val)
                expr[i][j] = max_expr

        return expr[0][n - 1]
```

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time | **O(n³)** |
| Space | **O(n²)** |

---

## Why This DP Works (Conceptual Summary)

* Division rewards **small denominators**
* Hence:

  * Use **min** on the right for **maximization**
  * Use **max** on the right for **minimization**
* This mirrors classic **Min-Max Interval DP** (same idea as Matrix Chain Multiplication)

---

If you want, I can also:

* Convert this into a **pure mathematical proof**
* Show why **greedy collapses to one parenthesis**
* Draw a **DP dependency graph**
* Compare **DP vs Greedy shortcut**

Tell me how deep you want to go.