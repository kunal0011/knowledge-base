---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 312: Burst Balloons"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 312: Burst Balloons

**LeetCode 312 – Burst Balloons**, focused on **state definition, transition, DP table construction, and a worked example**. This problem is a **classic interval DP** and frequently asked in senior-level interviews.

---

## LeetCode 312 – Burst Balloons

### Problem Statement

You are given `n` balloons, indexed `0` to `n-1`. Each balloon has a number on it.  
When you burst balloon `i`, you gain:

```
nums[left] * nums[i] * nums[right]
```

where `left` and `right` are the nearest unburst balloons to `i`.

Return the **maximum coins** you can collect by bursting all balloons.

---

## Key Observation (Core DP Insight)

Instead of asking:

> “Which balloon should I burst first?”

We reverse the thinking and ask:

> **“Which balloon is burst LAST in a given interval?”**

Why this works:

* When a balloon is burst last, its neighbors are already fixed.
* This removes ambiguity and enables optimal substructure.

---

## Preprocessing (Very Important)

Add **virtual balloons** with value `1` at both ends.

```text
Original: nums = [3,1,5,8]
After padding: arr = [1,3,1,5,8,1]
Index:          0 1 2 3 4 5
```

Now:

* We **never burst** index `0` and `n+1`
* We only burst balloons in `(1 ... n)`

---

## DP State Definition

Let:

```
dp[i][j] = maximum coins obtainable by bursting
           all balloons strictly between i and j
```

Important:

* `i` and `j` themselves are NOT burst
* We compute for increasing interval length

---

## DP Transition

Assume balloon `k` is the **last balloon burst** in interval `(i, j)`:

```
dp[i][j] =
    max over k in (i+1 ... j-1):
        dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
```

### Why this works

* Left subproblem `(i, k)`
* Right subproblem `(k, j)`
* Final burst gives coins using fixed neighbors `i` and `j`

---

## Base Case

If there is **no balloon between i and j**:

```
dp[i][j] = 0   when j = i + 1
```

---

## DP Table Size

If original `nums` length = `n`  
After padding → length = `n + 2`

```
dp size = (n+2) × (n+2)
```

---

## Order of Computation (Interval DP)

We must compute **smaller intervals first**.

```
for length from 2 to n+1:
    for i from 0 to n+1-length:
        j = i + length
```

---

## Example Walkthrough

### Input

```text
nums = [3,1,5,8]
arr  = [1,3,1,5,8,1]
```

---

### Step 1: Smallest meaningful intervals (length = 2)

No balloon inside → all `dp[i][i+1] = 0`

---

### Step 2: Interval length = 3

Example: `(1, 3)` → balloon `2` only

```
dp[1][3] =
    arr[1] * arr[2] * arr[3]
  = 3 * 1 * 5 = 15
```

---

### Step 3: Larger intervals

Example: `(1, 4)` → balloons `{2,3}`

Try both choices for **last burst**:

**k = 2**

```
dp[1][2] + dp[2][4] + 3*1*8
= 0 + 40 + 24 = 64
```

**k = 3**

```
dp[1][3] + dp[3][4] + 3*5*8
= 15 + 0 + 120 = 135
```

```
dp[1][4] = 135
```

---

### Final Interval `(0, 5)`

All real balloons inside.

Answer:

```
dp[0][5] = 167
```

---

## DP Table (Conceptual)

```
      0   1   2   3   4   5
   -------------------------
0 |  0   0   3  30 159 167
1 |      0   0  15 135 159
2 |          0   0  40  48
3 |              0   0  40
4 |                  0   0
5 |                      0
```

Final Answer → `dp[0][n+1]`

---

## Python 3 DP Solution (With Typing)

```python
from typing import List

class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # Pad with 1 at both ends
        arr = [1] + nums + [1]
        n = len(arr)

        # dp[i][j] = max coins from bursting balloons between i and j
        dp = [[0] * n for _ in range(n)]

        # length is the distance between i and j
        for length in range(2, n):
            for i in range(0, n - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j],
                        dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j]
                    )

        return dp[0][n - 1]
```

---

## Time and Space Complexity

```
Time:  O(n³)
Space: O(n²)
```

---

## Pattern Recognition

This problem belongs to:

* **Interval DP**
* Similar to:

  * Matrix Chain Multiplication
  * Palindrome Partitioning
  * Optimal BST

**Trigger phrase:**

> “Choose the last operation in a range”

---

If you want next:

* Visual **interval DP tree**
* Conversion to **top-down memoization**
* Comparison with **Matrix Chain Multiplication**
* How to identify interval DP in interviews

Tell me which one.