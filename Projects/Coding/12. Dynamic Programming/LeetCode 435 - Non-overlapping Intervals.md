---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 435: Non-overlapping Intervals"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 435: Non-overlapping Intervals

**LeetCode 435 – Non-overlapping Intervals**, with **clear state definition, transition logic, DP table construction, and a worked example**.

---

## LeetCode 435 – Non-overlapping Intervals

### Problem Statement

You are given an array of intervals where  
`intervals[i] = [start_i, end_i]`.

You must **remove the minimum number of intervals** so that the remaining intervals **do not overlap**.

Two intervals `[a, b]` and `[c, d]` overlap if:

```
c < b
```

---

## Key Insight (DP Perspective)

Instead of directly minimizing removals, we can:

> **Maximize the number of non-overlapping intervals**, then subtract from total.

```
min_removals = n - max_non_overlapping
```

This converts the problem into a **Longest Increasing Subsequence–style DP** on intervals.

---

## Step 1: Sort Intervals

Sort intervals by **end time**, because finishing earlier leaves more room.

```
intervals.sort(key=lambda x: x[1])
```

---

## Step 2: DP State Definition

### State

```
dp[i] = maximum number of non-overlapping intervals
        ending exactly at interval i
```

---

### Base Case

```
dp[i] = 1
```

(Each interval alone is a valid non-overlapping set)

---

## Step 3: DP Transition

For every pair `(j < i)`:

If interval `j` does **not overlap** with interval `i`:

```
intervals[j][1] <= intervals[i][0]
```

Then:

```
dp[i] = max(dp[i], dp[j] + 1)
```

---

## Step 4: Final Answer

```
max_non_overlapping = max(dp)
answer = n - max_non_overlapping
```

---

## Example Walkthrough

### Input

```
intervals = [[1,2], [2,3], [3,4], [1,3]]
```

---

### Step 1: Sort by end time

```
Index   Interval
0       [1,2]
1       [2,3]
2       [1,3]
3       [3,4]
```

---

### Step 2: DP Table Initialization

```
dp = [1, 1, 1, 1]
```

---

### Step 3: Fill DP Table

#### i = 1 → [2,3]

* j = 0 → [1,2]

  * 2 <= 2 → valid

  ```
  dp[1] = max(1, 1 + 1) = 2
  ```

```
dp = [1, 2, 1, 1]
```

---

#### i = 2 → [1,3]

* j = 0 → 2 <= 1 ❌
* j = 1 → 3 <= 1 ❌

```
dp = [1, 2, 1, 1]
```

---

#### i = 3 → [3,4]

* j = 0 → 2 <= 3 → valid

  ```
  dp[3] = 2
  ```
* j = 1 → 3 <= 3 → valid

  ```
  dp[3] = max(2, 2 + 1) = 3
  ```
* j = 2 → 3 <= 3 → valid

  ```
  dp[3] = max(3, 1 + 1) = 3
  ```

```
dp = [1, 2, 1, 3]
```

---

### Step 4: Compute Answer

```
max_non_overlapping = 3
total_intervals = 4

min_removals = 4 - 3 = 1
```

---

## Final DP Table

| Index | Interval | dp[i] |
| --- | --- | --- |
| 0 | [1,2] | 1 |
| 1 | [2,3] | 2 |
| 2 | [1,3] | 1 |
| 3 | [3,4] | 3 |

---

## Python 3 (DP Solution with Typing)

```python
from typing import List

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        # Step 1: sort by end time
        intervals.sort(key=lambda x: x[1])
        n = len(intervals)

        # Step 2: DP table
        dp = [1] * n

        # Step 3: DP transition
        for i in range(n):
            for j in range(i):
                if intervals[j][1] <= intervals[i][0]:
                    dp[i] = max(dp[i], dp[j] + 1)

        # Step 4: result
        return n - max(dp)
```

---

## Time and Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n²)** |
| Space | **O(n)** |

---

## Important Note (Why Greedy Is Preferred)

This DP solution is **correct but not optimal**.

Greedy solution achieves:

```
O(n log n)
```

by always picking the interval with the **earliest end**.

However, DP is valuable for:

* Understanding interval scheduling formally
* Recognizing LIS-style interval problems
* Extending to weighted interval scheduling

---

If you want, I can next:

* Convert this DP into **state-compressed greedy reasoning**
* Compare DP vs Greedy side-by-side
* Show **weighted interval scheduling DP**, which greedy cannot solve

Just tell me.