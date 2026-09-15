---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 646: Maximum Length of Pair Chain"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 646: Maximum Length of Pair Chain

**LeetCode 646 – Maximum Length of Pair Chain**, with **proper state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 646 — Maximum Length of Pair Chain

### Problem Statement

You are given `n` pairs of integers `pairs`, where `pairs[i] = [a, b]` and `a < b`.

A pair `[c, d]` can follow `[a, b]` **if and only if**:

```
b < c
```

Return the **maximum length** of a chain you can form.

---

## Key Observation (DP Perspective)

This problem is a **variation of Longest Increasing Subsequence (LIS)**.

* Each pair is an interval.
* We want to select the longest sequence such that:

  ```
  previous_pair.end < current_pair.start
  ```

Hence:

* Sort pairs by starting value.
* Use DP to compute the longest valid chain ending at each index.

---

## Step 1: Sorting

Sort the pairs by their **first element**:

```
pairs.sort(key=lambda x: x[0])
```

Why?

* Ensures we only look **leftwards** for valid predecessors.

---

## Step 2: DP State Definition

### DP State

```
dp[i] = length of the longest chain ending at index i
```

This means:

* The **last pair used** in the chain is `pairs[i]`.

---

## Step 3: DP Initialization

Each pair alone can form a chain of length 1:

```
dp[i] = 1   for all i
```

---

## Step 4: State Transition

For every `i`, check all previous pairs `j < i`:

### Transition Condition

```
if pairs[j][1] < pairs[i][0]:
    dp[i] = max(dp[i], dp[j] + 1)
```

Meaning:

* If pair `j` can come before `i`,
* Extend the chain ending at `j` by including `i`.

---

## Step 5: Final Answer

```
max(dp)
```

---

## Example Walkthrough

### Input

```
pairs = [[1,2],[2,3],[3,4]]
```

### Step 1: Sort

```
[[1,2],[2,3],[3,4]]
```

---

### Step 2: DP Table Construction

| i | Pair | Valid previous j | dp[i] |
| --- | --- | --- | --- |
| 0 | [1,2] | None | 1 |
| 1 | [2,3] | None (2 ≯ 2) | 1 |
| 2 | [3,4] | j=0 (2 < 3) | 2 |

---

### DP Array Evolution

```
Initial dp = [1, 1, 1]

i = 1:
  No valid j → dp = [1, 1, 1]

i = 2:
  j = 0 valid → dp[2] = dp[0] + 1 = 2
  dp = [1, 1, 2]
```

---

### Final Answer

```
max(dp) = 2
```

---

## Python 3 (DP Solution with Typing)

```python
from typing import List

class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        # Step 1: Sort by first element
        pairs.sort(key=lambda x: x[0])
        
        n = len(pairs)
        
        # Step 2: DP initialization
        dp = [1] * n
        
        # Step 3: State transition
        for i in range(n):
            for j in range(i):
                if pairs[j][1] < pairs[i][0]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # Step 4: Result
        return max(dp)
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n²)** |
| Space Complexity | **O(n)** |

---

## When to Use This DP Pattern

* Problems involving **interval chaining**
* Variants of **LIS**
* Constraints based on **previous element compatibility**

---

If you want, I can also provide:

* Greedy O(n log n) solution (interval scheduling)
* DP + binary search optimization
* Backtracking tree analogy for intuition

Just tell me.