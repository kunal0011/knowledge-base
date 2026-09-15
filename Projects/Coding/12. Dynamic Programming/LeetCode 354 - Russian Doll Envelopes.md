---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 354: Russian Doll Envelopes"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 354: Russian Doll Envelopes

**LeetCode 354 – Russian Doll Envelopes**, focusing strictly on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 354 – Russian Doll Envelopes

### Problem Statement

You are given envelopes where  
`envelopes[i] = [width_i, height_i]`.

One envelope can fit into another **iff**:

```
width1 < width2 AND height1 < height2
```

Return the **maximum number of envelopes** you can Russian-doll (nest).

---

## Key Observation (DP Perspective)

This is a **2D version of Longest Increasing Subsequence (LIS)**:

* We want the longest chain where **both width and height strictly increase**.
* Direct LIS on 2D is hard → we **sort first**, then apply DP.

---

## Step 1: Sorting Strategy (Critical)

Sort envelopes by:

1. **width ascending**
2. **height ascending** (for DP version)

```
(width ↑, height ↑)
```

Why?

* Ensures width condition is automatically respected
* We only need to check height during DP transition

> Note: Descending height is required for the optimized LIS solution, **not** for the O(n²) DP.

---

## Step 2: DP State Definition

### DP State

```
dp[i] = maximum number of envelopes ending at envelope i
```

Interpretation:

* Envelope `i` is the **outermost** envelope in this chain

---

## Step 3: DP Transition

For every `i`, try all previous envelopes `j < i`:

```
If height[j] < height[i]:
    dp[i] = max(dp[i], dp[j] + 1)
```

Why only height?

* Width is already guaranteed by sorting

---

## Step 4: Base Case

Every envelope can form a chain of length **1** by itself:

```
dp[i] = 1
```

---

## Step 5: Final Answer

```
answer = max(dp)
```

---

## Python 3 DP Solution (O(n²))

```python
from typing import List

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0

        # Step 1: sort by width asc, height asc
        envelopes.sort(key=lambda x: (x[0], x[1]))

        n = len(envelopes)
        dp = [1] * n  # Step 4: base case

        # Step 3: DP transition
        for i in range(n):
            for j in range(i):
                if envelopes[j][1] < envelopes[i][1]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
```

---

## Step 6: DP Table Construction (Worked Example)

### Input

```
envelopes = [[5,4],[6,4],[6,7],[2,3]]
```

### After Sorting

```
Index   Envelope
0       [2,3]
1       [5,4]
2       [6,4]
3       [6,7]
```

---

### DP Table Evolution

| i | Envelope | Valid Previous j | dp[i] |
| --- | --- | --- | --- |
| 0 | [2,3] | — | 1 |
| 1 | [5,4] | [2,3] | 2 |
| 2 | [6,4] | [2,3] | 2 |
| 3 | [6,7] | [2,3], [5,4], [6,4] | 3 |

---

### Detailed Transition Walkthrough

#### i = 0 → [2,3]

```
dp[0] = 1
```

---

#### i = 1 → [5,4]

```
[2,3] → [5,4] valid
dp[1] = dp[0] + 1 = 2
```

---

#### i = 2 → [6,4]

```
[2,3] → [6,4] valid
[5,4] → [6,4] invalid (height not <)

dp[2] = 2
```

---

#### i = 3 → [6,7]

```
[2,3] → [6,7] ✔
[5,4] → [6,7] ✔
[6,4] → [6,7] ✔

dp[3] = max(2+1, 2+1, 1+1) = 3
```

---

## Final DP Table

```
dp = [1, 2, 2, 3]
```

### Output

```
3
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time | O(n²) |
| Space | O(n) |

---

## When to Use This DP Approach

* When you want **clear state + transition reasoning**
* Useful in interviews to **explain correctness**
* Easier to debug and extend than LIS optimization

If you want next:

* **Why optimized LIS needs height descending**
* **Binary Search LIS version with proof**
* **Backtracking / reconstruction of actual envelope chain**

Just tell me.