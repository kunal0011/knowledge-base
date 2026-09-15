---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 338: Counting Bits"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 338: Counting Bits

**LeetCode 338 – Counting Bits**, written with explicit **state definition, transition, DP table construction, and example walkthrough**, aligned with interview-grade reasoning.

---

## LeetCode 338 – Counting Bits

### Problem Statement

Given an integer `n`, return an array `ans` of length `n + 1` where  
`ans[i]` is the number of `1`s in the binary representation of `i`, for all `0 ≤ i ≤ n`.

---

## Key Observation

Binary numbers exhibit a **repeating structure**:

* Any number `i` can be derived from a smaller number by:

  * Removing its **least significant bit**
  * Or shifting right by one bit

This allows us to reuse previously computed results → **Dynamic Programming**.

---

## DP Formulation

### 1. State Definition

Let:

```
dp[i] = number of set bits (1s) in binary representation of i
```

---

### 2. Base Case

```
dp[0] = 0
```

Binary of `0` → `0` → no set bits.

---

### 3. State Transition

There are two equivalent DP transitions.  
We will use the **most intuitive and interview-preferred one**.

#### Transition Formula

```
dp[i] = dp[i >> 1] + (i & 1)
```

#### Why this works

* `i >> 1` removes the **last bit** of `i`
* `(i & 1)` tells whether the last bit is `1` or `0`

So:

* Count bits of the smaller number
* Add `1` if the last bit is set

---

### 4. DP Table Creation Order

We build the DP table **bottom-up**:

```
i = 1 → n
```

Each `dp[i]` depends only on a **smaller index**, so it is already computed.

---

## Example Walkthrough

### Input

```
n = 5
```

### Binary Representations

| i | Binary | i >> 1 | dp[i >> 1] | i & 1 | dp[i] |
| --- | --- | --- | --- | --- | --- |
| 0 | 000 | — | — | — | 0 |
| 1 | 001 | 0 | 0 | 1 | 1 |
| 2 | 010 | 1 | 1 | 0 | 1 |
| 3 | 011 | 1 | 1 | 1 | 2 |
| 4 | 100 | 2 | 1 | 0 | 1 |
| 5 | 101 | 2 | 1 | 1 | 2 |

---

### DP Table Construction

```
dp = [0, 1, 1, 2, 1, 2]
```

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp: List[int] = [0] * (n + 1)

        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)

        return dp
```

---

## Time & Space Complexity

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(n)** |

---

## Intuition Summary (Interview Ready)

* Every number is built from a **smaller number**
* Right shift removes last bit
* Last bit contributes either `0` or `1`
* This makes it a **classic 1D DP problem**

---

## Alternative DP Insight (Optional)

Another valid transition:

```
dp[i] = dp[i & (i - 1)] + 1
```

This removes the **rightmost set bit**, but the earlier shift-based method is clearer and preferred for explanation.

---

If you want, I can also:

* Convert this into a **pattern-based DP explanation**
* Compare this with **Brian Kernighan’s approach**
* Show how this DP applies to **bit manipulation interview patterns**

Just let me know.