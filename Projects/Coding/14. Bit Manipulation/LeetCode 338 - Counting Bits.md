---
date: "2026-08-29"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 338: Counting Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
---

# LeetCode 338: Counting Bits

---

### Problem Statement

Given an integer `n`, return an array `ans` of length `n + 1` such that `ans[i]` is the number of `1`'s in the binary representation of `i`.

---

### Key Observation

* The number of set bits in `i` is equal to `ans[i >> 1] + (i & 1)`.
* `i >> 1` shifts off the lowest bit, whose count was already calculated. `(i & 1)` is 1 if odd, 0 if even.

---

### Core Technique: Bitwise DP Transition

---

### Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i >> 1] + (i & 1)
        return dp
```

---

### Worked-Out Example

```
n = 5
dp[0] = 0
dp[1] = dp[0] + 1 = 1
dp[2] = dp[1] + 0 = 1
dp[3] = dp[1] + 1 = 2
dp[4] = dp[2] + 0 = 1
dp[5] = dp[2] + 1 = 2
Result = [0, 1, 1, 2, 1, 2]
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Subproblem recurrence `dp[i] = dp[i >> 1] + (i & 1)` computes bitcounts in a single linear pass.