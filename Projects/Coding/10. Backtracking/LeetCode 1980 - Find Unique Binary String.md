---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1980: Find Unique Binary String"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 1980: Find Unique Binary String

## LeetCode 1980 — Find Unique Binary String

---

### Problem Statement

Given an array `nums` containing **n unique binary strings**, each of length `n`, return **any binary string of length `n`** that **does not appear** in `nums`.

You may return the answer in **any order**.

**Constraints**

* `n == nums.length`
* `1 ≤ n ≤ 16`
* Each string in `nums` has length `n`
* All strings in `nums` are **unique**
* Each string consists only of `'0'` and `'1'`

---

### Key Observations

1. There are exactly `2^n` possible binary strings of length `n`.
2. The input provides only `n` strings.
3. Since `2^n > n` for all `n ≥ 1`, **at least one binary string must be missing**.
4. This is fundamentally a **state-space search** problem:

   * At each position, choose `'0'` or `'1'`
   * Depth = `n`
5. A backtracking tree naturally represents **all possible binary strings**.
6. We stop as soon as we find one string **not present** in `nums`.

---

## Approach 1 — Backtracking (Explicit State-Space Search)

### Idea

* Convert `nums` into a set for `O(1)` lookup.
* Generate all binary strings of length `n` using backtracking.
* Return the first string not found in the set.

This matches your request for a **complete conceptual backtracking tree**.

---

### Python 3 Solution (with Typing)

```python
from typing import List, Set

class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        n: int = len(nums)
        seen: Set[str] = set(nums)

        def backtrack(path: str) -> str | None:
            if len(path) == n:
                if path not in seen:
                    return path
                return None

            for ch in ('0', '1'):
                res = backtrack(path + ch)
                if res is not None:
                    return res

            return None

        return backtrack("")  # guaranteed to find a solution
```

---

## Example Explanation

### Input

```text
nums = ["01", "10"]
n = 2
```

### All possible binary strings of length 2

```
00, 01, 10, 11
```

Strings already present:

```
01, 10
```

Valid missing strings:

```
00 or 11
```

Backtracking will return **one of them**.

---

## Backtracking Tree Structure (Complete Conceptual Tree)

![https://media.geeksforgeeks.org/wp-content/uploads/20190326214456/tree14-300x214.png?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/uploads/20190326214456/tree14-300x214.png?utm_source=chatgpt.com)

![https://i.sstatic.net/nGUfI.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/1ndZtHicu5mZUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw6KqArKLCmMKix2M8gvdSwrqvDIDHNxM800TTJJNEzMzg0MMC4KNYk3cSpKL0s2LDFQKwYAdOcmcw?utm_source=chatgpt.com)

![https://miro.medium.com/1%2AuxrFAipnpiJOYAl3C8ZX3Q.png?utm_source=chatgpt.com](https://miro.medium.com/1%2AuxrFAipnpiJOYAl3C8ZX3Q.png?utm_source=chatgpt.com)

### Conceptual Tree for `n = 2`

```
""
                  -----------------
                  |               |
                 "0"             "1"
              ---------        ---------
              |       |        |       |
            "00"    "01"     "10"     "11"
```

### Navigation Logic

* Start at root `""`
* At each level, branch into:

  * `'0'`
  * `'1'`
* Leaf nodes represent **complete binary strings**
* Check leaf against the input set
* Stop traversal when a missing string is found