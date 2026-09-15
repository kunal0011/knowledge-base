---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 131: Palindrome Partitioning"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 131: Palindrome Partitioning

**LeetCode 131 – Palindrome Partitioning**, focused on **state definition, transition, DP table creation, and a worked example**.

---

## Problem Statement (LeetCode 131)

Given a string `s`, partition `s` such that **every substring of the partition is a palindrome**.  
Return **all possible palindrome partitioning** of `s`.

---

## Key Insight

The problem has **two layers**:

1. **Palindrome checking** → overlapping subproblems → **Dynamic Programming**
2. **Generating all valid partitions** → combinatorial → **Backtracking / DFS**

The **correct DP solution** is:

> **Precompute palindromic substrings using DP**, then use **DFS to build partitions efficiently**.

---

# Part 1: DP for Palindrome Checking

## DP State Definition

Let:

```
dp[i][j] = True  if s[i...j] is a palindrome
           False otherwise
```

Where:

* `0 ≤ i ≤ j < n`
* `n = len(s)`

---

## State Transition

A substring `s[i...j]` is a palindrome **if and only if**:

```
s[i] == s[j] AND
(j - i <= 2 OR dp[i+1][j-1] == True)
```

### Explanation:

* Single character (`i == j`) → palindrome
* Two characters (`j - i == 1`) → palindrome if both equal
* More than two → ends must match AND inner substring must be palindrome

---

## Base Cases

```
dp[i][i] = True    # every single character is a palindrome
```

---

## DP Table Construction Order

We must compute **shorter substrings first**, so that `dp[i+1][j-1]` is already known.

Correct order:

```
length = 1 → 2 → 3 → ... → n
```

---

## DP Table Creation Code

```python
from typing import List

def build_palindrome_dp(s: str) -> List[List[bool]]:
    n = len(s)
    dp = [[False] * n for _ in range(n)]

    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length <= 2:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]

    return dp
```

---

## Example: `s = "aab"`

Index mapping:

```
0:a  1:a  2:b
```

### DP Table (`dp[i][j]`)

| i\j | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | T | T | F |
| 1 |  | T | F |
| 2 |  |  | T |

### How values were filled:

* `"a"` → True
* `"aa"` → True
* `"ab"` → False
* `"aab"` → False

---

# Part 2: Generating Partitions Using DFS + DP

Once we know **which substrings are palindromes**, partitioning becomes straightforward.

---

## DFS State Definition

```
dfs(start, path)
```

Where:

* `start` → current index in string
* `path` → current list of palindrome substrings

---

## DFS Transition

For each `end` from `start` to `n-1`:

```
If dp[start][end] is True:
    choose s[start:end+1]
    recurse from end+1
```

---

## DFS Base Case

```
If start == n:
    path is a valid partition
```

---

## Full DP + DFS Solution (Python 3)

```python
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        
        # 1. Build palindrome DP table
        dp = [[False] * n for _ in range(n)]
        for length in range(1, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                if s[i] == s[j]:
                    if length <= 2:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]

        # 2. Backtracking using DP table
        res = []

        def dfs(start: int, path: List[str]):
            if start == n:
                res.append(path[:])
                return
            
            for end in range(start, n):
                if dp[start][end]:
                    dfs(end + 1, path + [s[start:end + 1]])

        dfs(0, [])
        return res
```

---

## Example Walkthrough: `s = "aab"`

### DFS Paths

```
start=0
 ├── "a" (0..0)
 │    ├── "a" (1..1)
 │    │     └── "b" (2..2) → ["a","a","b"]
 │    └── "ab" ❌
 └── "aa" (0..1)
      └── "b" (2..2) → ["aa","b"]
```

### Output

```
[
  ["a", "a", "b"],
  ["aa", "b"]
]
```

---

## Time & Space Complexity

### DP Construction

* Time: `O(n²)`
* Space: `O(n²)`

### DFS Enumeration

* Worst case (all characters same): `O(2ⁿ)` partitions
* DP ensures **palindrome checks are O(1)**

---

## Why DP Is Essential Here

Without DP:

* Palindrome check = `O(n)`
* Total complexity becomes **exponential × linear**, leading to TLE

With DP:

* Palindrome check = `O(1)`
* Only unavoidable exponential partition generation remains

---

If you want, I can also provide:

* Pure recursive vs DP comparison
* Bottom-up DP-only formulation (counting partitions)
* Backtracking tree diagram visualization
* Interview-ready explanation summary

Just tell me.