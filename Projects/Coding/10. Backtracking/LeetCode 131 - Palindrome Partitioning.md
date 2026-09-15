---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 131: Palindrome Partitioning"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 131: Palindrome Partitioning

## LeetCode 131 — Palindrome Partitioning

---

### Problem Statement

Given a string `s`, partition `s` such that **every substring of the partition is a palindrome**.  
Return **all possible palindrome partitionings** of `s`.

**Constraints**

* `1 ≤ s.length ≤ 16`
* `s` consists of lowercase English letters

**Example**

```text
Input: s = "aab"
Output:
[
  ["a","a","b"],
  ["aa","b"]
]
```

---

## Key Observations

1. We must split the string into **contiguous substrings**.
2. Every chosen substring must satisfy the **palindrome property**.
3. This is a **partitioning backtracking problem**, not combinations or permutations.
4. The decision at each index is:

   > “Where should I cut next?”
5. Overlapping palindrome checks can be optimized using **DP precomputation**.

---

## Core Insight

At position `start`, try **all substrings**:

```
s[start : end + 1]   for end ∈ [start … n-1]
```

If the substring is a palindrome:

* Choose it
* Recurse from `end + 1`

---

## Approach 1 — Backtracking (Straightforward)

### Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        result: List[List[str]] = []
        path: List[str] = []
        n = len(s)

        def is_palindrome(left: int, right: int) -> bool:
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True

        def backtrack(start: int) -> None:
            if start == n:
                result.append(path.copy())
                return

            for end in range(start, n):
                if is_palindrome(start, end):
                    path.append(s[start:end + 1])
                    backtrack(end + 1)
                    path.pop()

        backtrack(0)
        return result
```

---

## Example Walkthrough (`s = "aab"`)

### Step-by-step

1. Start at index `0`
2. `"a"` → palindrome → choose

   * Start at index `1`
   * `"a"` → palindrome → choose

     * Start at index `2`
     * `"b"` → palindrome → choose → **valid**
   * Backtrack
3. From index `0`, try `"aa"` → palindrome → choose

   * Start at index `2`
   * `"b"` → palindrome → choose → **valid**

**Output**

```
["a","a","b"]
["aa","b"]
```

---

## Backtracking Tree Structure (Navigation)

![https://i.ytimg.com/vi/fhXo6BDTIJc/maxresdefault.jpg?utm_source=chatgpt.com](https://i.ytimg.com/vi/fhXo6BDTIJc/maxresdefault.jpg?utm_source=chatgpt.com)

![https://static.takeuforward.org/content/6.png-gYwsPCwz?utm_source=chatgpt.com](https://static.takeuforward.org/content/6.png-gYwsPCwz?utm_source=chatgpt.com)

![https://miro.medium.com/v2/resize%3Afit%3A930/1%2AO-ej91KzUFIY9uwHkrwePQ.jpeg?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A930/1%2AO-ej91KzUFIY9uwHkrwePQ.jpeg?utm_source=chatgpt.com)

### Conceptual Tree

```
start=0
                               |
            -----------------------------------
            |                                 |
           "a"                              "aa"
            |                                 |
         start=1                           start=2
            |                                 |
         ----------                          "b"
         |        |                           |
       "a"       ❌                        ["aa","b"]
         |
      start=2
         |
        "b"
         |
  ["a","a","b"]
```

---

## Approach 2 — Backtracking + DP Optimization

### Why DP Helps

Palindrome checking costs `O(n)` each time.  
Precomputing palindromes reduces it to `O(1)`.

---

### DP Table

```
dp[i][j] = True if s[i:j+1] is a palindrome
```

---

### Optimized Python Code

```python
from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        dp = [[False] * n for _ in range(n)]

        # Precompute palindromes
        for end in range(n):
            for start in range(end + 1):
                if s[start] == s[end] and (end - start <= 2 or dp[start + 1][end - 1]):
                    dp[start][end] = True

        result: List[List[str]] = []
        path: List[str] = []

        def backtrack(start: int) -> None:
            if start == n:
                result.append(path.copy())
                return

            for end in range(start, n):
                if dp[start][end]:
                    path.append(s[start:end + 1])
                    backtrack(end + 1)
                    path.pop()

        backtrack(0)
        return result
```

---

## Complexity Analysis

| Version | Time | Space |
| --- | --- | --- |
| Plain Backtracking | `O(n · 2^n)` | `O(n)` |
| With DP | `O(2^n)` | `O(n^2)` |

---

## Pattern Recognition

LeetCode 131 is a **Partition-Based Backtracking** problem:

* Fixed traversal direction
* Variable cut points
* Validation condition on each segment

Related problems:

* LC 132 (Palindrome Partitioning II — DP optimization)
* LC 93 (Restore IP Addresses)
* LC 784 (Letter Case Permutation – partial partitioning)

---

### One-Line Interview Explanation

> “At each index, we try every possible cut that forms a palindrome and recursively partition the remaining suffix.”

If you want next, I can:

* Draw **full recursion tree for a longer input**
* Explain why **greedy fails**
* Compare **LC 131 vs LC 132** in detail