---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 140: Word Break II"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 140: Word Break II

**LeetCode 140 (Word Break II)** with **precise state definition, transition logic, DP table construction, and a worked example**.  
The focus is on *how DP models the problem*, not just code.

---

## LeetCode 140 — Word Break II

### Problem Statement

Given:

* a string `s`
* a list of strings `wordDict`

Return **all possible sentences** where:

* the sentence is formed by inserting spaces into `s`
* every word is present in `wordDict`

Order of output does not matter.

---

## Key Insight (Why DP is Needed)

This is **not a yes/no problem** (unlike LeetCode 139).  
We must **construct all valid sentences**, which implies:

* Multiple decompositions
* Overlapping subproblems
* Exponential combinations without memoization

Hence, we use **DP where each state stores all possible sentences for a suffix**.

---

## DP State Definition

Let:

```
dp[i] = list of all valid sentences that can be formed from substring s[i:]
```

Where:

* `i` ranges from `0` to `len(s)`
* `dp[n] = [""]` (base case: empty string has one valid decomposition)

---

## DP Transition

For a given index `i`:

1. Try **every word** in `wordDict`
2. If `s[i:]` starts with `word`
3. Then:

   * append `word` in front of each sentence in `dp[i + len(word)]`

Formally:

```
dp[i] += word + (" " if suffix != "" else "") + suffix
         for each suffix in dp[i + len(word)]
```

---

## Base Case

```
dp[len(s)] = [""]
```

This allows sentence building to terminate cleanly without extra spaces.

---

## DP Table Construction Order

We fill the DP table **bottom-up**:

```
i = len(s) → 0
```

Reason:

* `dp[i]` depends on `dp[i + len(word)]`
* Suffix results must be computed first

---

## Example Walkthrough

### Input

```
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
```

Length `n = 10`

---

### Step 1: Initialize DP Table

| Index | Substring | dp[i] |
| --- | --- | --- |
| 10 | "" | [""] |
| 0–9 | — | [] |

---

### Step 2: Fill DP Bottom-Up

#### i = 7 → `"dog"`

* matches `"dog"`
* `dp[7] = ["dog"]`

---

#### i = 4 → `"sanddog"`

* matches `"sand"`
* `dp[4] = ["sand dog"]`

---

#### i = 3 → `"sanddog"`

* matches `"sand"`
* `dp[3] = ["sand dog"]`

---

#### i = 0 → `"catsanddog"`

Matches:

* `"cat"` → `dp[3] = ["sand dog"]`
* `"cats"` → `dp[4] = ["sand dog"]`

So:

```
dp[0] = [
  "cat sand dog",
  "cats sand dog"
]
```

---

## Final DP Table Snapshot

| i | s[i:] | dp[i] |
| --- | --- | --- |
| 10 | "" | [""] |
| 7 | "dog" | ["dog"] |
| 4 | "sanddog" | ["sand dog"] |
| 3 | "sanddog" | ["sand dog"] |
| 0 | "catsanddog" | ["cat sand dog", "cats sand dog"] |

---

## Final Answer

```
[
  "cat sand dog",
  "cats sand dog"
]
```

---

## Python 3 DP Solution (Typed, Bottom-Up)

```python
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        n = len(s)
        word_set = set(wordDict)

        # dp[i] = all sentences formed from s[i:]
        dp: List[List[str]] = [[] for _ in range(n + 1)]
        dp[n] = [""]  # base case

        for i in range(n - 1, -1, -1):
            for word in word_set:
                if s.startswith(word, i):
                    for suffix in dp[i + len(word)]:
                        if suffix == "":
                            dp[i].append(word)
                        else:
                            dp[i].append(word + " " + suffix)

        return dp[0]
```

---

## Complexity Analysis

### Time Complexity

* Worst case: **O(N × W × S)**  
  Where:

  * `N` = length of string
  * `W` = number of dictionary words
  * `S` = number of generated sentences (can be exponential)

### Space Complexity

* **O(N × S)** due to storing all sentence combinations

---

## Why This DP Works (Conceptual Summary)

* Each index represents a **sentence boundary**
* DP stores **complete sentence solutions**, not booleans
* Bottom-up avoids recomputation
* Base case ensures clean concatenation

---

If you want next:

* **Top-down DP + memo recursion**
* **Why pure backtracking TLEs**
* **Optimization using Trie**
* **Relation between LeetCode 139 and 140**

Tell me which direction you want.