---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 472: Concatenated Words"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 472: Concatenated Words

**LeetCode 472 – Concatenated Words**, with **precise state definition, transitions, DP table construction, and a worked example**. The explanation is intentionally formal and algorithmic.

---

## LeetCode 472 — Concatenated Words

### Problem Statement

Given an array of strings `words`, return all words that are **concatenations of at least two shorter words** from the same array.

* A word can be reused multiple times.
* A word **cannot count itself** as a component.

---

## Key Observation

This problem reduces to a **Word Break problem (LeetCode 139)** applied **independently to each word**, with a critical constraint:

> While checking a word `w`, we must **exclude `w` itself** from the dictionary.

Thus:

* Sort words by length
* Build a dictionary incrementally
* For each word, check if it can be formed using **previous (shorter) words**

---

## DP State Definition

For a given word `w` of length `n`:

```
dp[i] = True  if substring w[0:i] can be formed using words from dictionary
       False otherwise
```

* `i` represents a **prefix ending index**
* `dp[0] = True` → empty string is always valid

---

## DP Transition

For every `i` from `1` to `n`:

```
dp[i] = True
         if ∃ j such that:
             dp[j] == True
             AND w[j:i] ∈ dictionary
```

This means:

* We split `w[0:i]` at position `j`
* Left part is valid (`dp[j]`)
* Right part exists in dictionary

---

## Acceptance Condition

To ensure **at least two words are used**:

* The dictionary **does not include the word itself**
* If `dp[n] == True`, then the word is concatenated

---

## DP Table Creation (Worked Example)

### Input

```
words = ["cat", "cats", "dog", "catsdog"]
```

After sorting by length:

```
["cat", "dog", "cats", "catsdog"]
```

Dictionary before checking `"catsdog"`:

```
{"cat", "dog", "cats"}
```

---

### Word = `"catsdog"` (length = 7)

Indexing:

```
c a t s d o g
0 1 2 3 4 5 6
```

### DP Table

| i | Prefix | dp[i] | Reason |
| --- | --- | --- | --- |
| 0 | "" | True | Base case |
| 1 | "c" | False | Not in dict |
| 2 | "ca" | False | Not in dict |
| 3 | "cat" | True | "cat" ∈ dict |
| 4 | "cats" | True | "cats" ∈ dict |
| 5 | "catsd" | False | No valid split |
| 6 | "catsdo" | False | No valid split |
| 7 | "catsdog" | True | dp[4]=True + "dog" ∈ dict |

Final:

```
dp = [T, F, F, T, T, F, F, T]
```

Since `dp[7] == True`, `"catsdog"` is a **concatenated word**.

---

## Python 3 DP Solution (Typed)

```python
from typing import List, Set

class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        dictionary: Set[str] = set()
        result: List[str] = []

        def can_form(word: str) -> bool:
            n = len(word)
            dp = [False] * (n + 1)
            dp[0] = True

            for i in range(1, n + 1):
                for j in range(i):
                    if dp[j] and word[j:i] in dictionary:
                        dp[i] = True
                        break

            return dp[n]

        for word in words:
            if not word:
                continue

            if can_form(word):
                result.append(word)

            dictionary.add(word)

        return result
```

---

## Time & Space Complexity

### Let:

* `N` = number of words
* `L` = max word length

**Time**

```
O(N * L²)
```

Each word performs a full DP with nested splits.

**Space**

```
O(L) per word (DP array)
```

---

## Why This DP Is Correct

1. Ensures **prefix-based construction**
2. Prevents self-usage via incremental dictionary
3. Guarantees **≥ 2 words**, not trivial matches
4. Deterministic and order-independent after sorting

---

## Mental Model (One-Line)

> “For each word, ask: can I reach the end by jumping through dictionary words using DP?”

---

If you want next:

* Trie + DP optimization
* DFS vs DP comparison
* Visual DP split tree
* Dry run on a tricky case like `["a","aa","aaa","aaaa"]`

Say the word.