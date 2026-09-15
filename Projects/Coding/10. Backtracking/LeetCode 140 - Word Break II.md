---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 140: Word Break II"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 140: Word Break II

## LeetCode 140 — Word Break II

---

### Problem Statement

Given a string `s` and a dictionary of strings `wordDict`, return **all possible sentences** where `s` is segmented into a sequence of dictionary words.

* Each word in `wordDict` can be used **multiple times**.
* Words must cover the string **completely and contiguously**.
* Return all valid sentences in **any order**.

**Example**

```text
Input:
s = "catsanddog"
wordDict = ["cat","cats","and","sand","dog"]

Output:
[
  "cats and dog",
  "cat sand dog"
]
```

---

## Key Observations (Critical)

1. This is **not just Word Break I (LC 139)**:

   * LC 139 → boolean (can break or not)
   * LC 140 → **enumerate all valid segmentations**
2. This is a **prefix-based backtracking problem**:

   * At each index, try all dictionary words that match the prefix starting there.
3. Naive backtracking is **exponential** due to overlapping subproblems.
4. **Memoization (top-down DP)** is mandatory to avoid TLE:

   * Cache results for each starting index.
5. The backtracking tree is a **directed acyclic graph (DAG)**, not a pure tree.

---

## Core Idea

Define a function:

> `dfs(start)` → all valid sentences that can be formed from `s[start:]`

For each word in `wordDict`:

* If `s[start:]` starts with `word`

  * Recurse on `start + len(word)`
  * Append `word` in front of returned sentences

---

## Python 3 Solution (Backtracking + Memoization)

```python
from typing import List, Dict

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        word_set = set(wordDict)
        memo: Dict[int, List[str]] = {}

        def dfs(start: int) -> List[str]:
            if start in memo:
                return memo[start]

            if start == len(s):
                return [""]  # base case: valid sentence end

            sentences: List[str] = []

            for word in word_set:
                if s.startswith(word, start):
                    sub_sentences = dfs(start + len(word))
                    for sub in sub_sentences:
                        if sub:
                            sentences.append(word + " " + sub)
                        else:
                            sentences.append(word)

            memo[start] = sentences
            return sentences

        return dfs(0)
```

---

## Example Walkthrough

### Input

```
s = "catsanddog"
wordDict = ["cat","cats","and","sand","dog"]
```

---

### Valid Segmentations

1. `"cats" → "and" → "dog"`
2. `"cat" → "sand" → "dog"`

---

## Complete Conceptual Backtracking Tree

(**Navigation view — full tree, no code-level pruning shown**)

![https://assets.algo.monster/fib.001.png?utm_source=chatgpt.com](https://images.openai.com/thumbnails/url/3kYbBHicu5mVUVJSUGylr5-al1xUWVCSmqJbkpRnoJdeXJJYkpmsl5yfq5-Zm5ieWmxfaAuUsXL0S7F0Tw7Mq0guKHFKc7LwyXNNMs7PyPFLT3LMd60y8c7INSrJtyzMLjNML8spDHYuck4uC08pUSsGAKAuJ7Y?utm_source=chatgpt.com)

![https://media.geeksforgeeks.org/wp-content/cdn-uploads/wordBreak1.png?utm_source=chatgpt.com](https://media.geeksforgeeks.org/wp-content/cdn-uploads/wordBreak1.png?utm_source=chatgpt.com)

![https://deen3evddmddt.cloudfront.net/uploads/content-images/segment-tree.webp?utm_source=chatgpt.com](https://deen3evddmddt.cloudfront.net/uploads/content-images/segment-tree.webp?utm_source=chatgpt.com)

### Conceptual Tree (Textual Representation)

```
Start: "catsanddog"
│
├── "cat"
│     └── "sanddog"
│           └── "sand"
│                 └── "dog"
│                       └── END
│
└── "cats"
      └── "anddog"
            └── "and"
                  └── "dog"
                        └── END
```

---

### Tree Interpretation

* **Each node** represents a remaining suffix of the string.
* **Each edge** represents choosing a dictionary word that matches the prefix.
* **Leaf nodes** (`END`) indicate a complete valid sentence.
* Paths from root to leaf form one valid sentence.

---

## Why Memoization Is Essential

### Without Memoization

```
"dog" subtree explored multiple times
"sanddog" subtree explored multiple times
```

### With Memoization

* Each `start index` is computed **once**
* Tree collapses into a **DAG**
* Performance becomes manageable

---

## Complexity Analysis

* **Time Complexity:** Exponential in worst case, but **greatly reduced** by memoization
* **Space Complexity:**

  * `O(n)` recursion depth
  * Output size dominates space

---

## Pattern Recognition

LeetCode 140 is a classic example of:

> **Backtracking on strings + memoization + sentence reconstruction**

This same pattern appears in:

* Palindrome Partitioning II (LC 132)
* Restore IP Addresses (LC 93)
* Expression Add Operators (LC 282)

---

## One-Line Interview Explanation

> “We perform DFS on the string, trying all dictionary prefixes, and memoize results per index to avoid recomputation. Each root-to-leaf path forms a valid sentence.”

If you want next:

* Contrast **LC 139 vs LC 140**
* Show **exact TLE behavior without memo**
* Convert this into a **generic string-DP backtracking template**