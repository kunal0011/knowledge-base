---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 139: Word Break"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 139: Word Break

**LeetCode 139 – Word Break**, focusing on **state definition, transition, DP table construction, and a worked example**.

---

## LeetCode 139 – Word Break

### Problem Statement

Given a string `s` and a list of words `wordDict`, determine whether `s` can be segmented into a **space-separated sequence of one or more dictionary words**.

Each word in the dictionary may be reused any number of times.

---

## Dynamic Programming Approach

This is a **1-D DP on prefix segmentation**.

---

## 1. DP State Definition

Let:

```
dp[i] = True if substring s[0 : i] can be segmented using wordDict
```

* `i` represents the **length** of the prefix
* `s[0:i]` is the prefix ending at index `i-1`
* Final answer: `dp[n]`, where `n = len(s)`

---

## 2. Base Case

```
dp[0] = True
```

Why?

* Empty string can always be segmented (choose no words)
* This is crucial to allow valid first word matches

---

## 3. State Transition

For each `i` from `1` to `n`:

```
dp[i] = True 
if there exists a j such that:
    dp[j] == True
    AND s[j:i] ∈ wordDict
```

### Transition Formula

```
dp[i] = OR over all j in [0, i):
           dp[j] AND (s[j:i] in wordDict)
```

### Interpretation

* Try to **cut** the string at position `j`
* If prefix `s[0:j]` is valid
* And substring `s[j:i]` is a dictionary word
* Then `s[0:i]` is valid

---

## 4. DP Table Creation (Step-by-Step Example)

### Example

```
s = "leetcode"
wordDict = ["leet", "code"]
```

Length:

```
n = 8
```

### DP Array (Index = prefix length)

| i | Prefix s[0:i] | dp[i] | Reason |
| --- | --- | --- | --- |
| 0 | "" | True | Base case |
| 1 | "l" | False | no word |
| 2 | "le" | False | no word |
| 3 | "lee" | False | no word |
| 4 | "leet" | True | "leet" ∈ dict and dp[0]=True |
| 5 | "leetc" | False | no valid split |
| 6 | "leetco" | False | no valid split |
| 7 | "leetcod" | False | no valid split |
| 8 | "leetcode" | True | dp[4]=True + "code" |

---

## 5. DP Visualization (Cut-Based Thinking)

```python
leetcode
^----^        "leet"  -> dp[4] = True
     ^----^   "code"  -> dp[8] = True
```

We confirm:

```
dp[4] = True AND "code" ∈ dict → dp[8] = True
```

---

## 6. Python 3 DP Solution (Typed)

```python
from typing import List, Set

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set: Set[str] = set(wordDict)
        n: int = len(s)

        dp: List[bool] = [False] * (n + 1)
        dp[0] = True  # base case

        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]
```

---

## 7. Time and Space Complexity

### Time

```
O(n²) substring checks
```

* `n` states
* each checks up to `n` previous splits
* substring lookup is `O(1)` with a set

### Space

```
O(n)
```

* DP array only

---

## 8. Why This DP Works (Key Insight)

* We are **not building words**
* We are validating **prefix feasibility**
* Every valid segmentation is composed of:

  * a valid prefix
  * followed by a valid dictionary word

This is a **classic prefix DP** problem.

---

## 9. Common Pitfalls

1. Forgetting `dp[0] = True`
2. Using recursion without memo → TLE
3. Checking words first instead of prefix DP
4. Assuming greedy works (it does not)

---

If you want, I can next:

* Optimize using **max word length**
* Show **BFS vs DP comparison**
* Explain **why greedy fails**
* Extend this to **LeetCode 140 (Word Break II)**

Tell me how deep you want to go.