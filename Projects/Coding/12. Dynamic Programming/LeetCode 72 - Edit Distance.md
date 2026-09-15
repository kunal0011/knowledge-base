---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 72: Edit Distance"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 72: Edit Distance

**LeetCode 72 – Edit Distance** using **Dynamic Programming**, covering **state definition, transitions, DP table construction, and a worked example**.

---

## Problem Statement (LeetCode 72 – Edit Distance)

Given two strings `word1` and `word2`, return the **minimum number of operations** required to convert `word1` into `word2`.

Allowed operations:

1. **Insert** a character
2. **Delete** a character
3. **Replace** a character

---

## Key Observation

At any position, the problem reduces to:

> “What is the minimum cost to convert a **prefix of `word1`** into a **prefix of `word2`**?”

This naturally suggests a **2D Dynamic Programming** formulation over prefixes.

---

## DP State Definition

Let:

```
dp[i][j] = minimum number of operations required
           to convert word1[0..i-1] into word2[0..j-1]
```

Important notes:

* `i` represents length of prefix from `word1`
* `j` represents length of prefix from `word2`
* Final answer = `dp[n][m]`

---

## Base Cases (DP Initialization)

1. **Converting empty string to word2 prefix**

```
dp[0][j] = j     (insert all j characters)
```

2. **Converting word1 prefix to empty string**

```
dp[i][0] = i     (delete all i characters)
```

These form the **first row and first column** of the DP table.

---

## State Transition

For `i > 0` and `j > 0`:

### Case 1: Characters match

If:

```
word1[i-1] == word2[j-1]
```

Then:

```
dp[i][j] = dp[i-1][j-1]
```

(No operation required)

---

### Case 2: Characters do not match

We consider **three operations**:

1. **Insert**  
   Insert `word2[j-1]` into `word1`

   ```
   dp[i][j-1] + 1
   ```
2. **Delete**  
   Delete `word1[i-1]`

   ```
   dp[i-1][j] + 1
   ```
3. **Replace**  
   Replace `word1[i-1]` with `word2[j-1]`

   ```
   dp[i-1][j-1] + 1
   ```

Take the minimum:

```
dp[i][j] = 1 + min(
    dp[i][j-1],   // insert
    dp[i-1][j],   // delete
    dp[i-1][j-1]  // replace
)
```

---

## Full DP Formula

```
dp[i][j] =
    if word1[i-1] == word2[j-1]:
        dp[i-1][j-1]
    else:
        1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
```

---

## Example Walkthrough

### Input

```
word1 = "horse"
word2 = "ros"
```

Lengths:

```
n = 5, m = 3
```

---

## DP Table Construction

Rows = `horse` (including empty prefix)  
Columns = `ros` (including empty prefix)

![Image](https://www.interviewbit.com/blog/wp-content/uploads/2021/11/Image-1-9-1024x712.png?utm_source=chatgpt.com)

![Image](https://afteracademy.com/images/edit-distance-table1-e74d16337d56f3ae.png?utm_source=chatgpt.com)

![Image](https://labuladong.online/algo/images/editDistance/dp.jpg?utm_source=chatgpt.com)

### DP Table (Values)

|  | "" | r | o | s |
| --- | --- | --- | --- | --- |
| "" | 0 | 1 | 2 | 3 |
| h | 1 | 1 | 2 | 3 |
| o | 2 | 2 | 1 | 2 |
| r | 3 | 2 | 2 | 2 |
| s | 4 | 3 | 3 | 2 |
| e | 5 | 4 | 4 | 3 |

---

## How Cells Are Computed (Sample)

### Cell `dp[2][2]` → `"ho"` → `"ro"`

Characters:

```
word1[1] = 'o'
word2[1] = 'o'
```

They match:

```
dp[2][2] = dp[1][1] = 1
```

---

### Cell `dp[5][3]` → `"horse"` → `"ros"`

Characters:

```
'e' != 's'
```

Operations:

```
Insert  → dp[5][2] + 1 = 4 + 1 = 5
Delete  → dp[4][3] + 1 = 2 + 1 = 3
Replace → dp[4][2] + 1 = 3 + 1 = 4
```

Minimum:

```
dp[5][3] = 3
```

---

## Final Answer

```
Minimum Edit Distance = dp[5][3] = 3
```

---

## Python 3 Implementation (With Typing)

```python
from typing import List

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)

        dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]

        # Base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i][j - 1],   # insert
                        dp[i - 1][j],   # delete
                        dp[i - 1][j - 1]  # replace
                    )

        return dp[n][m]
```

---

## Complexity Analysis

* **Time Complexity:** `O(n × m)`
* **Space Complexity:** `O(n × m)`

  * Can be optimized to `O(m)` using rolling arrays (ask if needed)

---

## Mental Model (Interview Tip)

> “Each DP cell answers: *What is the cheapest way to align these two prefixes?*  
> Character match → move diagonally  
> Character mismatch → try insert, delete, replace”

---

If you want, I can next provide:

* Space-optimized 1D DP version
* DP path reconstruction (actual edit operations)
* Visual backtracking tree of transitions
* Comparison with LCS-based intuition