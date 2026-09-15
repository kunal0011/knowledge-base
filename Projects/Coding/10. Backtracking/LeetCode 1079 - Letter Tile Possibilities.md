---
date: "2025-12-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 1079: Letter Tile Possibilities"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 1079: Letter Tile Possibilities

## LeetCode 1079 — Letter Tile Possibilities

---

### Problem Statement

You are given a string `tiles` representing **letter tiles**, where each tile has a letter printed on it.  
Return the number of **non-empty sequences** of letters you can make using the tiles.

**Rules**

* Each tile can be used **at most once**.
* The order of letters **matters**.
* The result counts **sequences**, not combinations.

**Constraints**

* `1 ≤ tiles.length ≤ 7`
* `tiles` consists of uppercase English letters.

**Example**

```text
Input: tiles = "AAB"
Output: 8
```

---

## Key Observations

1. This is a **permutation-based** problem, but:

   * We count **all lengths** (1 to n), not just full-length permutations.
2. Duplicate letters exist → naive permutation generation will **overcount**.
3. We must:

   * Sort the tiles
   * Skip duplicate choices **at the same recursion level**
4. Every **non-empty node** in the backtracking tree represents **one valid sequence**.

---

## Core Insight

> **Do not wait for a leaf node** to count a valid answer.  
> Every step that adds a character produces a valid sequence.

---

## Approach (Backtracking + Duplicate Pruning)

### State

* `used[i]`: whether tile at index `i` is already used in the current path
* `path`: current sequence (conceptual)

### Duplicate Pruning Rule

After sorting:

```
If tiles[i] == tiles[i - 1] AND used[i - 1] == False
→ skip
```

This prevents generating duplicate sequences at the same depth.

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        chars: List[str] = sorted(tiles)
        used = [False] * len(chars)
        count = 0

        def backtrack() -> None:
            nonlocal count

            for i in range(len(chars)):
                if used[i]:
                    continue

                # Skip duplicates at the same tree level
                if i > 0 and chars[i] == chars[i - 1] and not used[i - 1]:
                    continue

                used[i] = True
                count += 1              # every non-empty path is valid
                backtrack()
                used[i] = False

        backtrack()
        return count
```

---

## Example Explanation (`tiles = "AAB"`)

Sorted tiles:

```
['A', 'A', 'B']
```

Valid sequences:

```
A
A
B
AA
AB
BA
AAB
ABA
```

Total = **8**

---

## Complete Conceptual Backtracking Tree (`tiles = "AAB"`)

```
[]
                ------------------------------------------------
                |                         |                    |
              A(i=0)                    A(i=1) ❌              B(i=2)
                |                    (skipped)                  |
        ------------------                                  ------------------
        |                |                                  |                |
     A(i=1)           B(i=2)                              A(i=0)          A(i=1)
        |                |                                  |                |
     AA               AB                                  BA                ❌
        |                |
     AAB              ABA
```

---

### How to Read This Tree

* Each **node (except root)** = one valid sequence
* ❌ = duplicate branch skipped
* Vertical duplicates are allowed
* Horizontal duplicates are skipped

---

## Why This Tree Is Different from LC 46 / LC 47

| Problem | Count When |
| --- | --- |
| LC 46 | Only at leaf (length = n) |
| LC 47 | Only at leaf (unique full perms) |
| **LC 1079** | **Every non-empty node** |

---

## Complexity Analysis

* **Time Complexity:** `O(n! * n)` (upper bound)
* **Space Complexity:** `O(n)` recursion depth

---

## Pattern Recognition

This problem is a hybrid of:

* **Permutations with duplicates (LC 47)**
* **Prefix counting (count at every depth)**

---

### One-Line Interview Explanation

> “We generate permutations of all possible lengths using backtracking, and count each non-empty path while skipping duplicate branches at the same depth.”

If you want next, I can:

* Show how this reduces to **multiset permutations**
* Explain the **math-based counting approach**
* Compare the tree with **LC 47 side-by-side**