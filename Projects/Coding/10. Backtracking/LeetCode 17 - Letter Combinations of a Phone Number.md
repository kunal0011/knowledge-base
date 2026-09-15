---
date: "2025-12-14"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 17: Letter Combinations of a Phone Number"
tags:
  - leetcode
  - coding
  - backtracking
---

# LeetCode 17: Letter Combinations of a Phone Number

## LeetCode 17 — Letter Combinations of a Phone Number

### Problem Statement

Given a string `digits` containing digits from `2` to `9`, return **all possible letter combinations** that the number could represent based on the telephone keypad mapping. Return the answer in **any order**.

**Digit-to-Letter Mapping**

```
2 → abc
3 → def
4 → ghi
5 → jkl
6 → mno
7 → pqrs
8 → tuv
9 → wxyz
```

**Constraints**

* `0 <= digits.length <= 4`
* `digits[i]` is a digit in the range `'2'` to `'9'`.

**Example**

* Input: `digits = "23"`
* Output: `["ad","ae","af","bd","be","bf","cd","ce","cf"]`

---

### Key Observations

1. Each digit maps to a **set of characters**; the problem is to compute the **Cartesian product** of these sets.
2. The order of digits is fixed; at each position, we choose **one letter** from the corresponding mapping.
3. This is a classic **backtracking / DFS** problem:

   * Depth of recursion = `len(digits)`
   * Branching factor = number of letters for the current digit (3 or 4)
4. If `digits` is empty, there are **no combinations** → return an empty list.

---

### Approach (Backtracking)

* Maintain a mapping of digits to letters.
* Build the combination one character at a time.
* At recursion depth `index`:

  * Iterate through all letters for `digits[index]`.
  * Append one letter and recurse to `index + 1`.
* When `index == len(digits)`, a complete combination is formed.

---

### Python 3 Solution (with Typing)

```python
from typing import List, Dict

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        phone: Dict[str, str] = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }

        result: List[str] = []

        def backtrack(index: int, path: str) -> None:
            # Base case: one full combination formed
            if index == len(digits):
                result.append(path)
                return

            # Choose a letter for the current digit
            for ch in phone[digits[index]]:
                backtrack(index + 1, path + ch)

        backtrack(0, "")
        return result
```

---

### Example Walkthrough (`digits = "23"`)

Mappings:

```
2 → a b c
3 → d e f
```

Steps:

1. Start at index `0`, digit `'2'`
2. Choose `'a'`, move to index `1`
3. Choose `'d'` → `"ad"` ✓
4. Backtrack, choose `'e'` → `"ae"` ✓
5. Backtrack, choose `'f'` → `"af"` ✓
6. Backtrack to index `0`, choose `'b'`
7. Repeat for `"bd"`, `"be"`, `"bf"`
8. Repeat for `"c"` → `"cd"`, `"ce"`, `"cf"`

**Final Output**

```
["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

---

### Backtracking Tree Structure (`digits = "23"`)

![https://afteracademy.com/images/letter-combination-of-a-phone-number-example-tree-89debbca4854285a.png?utm_source=chatgpt.com](https://afteracademy.com/images/letter-combination-of-a-phone-number-example-tree-89debbca4854285a.png?utm_source=chatgpt.com)

![https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ADe1V_tw93TN3gwLyB0SH7Q.jpeg?utm_source=chatgpt.com](https://miro.medium.com/v2/resize%3Afit%3A1400/1%2ADe1V_tw93TN3gwLyB0SH7Q.jpeg?utm_source=chatgpt.com)

**Conceptual Tree**

```
""
                      |
          --------------------------------
          |              |              |
          a              b              c
        / | \          / | \          / | \
       ad ae af       bd be bf       cd ce cf
```

**Tree Characteristics**

* Each level corresponds to **one digit**.
* Each edge corresponds to choosing **one letter**.
* Leaf nodes represent **complete combinations**.
* Total leaves = product of letter counts per digit.

---

### Complexity Analysis

* **Time Complexity:**  
  `O(3^n * 4^m)`  
  where `n` = digits with 3 letters, `m` = digits with 4 letters.
* **Space Complexity:**  
  `O(n)` recursion depth (excluding output storage).

---

### Pattern Recognition

This problem exemplifies the **“Fixed Depth Combinatorial Backtracking”** pattern:

* Fixed number of decisions
* Independent choices at each level
* Cartesian product generation

## Contrast with classical backtracking (where popping IS required)

### Mutable state example (list)

```python
def backtrack(index: int, path: List[str]) -> None:
    if index == len(digits):
        result.append("".join(path))
        return

    for ch in phone[digits[index]]:
        path.append(ch)        # modify shared state
        backtrack(index + 1, path)
        path.pop()             # MUST undo change
```

### Why pop is required here

* `path` is a **single list object**
* All recursive calls reference the same list
* Without `pop()`, letters would accumulate incorrectly

This is **true backtracking**:

> choose → recurse → unchoose