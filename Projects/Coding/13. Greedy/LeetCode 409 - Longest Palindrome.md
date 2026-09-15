---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 409: Longest Palindrome"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 409: Longest Palindrome

**LeetCode 409 – Longest Palindrome**, covering the problem statement, key observations, greedy reasoning, implementation details, and a fully worked example with step-by-step processing.

---

## LeetCode 409 – Longest Palindrome

### Problem Statement

Given a string `s` consisting of lowercase and/or uppercase English letters, return the **length of the longest palindrome**that can be built using the letters of `s`.

* Letters are **case-sensitive** (`'A'` and `'a'` are different).
* You may rearrange the letters.
* Each character can be used **at most as many times as it appears** in `s`.

---

### Key Observations

1. **Palindrome Structure**

   * A palindrome reads the same forward and backward.
   * Characters must appear in **pairs** (one on the left, one on the right).
   * At most **one character** may appear an **odd number of times**, placed in the **center**.
2. **Character Frequency Is Sufficient**

   * The order of characters in the string is irrelevant.
   * Only the **count (frequency)** of each character matters.
3. **Even vs Odd Counts**

   * If a character appears `k` times:

     * If `k` is even → all `k` can be used.
     * If `k` is odd → only `k - 1` can be used for pairs.
   * If **any** character has an odd count, we can place **exactly one** odd character in the center.

---

### Greedy Solution – Core Idea

This is a **greedy counting problem**.

Greedy decision:

* Use **as many pairs as possible** from each character.
* If **at least one odd count exists**, add **one extra character** to the total length (center of palindrome).

Why greedy works:

* Using pairs always increases palindrome length by 2.
* Using more than one odd-count character in the center is impossible.
* Therefore, the locally optimal choice (consume all pairs) leads to a globally optimal palindrome length.

---

### Algorithm Steps

1. Count frequency of each character.
2. Initialize:

   * `length = 0`
   * `has_odd = False`
3. For each character frequency `count`:

   * Add `(count // 2) * 2` to `length`
   * If `count` is odd → set `has_odd = True`
4. If `has_odd` is true:

   * Add `1` to `length`
5. Return `length`

---

### Python 3 Solution (With Typing)

```python
from collections import Counter
from typing import Dict

class Solution:
    def longestPalindrome(self, s: str) -> int:
        freq: Dict[str, int] = Counter(s)
        
        length: int = 0
        has_odd: bool = False
        
        for count in freq.values():
            # Use all possible pairs
            length += (count // 2) * 2
            
            # Check if there's at least one odd count
            if count % 2 == 1:
                has_odd = True
        
        # One odd character can be placed in the center
        if has_odd:
            length += 1
        
        return length
```

---

### Complete Worked Example (Step-by-Step)

#### Input

```
s = "abccccdd"
```

#### Step 1: Frequency Count

| Character | Count |
| --- | --- |
| a | 1 |
| b | 1 |
| c | 4 |
| d | 2 |

#### Step 2: Process Each Character

Initialize:

```
length = 0
has_odd = False
```

---

**Character `'a'` → count = 1**

* Pairs: `(1 // 2) * 2 = 0`
* Odd count → `has_odd = True`
* `length = 0`

---

**Character `'b'` → count = 1**

* Pairs: `0`
* Odd count → `has_odd` already true
* `length = 0`

---

**Character `'c'` → count = 4**

* Pairs: `(4 // 2) * 2 = 4`
* Even count
* `length = 4`

---

**Character `'d'` → count = 2**

* Pairs: `(2 // 2) * 2 = 2`
* Even count
* `length = 6`

---

#### Step 3: Add Center Character

* `has_odd = True` → add `1`

```
final length = 6 + 1 = 7
```

---

### Final Answer

```
7
```

#### Example Palindrome Construction

One valid palindrome:

```
dccaccd
```

(Exact order does not matter; only length matters.)

---

### Complexity Analysis

* **Time Complexity:** `O(n)`

  * Single pass to count characters.
* **Space Complexity:** `O(1)`

  * At most 52 characters (uppercase + lowercase English letters).

---

### Summary

* This problem is a **frequency-based greedy problem**.
* Use all possible pairs.
* Allow **only one odd-count character** in the center.
* Simple counting leads to an optimal solution.

If you want, I can also explain:

* Why **two odd centers are impossible**
* How this problem relates to **palindrome construction problems**
* Variations of this problem in interviews