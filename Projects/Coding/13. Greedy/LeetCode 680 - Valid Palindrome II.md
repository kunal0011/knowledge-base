---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 680: Valid Palindrome II"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 680: Valid Palindrome II

Below is a **complete, interview-grade explanation** of **LeetCode 680 – Valid Palindrome II**, structured exactly as requested.

---

## LeetCode 680 – Valid Palindrome II

---

## Problem Statement

Given a string `s`, return `True` if the string can be a palindrome **after deleting at most one character**.  
Otherwise, return `False`.

### Definition

A string is a palindrome if it reads the same forward and backward.

### Constraints

* `1 ≤ s.length ≤ 10^5`
* `s` consists of lowercase English letters.

---

## Key Observation

1. **A palindrome fails at the first mismatch** between characters at symmetric positions.
2. Since **only one deletion is allowed**, once a mismatch occurs, we have **exactly two valid choices**:

   * Delete the **left** character
   * Delete the **right** character
3. After making one deletion, the **remaining substring must be a perfect palindrome**.

This reduces the problem to:

> Can we skip **one** character and still have a palindrome?

---

## Why Greedy Works Here

* We compare characters from **both ends** using two pointers.
* As long as characters match, we move inward.
* At the **first mismatch**, we try both deletion options:

  * Skip `s[left]`
  * Skip `s[right]`
* If **either** resulting substring is a palindrome, the answer is `True`.

### Important Greedy Insight

We **do not backtrack deeply**.  
We only branch **once**, because only **one deletion is allowed**.

This guarantees:

* **O(n)** time complexity
* **O(1)** extra space

---

## Greedy Strategy (Two-Pointer)

### Algorithm

1. Initialize `left = 0`, `right = len(s) - 1`
2. While `left < right`:

   * If `s[left] == s[right]`, move inward
   * Else:

     * Check if substring `(left+1, right)` is palindrome
     * OR substring `(left, right-1)` is palindrome
3. If we never fail, return `True`

---

## Python 3 Solution (With Typing)

```python
from typing import *

class Solution:
    def validPalindrome(self, s: str) -> bool:
        def is_palindrome(l: int, r: int) -> bool:
            while l < r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True

        left, right = 0, len(s) - 1

        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                # Try deleting one character
                return is_palindrome(left + 1, right) or is_palindrome(left, right - 1)

        return True
```

---

## Complete Worked Example (Step-by-Step)

### Example

```
s = "abca"
```

---

### Initial State

| Pointer | Index | Character |
| --- | --- | --- |
| left | 0 | 'a' |
| right | 3 | 'a' |

✔ Match → Move inward

---

### Step 2

| Pointer | Index | Character |
| --- | --- | --- |
| left | 1 | 'b' |
| right | 2 | 'c' |

❌ Mismatch found

We have **one deletion allowed**, so we try both possibilities.

---

### Option 1: Delete `s[left]` ('b')

Check substring:

```
s[2:3] → "c"
```

✔ Single character → Palindrome

---

### Option 2: Delete `s[right]` ('c')

Check substring:

```
s[1:1] → "b"
```

✔ Single character → Palindrome

---

### Final Result

At least **one deletion path succeeds**  
→ **Return True**

---

## Failure Example

### Input

```
s = "abc"
```

Mismatch at `'a'` vs `'c'`

* Delete `'a'` → `"bc"` ❌
* Delete `'c'` → `"ab"` ❌

Both fail → **Return False**

---

## Complexity Analysis

| Metric | Value |
| --- | --- |
| Time Complexity | **O(n)** |
| Space Complexity | **O(1)** |
| Greedy | ✔ |
| Two-Pointer | ✔ |

---

## Interview Notes

* This problem is a **classic greedy + two-pointer hybrid**
* Key trick: **branch only once**
* Avoid recursion or DP – unnecessary
* Handle mismatch immediately

---

If you want, I can also provide:

* Edge case walkthroughs
* Visual pointer movement diagram
* Comparison with brute force or DP
* Why deleting more than one breaks greedy

Just let me know.