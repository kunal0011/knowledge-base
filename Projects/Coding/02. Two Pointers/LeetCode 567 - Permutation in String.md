---
date: "2025-12-22"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 567: Permutation in String"
tags:
  - leetcode
  - coding
  - two-pointers
---

# LeetCode 567: Permutation in String

Below is a structured, interview-grade explanation of **LeetCode 567 – Permutation in String**, following your requested format.

---

## LeetCode 567 – Permutation in String

### Problem Statement

You are given two strings `s1` and `s2`.  
Return `true` if `s2` contains a **permutation** of `s1` as a substring; otherwise, return `false`.

In other words, check whether any substring of `s2` is an anagram of `s1`.

**Constraints**

* `1 <= len(s1), len(s2) <= 10^4`
* Strings consist of lowercase English letters only.

---

## Key Observation

1. A permutation (anagram) means:

   * Same characters
   * Same frequencies
   * Order does **not** matter
2. Therefore:

   * Any valid substring in `s2` must have **exactly the same length** as `s1`
   * We only need to compare **character frequency counts**
3. Brute force approach:

   * Generate all substrings of length `len(s1)` in `s2`
   * Compare frequency arrays  
     This works but needs optimization to avoid recomputing counts repeatedly.

---

## Two Pointer (Sliding Window) Technique

### Core Idea

* Maintain a **window of size `len(s1)`** over `s2`
* Use two pointers:

  * `left` → start of window
  * `right` → end of window
* Slide the window one character at a time:

  * Add the new character (`right`)
  * Remove the old character (`left`)
* Compare frequency maps efficiently

### Why Sliding Window Works

* Window size is fixed
* Only **two characters change** when the window moves
* Allows `O(n)` time complexity

---

## Algorithm

1. If `len(s1) > len(s2)`, return `False`
2. Build frequency array of `s1`
3. Build frequency array of the first window in `s2`
4. If both frequency arrays match → return `True`
5. Slide the window:

   * Add `s2[right]`
   * Remove `s2[left]`
   * Compare frequencies
6. If no match found, return `False`

---

## Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        freq1: List[int] = [0] * 26
        freq2: List[int] = [0] * 26

        # build frequency for s1
        for ch in s1:
            freq1[ord(ch) - ord('a')] += 1

        window_size: int = len(s1)

        # initial window in s2
        for i in range(window_size):
            freq2[ord(s2[i]) - ord('a')] += 1

        if freq1 == freq2:
            return True

        left: int = 0

        # slide window
        for right in range(window_size, len(s2)):
            # add new character
            freq2[ord(s2[right]) - ord('a')] += 1
            # remove old character
            freq2[ord(s2[left]) - ord('a')] -= 1
            left += 1

            if freq1 == freq2:
                return True

        return False
```

---

## Worked Out Example

### Input

```
s1 = "ab"
s2 = "eidbaooo"
```

### Step-by-Step Execution

| Window | Substring | Frequency Match? |
| --- | --- | --- |
| [0–1] | "ei" | ❌ |
| [1–2] | "id" | ❌ |
| [2–3] | "db" | ❌ |
| [3–4] | "ba" | ✅ |

Explanation:

* `s1 = "ab"` → frequency: `{a:1, b:1}`
* Window `"ba"` has same frequencies
* Order does not matter → valid permutation found

### Output

```
True
```

---

## Time and Space Complexity

* **Time Complexity:** `O(n)`  
  (`n = len(s2)`; each character processed once)
* **Space Complexity:** `O(1)`  
  (Fixed 26-size arrays for lowercase letters)

---

## Key Interview Takeaways

* This is a **fixed-size sliding window** problem
* Focus on **frequency matching**, not string comparison
* Two pointers are used to maintain and shift the window efficiently
* Commonly paired with problems like:

  * LeetCode 438 (Find All Anagrams in a String)
  * LeetCode 76 (Minimum Window Substring – variable window)

If you want, I can also:

* Convert this to a **difference-count optimized version**
* Explain **why hashmap is slower than arrays here**
* Compare with **brute force vs sliding window**

Tell me how deep you want to go.