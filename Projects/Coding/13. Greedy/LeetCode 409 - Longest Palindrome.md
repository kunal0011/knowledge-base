---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 409: Longest Palindrome"
tags:
  - leetcode
  - coding
  - greedy
  - hash-table
  - string
  - counting
  - amazon
  - google
---

# LeetCode 409: Longest Palindrome

**Target Companies:** Amazon, Google, Microsoft, Apple, Bloomberg  
**Difficulty:** Easy  
**Topic:** Greedy / Hash Table / String / Counting  

---

### Problem Statement

Given a string `s` which consists of lowercase or uppercase letters, return the length of the **longest palindrome** that can be built with those letters.

Letters are **case sensitive**, for example, `"Aa"` is not considered a palindrome.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 2000$).
- **Output:**
  - `int` — the maximum possible length of a palindrome formed by rearranging characters from `s`.
- **Constraints:**
  - $1 \le \text{s.length} \le 2000$
  - `s` consists of lowercase and/or uppercase English letters only.

---

### Key Idea & Intuition

A palindrome reads identically forward and backward:
1. Every character off-center must appear in symmetric **pairs** (one on the left wing, one mirrored on the right wing).
2. At most **one** character can have an odd count, acting as the singular **center pivot**.

#### The Greedy Character Utilization Invariant:
For each unique character with frequency $c$:
- We can always form $\lfloor c / 2 \rfloor \times 2$ symmetric pairs:
  $$\text{paired\_count} = c - (c \pmod 2)$$
- If $c$ is odd ($c \pmod 2 == 1$), one instance of this character is leftover and cannot be paired.
- If there is **at least one** character with an odd frequency across the entire string, we can place exactly one odd leftover in the center of the palindrome, adding $+1$ to our total length.
- Further leftovers cannot be used because a palindrome cannot have more than one center pivot.

Thus, the maximum palindrome length is:
$$\text{Length} = \sum_{c \in \text{freq}} (c - (c \pmod 2)) + \mathbb{I}(\exists c \text{ such that } c \pmod 2 == 1)$$

---

### Solution Approach (Step-by-Step)

1. Count frequencies of all characters in `s`.
2. Initialize `length = 0` and `has_odd = False`.
3. For each frequency `count` in character counts:
   - If `count % 2 == 0`:
     - `length += count`
   - Else:
     - `length += count - 1`
     - `has_odd = True`
4. If `has_odd`:
   - `length += 1`
5. Return `length`.

---

### Visual Algorithm Walkthrough

For `s = "abccccdd"`:

```
Character Frequencies:
  'a': 1
  'b': 1
  'c': 4
  'd': 2

Evaluation:
  'a' (count 1): odd -> add 0, has_odd = True
  'b' (count 1): odd -> add 0, has_odd = True
  'c' (count 4): even -> add 4 (total: 4)
  'd' (count 2): even -> add 2 (total: 6)

After loop:
  length = 6
  has_odd = True -> add 1 for center pivot
  Total Length = 7

One optimal palindrome construction:
  "dccaccd" or "dccbccd" (length 7).
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "abccccdd"`
- **Output:** `7`

#### Example 2:
- **Input:** `s = "a"`
- **Tracing:** Frequency: `{'a': 1}`. 1 is odd $\rightarrow$ length = 1.
- **Output:** `1`

#### Example 3 (All Characters Even):
- **Input:** `s = "bb"`
- **Tracing:** Frequency: `{'b': 2}`. No odd leftover.
- **Output:** `2`

#### Example 4 (Case Sensitivity):
- **Input:** `s = "Aa"`
- **Tracing:** `'A': 1, 'a': 1`. Only one can be in center.
- **Output:** `1`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter

class Solution:
    def longestPalindrome(self, s: str) -> int:
        counts = Counter(s)
        length = 0
        has_odd = False
        
        for count in counts.values():
            length += count // 2 * 2
            if count % 2 == 1:
                has_odd = True
                
        return length + 1 if has_odd else length
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    int longestPalindrome(const std::string& s) {
        std::vector<int> counts(128, 0);
        for (char c : s) {
            counts[static_cast<unsigned char>(c)]++;
        }
        
        int length = 0;
        bool has_odd = false;
        
        for (int count : counts) {
            length += (count / 2) * 2;
            if (count % 2 == 1) {
                has_odd = true;
            }
        }
        
        return has_odd ? length + 1 : length;
    }
};
```

#### Java 17
```java
class Solution {
    public int longestPalindrome(String s) {
        int[] counts = new int[128];
        for (int i = 0; i < s.length(); i++) {
            counts[s.charAt(i)]++;
        }
        
        int length = 0;
        boolean hasOdd = false;
        
        for (int count : counts) {
            length += (count / 2) * 2;
            if (count % 2 == 1) {
                hasOdd = true;
            }
        }
        
        return hasOdd ? length + 1 : length;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Scanning string `s` of length $n$ to populate counts takes $\mathcal{O}(n)$ time.
  - Iterating over the 128 ASCII buckets takes $\mathcal{O}(1)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The array or hash table stores at most 52 unique English letters (uppercase and lowercase).

---

### Takeaway Pattern & Interview Traps

- **Case Sensitivity:** `'A'` and `'a'` are distinct characters and must not be grouped together. An array of size 128 indexed by ASCII character values avoids case normalization pitfalls.
- **One-Liner Alternative:** Notice that every odd character wastes exactly 1 element, except for the single center element. Thus:
  $$\text{Length} = n - \max(0, \text{odd\_count} - 1)$$
  where `odd_count` is the number of characters with odd frequency!