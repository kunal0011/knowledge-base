---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 567: Permutation in String"
tags:
  - leetcode
  - coding
  - two-pointers
  - sliding-window
  - hash-table
  - string
  - amazon
  - google
  - meta
---

# LeetCode 567: Permutation in String

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Frequency Array

---

### Problem Statement

Given two strings `s1` and `s2`, return `true` if `s2` contains a **permutation** of `s1`, or `false` otherwise.

In other words, return `true` if one of `s1`'s permutations is the substring of `s2`.

---

### Input & Output Formats & Constraints

- **Input:** `s1: str`, `s2: str`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{s1.length}, \text{s2.length} \le 10^4$
  - `s1` and `s2` consist of lowercase English letters.

---

### Key Idea & Intuition

#### 1. Permutation Invariant:
A string $A$ is a permutation of $B$ if and only if:
1. $|A| == |B|$
2. The frequency of every character in $A$ matches that of $B$.

#### 2. Fixed-Size Sliding Window:
Instead of generating all $n_1!$ permutations of $s1$:
- We maintain a window of fixed width $W = |s_1|$ over $s_2$.
- If $|s_1| > |s_2|$, it is impossible for $s_2$ to contain a permutation of $s_1$, return `False`.
- We initialize the character counts for $s_1$ and the first window of $s_2$ ($s_2[0 \dots W-1]$) using two 26-element arrays `c1` and `c2`.
- If `c1 == c2`, return `True` immediately.
- As the window slides from index $W$ to $|s_2| - 1$:
  - Add the new incoming character $s_2[i]$: `c2[s2[i] - 'a']++`.
  - Remove the outgoing character $s_2[i - W]$: `c2[s2[i - W] - 'a']--`.
  - Compare `c1 == c2` in $\mathcal{O}(26) = \mathcal{O}(1)$ time.
- If no window matches by the end of the string, return `False`.

---

### Solution Approach (Step-by-Step)

1. **Length Guard:**
   - If `len(s1) > len(s2)`, return `False`.
2. **Initialize Count Arrays:**
   - Create `c1 = [0] * 26` and `c2 = [0] * 26`.
   - Populate `c1` and the initial window of `c2` for $i \in [0, |s_1| - 1]$.
3. **Check Initial Window:**
   - If `c1 == c2`, return `True`.
4. **Slide the Window Across $s_2$:**
   - For $i$ from $|s_1|$ to $|s_2| - 1$:
     - Add incoming: `c2[ord(s2[i]) - ord('a')] += 1`
     - Remove outgoing: `c2[ord(s2[i - len(s1)]) - ord('a')] -= 1`
     - If `c1 == c2`, return `True`.
5. **Fallback:**
   - Return `False`.

---

### Visual Algorithm Walkthrough

#### Example: `s1 = "ab"`, `s2 = "eidbaooo"` ($W = 2$)

```
s1 = "ab" -> Target counts: {'a': 1, 'b': 1}

Window 0: s2[0..1] = "ei"
  Counts: {'e': 1, 'i': 1} != Target -> Slide

Window 1: s2[1..2] = "id" (remove 'e', add 'd')
  Counts: {'i': 1, 'd': 1} != Target -> Slide

Window 2: s2[2..3] = "db" (remove 'i', add 'b')
  Counts: {'d': 1, 'b': 1} != Target -> Slide

Window 3: s2[3..4] = "ba" (remove 'd', add 'a')
  Counts: {'a': 1, 'b': 1} == Target!
  MATCH FOUND!
  Return True immediately.
```

---

### Solved Examples with Multiple Inputs

| `s1` | `s2` | Window Size | Matching Substring Found | Output |
| :--- | :--- | :--- | :--- | :--- |
| `"ab"` | `"eidbaooo"` | 2 | `"ba"` at indices 3..4 | `True` |
| `"ab"` | `"eidboaoo"` | 2 | None (`"bo"`, `"oa"` do not match) | `False` |
| `"adc"` | `"dcda"` | 3 | `"cda"` at indices 1..3 | `True` |
| `"hello"` | `"ooolleoooleh"` | 5 | None | `False` |
| `"a"` | `"a"` | 1 | `"a"` at index 0 | `True` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)
        if n1 > n2:
            return False
            
        c1, c2 = [0] * 26, [0] * 26
        for i in range(n1):
            c1[ord(s1[i]) - ord('a')] += 1
            c2[ord(s2[i]) - ord('a')] += 1
            
        if c1 == c2:
            return True
            
        for i in range(n1, n2):
            c2[ord(s2[i]) - ord('a')] += 1
            c2[ord(s2[i - n1]) - ord('a')] -= 1
            if c1 == c2:
                return True
                
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <vector>

class Solution {
public:
    bool checkInclusion(std::string s1, std::string s2) {
        int n1 = s1.size(), n2 = s2.size();
        if (n1 > n2) return false;

        std::vector<int> c1(26, 0), c2(26, 0);
        for (int i = 0; i < n1; ++i) {
            c1[s1[i] - 'a']++;
            c2[s2[i] - 'a']++;
        }
        if (c1 == c2) return true;

        for (int i = n1; i < n2; ++i) {
            c2[s2[i] - 'a']++;
            c2[s2[i - n1] - 'a']--;
            if (c1 == c2) return true;
        }
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public boolean checkInclusion(String s1, String s2) {
        int n1 = s1.length(), n2 = s2.length();
        if (n1 > n2) return false;

        int[] c1 = new int[26];
        int[] c2 = new int[26];

        for (int i = 0; i < n1; i++) {
            c1[s1.charAt(i) - 'a']++;
            c2[s2.charAt(i) - 'a']++;
        }
        if (Arrays.equals(c1, c2)) return true;

        for (int i = n1; i < n2; i++) {
            c2[s2.charAt(i) - 'a']++;
            c2[s2.charAt(i - n1) - 'a']--;
            if (Arrays.equals(c1, c2)) return true;
        }
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(|s_1| + 26 \cdot (|s_2| - |s_1|)) = \mathcal{O}(|s_2|)$. The window slides across $s_2$ one character at a time, performing an $\mathcal{O}(26) = \mathcal{O}(1)$ comparison on each shift.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space using two fixed 26-element arrays.

---

### Takeaway Pattern & Interview Traps

1. **Fixed Window vs Variable Window:**
   - Because we are matching an exact permutation of $s_1$, the substring length in $s_2$ **must** be exactly $|s_1|$. A fixed window of size $|s_1|$ ensures we never consider invalid substring lengths.
2. **$\mathcal{O}(1)$ Matches Counter Optimization:**
   - Rather than comparing all 26 entries (`c1 == c2`) each step, one can maintain a single integer `matches` representing the number of character positions (out of 26) where counts are identical, achieving pure $\mathcal{O}(1)$ window updates.
