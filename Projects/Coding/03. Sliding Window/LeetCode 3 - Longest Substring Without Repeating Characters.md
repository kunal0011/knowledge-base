---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 3: Longest Substring Without Repeating Characters"
tags:
  - leetcode
  - coding
  - sliding-window
  - amazon
  - google
---

# LeetCode 3: Longest Substring Without Repeating Characters

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Dynamic Sliding Window / Last Seen Index Map

---

### Problem Statement

Given a string `s`, find the length of the **longest substring** without repeating characters.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `int` (maximum length of a non-repeating contiguous substring)
- **Constraints:**
  - $0 \le \text{s.length} \le 5 \times 10^4$
  - `s` consists of English letters, digits, symbols and spaces.

---

### Key Idea & Intuition

- **Sliding Window Invariant:**
  - Maintain a window `[left, right]` containing only unique characters.
- **Direct Jump Optimization:**
  - Instead of shrinking `left` one step at a time when a duplicate is found, store the **last seen index** of every character in a hash map `char_map`.
  - When character $s[\text{right}]$ is already in `char_map` and its previous index is $\ge \text{left}$:
    - Instantly jump the `left` pointer to `char_map[s[right]] + 1`!
  - Update `char_map[s[right]] = right`.
  - Record `max_len = max(max_len, right - left + 1)`.

---

### Solution Approach (Step-by-Step)

1. Initialize `char_map = {}`, `left = 0`, `max_len = 0`.
2. For `right, ch` in `enumerate(s)`:
   - If `ch in char_map and char_map[ch] >= left`:
     - `left = char_map[ch] + 1`
   - `char_map[ch] = right`
   - `max_len = max(max_len, right - left + 1)`
3. Return `max_len`.

---

### Visual Algorithm Walkthrough

```
s = "pwwkew"

right=0 ('p'): left=0, char_map={'p':0}, len=1 ("p")
right=1 ('w'): left=0, char_map={'p':0, 'w':1}, len=2 ("pw")
right=2 ('w'): 'w' seen at idx 1 >= left(0) -> Jump left = 2.
               char_map={'p':0, 'w':2}, len=1 ("w")
right=3 ('k'): left=2, char_map={'p':0, 'w':2, 'k':3}, len=2 ("wk")
right=4 ('e'): left=2, char_map={'p':0, 'w':2, 'k':3, 'e':4}, len=3 ("wke")
right=5 ('w'): 'w' seen at idx 2 >= left(2) -> Jump left = 3.
               char_map={'w':5, ...}, len=3 ("kew")

Max length = 3 ("wke" or "kew")
```

---

### Solved Examples with Multiple Inputs

| Input `s` | Window Shifts & Jump Decisions | Longest Unique Substring | Max Length |
| :--- | :--- | :--- | :--- |
| `"abcabcbb"` | Jumps left past repeated 'a', 'b', 'c' | `"abc"` | `3` |
| `"bbbbb"` | 'b' repeated at every step; window length stays 1 | `"b"` | `1` |
| `"pwwkew"` | 'w' repeat triggers jump from 0 to 2; finds `"wke"` | `"wke"` | `3` |
| `""` | Empty string $\implies$ loop doesn't run | `""` | `0` |
| `" "` | Single space $\implies$ valid unique char | `" "` | `1` |
| `"abba"` | At index 3 ('a'), last seen 'a' is 0 < left(2), so left does NOT jump back! | `"ab"`, `"ba"` | `2` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_map = {}
        left = 0
        max_len = 0
        
        for right, ch in enumerate(s):
            if ch in char_map and char_map[ch] >= left:
                left = char_map[ch] + 1
            char_map[ch] = right
            max_len = max(max_len, right - left + 1)
            
        return max_len
```

#### 2. C++ (C++17 / STL)
```cpp
#include <string>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int lengthOfLongestSubstring(std::string s) {
        std::unordered_map<char, int> charMap;
        int left = 0, maxLen = 0;

        for (int right = 0; right < s.size(); ++right) {
            char ch = s[right];
            if (charMap.count(ch) && charMap[ch] >= left) {
                left = charMap[ch] + 1;
            }
            charMap[ch] = right;
            maxLen = std::max(maxLen, right - left + 1);
        }
        return maxLen;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> charMap = new HashMap<>();
        int left = 0, maxLen = 0;

        for (int right = 0; right < s.length(); right++) {
            char ch = s.charAt(right);
            if (charMap.containsKey(ch) && charMap.get(ch) >= left) {
                left = charMap.get(ch) + 1;
            }
            charMap.put(ch, right);
            maxLen = Math.max(maxLen, right - left + 1);
        }
        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — The `right` pointer iterates through the string once, and `left` only jumps forward.
- **Space Complexity:** $O(\min(N, M))$ where $M$ is alphabet size ($\le 128$ for standard ASCII).

---

### Takeaway Pattern & Interview Traps

1. **The Left-Pointer Regression Trap (`"abba"`):**
   - When encountering a duplicate character, you must ensure its previously recorded index is $\ge left$.
   - In `"abba"`, when reaching the second `'a'` at index 3, `char_map['a'] = 0`. But $left$ has already advanced to 2 (due to `'b'`). If you blindly set $left = char_map['a'] + 1$, $left$ would move backwards from 2 to 1, introducing duplicates into the window!
   - Always use `if char_map[ch] >= left: left = char_map[ch] + 1` or `left = max(left, char_map[ch] + 1)`.
2. **Fixed Alphabet Array vs Hash Map:**
   - For ASCII strings, an `int[128]` array initialized to `-1` is significantly faster than a generic `HashMap` due to zero cache-miss pointer chasing.

