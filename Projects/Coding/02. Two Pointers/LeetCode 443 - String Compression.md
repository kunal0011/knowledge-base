---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 443: String Compression"
tags:
  - leetcode
  - coding
  - two-pointers
  - in-place
  - amazon
  - google
---

# LeetCode 443: String Compression

**Target Companies:** Amazon (Top #1 In-Place String), Google, Meta  
**Difficulty:** Medium  
**Topic:** Fast / Slow Write Pointers In-Place

---

### Problem Statement

Given an array of characters `chars`, compress it using the following algorithm:

Begin with an empty string `s`. For each group of **consecutive repeating characters** in `chars`:
- If the group's length is `1`, append the character to `s`.
- Otherwise, append the character followed by the group's length.

The compressed string `s` **should not be returned separately**, but instead, be stored **in the input character array `chars`**. Note that group lengths that are 10 or longer will be split into multiple characters in `chars`.

Must use only **$O(1)$ extra space**. Return the new length of the array.

---

### Input & Output Formats & Constraints

- **Input:** `chars: List[str]`
- **Output:** `int` (new compressed length; modifies `chars` in-place)
- **Constraints:**
  - $1 \le \text{chars.length} \le 2000$

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def compress(self, chars: List[str]) -> int:
        write = 0
        read = 0
        n = len(chars)
        
        while read < n:
            ch = chars[read]
            count = 0
            while read < n and chars[read] == ch:
                read += 1
                count += 1
                
            chars[write] = ch
            write += 1
            
            if count > 1:
                for digit in str(count):
                    chars[write] = digit
                    write += 1
                    
        return write
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>

class Solution {
public:
    int compress(std::vector<char>& chars) {
        int write = 0, read = 0, n = chars.size();

        while (read < n) {
            char ch = chars[read];
            int count = 0;
            while (read < n && chars[read] == ch) {
                read++;
                count++;
            }

            chars[write++] = ch;
            if (count > 1) {
                for (char d : std::to_string(count)) {
                    chars[write++] = d;
                }
            }
        }
        return write;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int compress(char[] chars) {
        int write = 0, read = 0, n = chars.length;

        while (read < n) {
            char ch = chars[read];
            int count = 0;
            while (read < n && chars[read] == ch) {
                read++;
                count++;
            }

            chars[write++] = ch;
            if (count > 1) {
                for (char d : Integer.toString(count).toCharArray()) {
                    chars[write++] = d;
                }
            }
        }
        return write;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass over `chars`.
- **Space Complexity:** $O(1)$ strict in-place modification.
