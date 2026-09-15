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
  - string
  - amazon
  - google
---

# LeetCode 443: String Compression

**Target Companies:** Amazon (Top #1 In-Place String), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Fast Read / Slow Write Pointers In-Place

---

### Problem Statement

Given an array of characters `chars`, compress it using the following algorithm:

Begin with an empty string `s`. For each group of **consecutive repeating characters** in `chars`:
- If the group's length is `1`, append the character to `s`.
- Otherwise, append the character followed by the group's length.

The compressed string `s` **should not be returned separately**, but instead, be stored **in the input character array `chars`**. Note that group lengths that are 10 or longer will be split into multiple characters in `chars`.

Must use only **$\mathcal{O}(1)$ extra space**. Return the new length of the array.

---

### Input & Output Formats & Constraints

- **Input:** `chars: List[str]`
- **Output:** `int` (new compressed length; modifies `chars` in-place)
- **Constraints:**
  - $1 \le \text{chars.length} \le 2000$
  - `chars[i]` is a lowercase English letter, uppercase English letter, digit, or symbol.

---

### Key Idea & Intuition

#### Two-Pointer Read / Write Invariant:
Because compressed groups are **always less than or equal to** the length of the original repeating sequence (e.g., `"a"` $\to 1$ char, `"aa"` $\to 2$ chars `"a2"`, `"aaa"` $\to 2$ chars `"a3"`, `"aaaaaaaaaaaa"` $\to 3$ chars `"a12"`), the write head will **never overtake** the read head:
$$\text{write} \le \text{read}$$
This mathematical guarantee ensures that overwriting `chars[write]` will never destroy unread future data.

#### Algorithm Steps:
1. Maintain `write = 0` (where to write the next compressed character) and `read = 0` (scanning the input).
2. While `read < n`:
   - Identify the current character `ch = chars[read]`.
   - Count how many consecutive times `ch` repeats by advancing `read`.
   - Write `ch` at `chars[write++]`.
   - If `count > 1`, convert `count` to its string/character representation (e.g. `12` $\to$ `'1'`, `'2'`) and write each digit into `chars[write++]`.
3. Return `write`.

---

### Solution Approach (Step-by-Step)

1. **Initialize Pointers:**
   - `write = 0`, `read = 0`, `n = len(chars)`.
2. **Process Run-Length Encodings:**
   - While `read < n`:
     - Set `ch = chars[read]` and initialize `count = 0`.
     - While `read < n` and `chars[read] == ch`:
       - `read += 1`
       - `count += 1`
     - Overwrite: `chars[write] = ch`, `write += 1`.
     - If `count > 1`:
       - For each character `digit` in `str(count)`:
         - `chars[write] = digit`
         - `write += 1`
3. **Return Length:**
   - Return `write`.

---

### Visual Algorithm Walkthrough

#### Example: `chars = ["a","a","b","b","c","c","c"]` ($N = 7$)

```
read=0: ch='a'.
  chars[0]=='a', chars[1]=='a' -> count = 2, read advances to 2.
  chars[write=0] = 'a' (write=1)
  count=2 > 1 -> chars[write=1] = '2' (write=2)
  Array: ['a', '2', 'b', 'b', 'c', 'c', 'c']

read=2: ch='b'.
  chars[2]=='b', chars[3]=='b' -> count = 2, read advances to 4.
  chars[write=2] = 'b' (write=3)
  count=2 > 1 -> chars[write=3] = '2' (write=4)
  Array: ['a', '2', 'b', '2', 'c', 'c', 'c']

read=4: ch='c'.
  chars[4]=='c', chars[5]=='c', chars[6]=='c' -> count = 3, read advances to 7.
  chars[write=4] = 'c' (write=5)
  count=3 > 1 -> chars[write=5] = '3' (write=6)
  Array: ['a', '2', 'b', '2', 'c', '3', 'c']

read == 7 == n. Finished!
Return write = 6. Compressed prefix: ["a","2","b","2","c","3"].
```

---

### Solved Examples with Multiple Inputs

| Input `chars` | Group Breakdown | In-Place Modification | Return Value |
| :--- | :--- | :--- | :--- |
| `["a","a","b","b","c","c","c"]` | `a:2, b:2, c:3` | `["a","2","b","2","c","3"]` | `6` |
| `["a"]` | `a:1` | `["a"]` | `1` (No count appended) |
| `["a","b","1","2"]` | All groups size 1 | `["a","b","1","2"]` | `4` |
| `["a"] * 12` | `a:12` | `["a","1","2"]` | `3` (Split digits '1' and '2') |

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
        int write = 0;
        int read = 0;
        int n = chars.size();

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
        int write = 0;
        int read = 0;
        int n = chars.length;

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

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of `chars`. Each character is read exactly once and written at most twice (character + digit characters).
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. The modification is performed directly in-place within the provided array.

---

### Takeaway Pattern & Interview Traps

1. **Multi-Digit Group Count Expansion:**
   - When count is $\ge 10$, each digit must be written as an individual character (e.g., $12 \to '1', '2'$). Failing to iterate through each digit of the number string is a common bug.
2. **Single Occurrence Rule:**
   - When count is $1$, no number is appended. Writing `"a1"` for `"a"` will fail test cases.
3. **Guarantee of Non-Overtaking:**
   - Why is `chars[write]` safe to overwrite? For any run length $L \ge 1$:
     - $L = 1$: uses 1 slot (original took 1 slot).
     - $L \in [2, 9]$: uses 2 slots (original took $\ge 2$ slots).
     - $L \ge 10$: uses $1 + \lfloor \log_{10} L \rfloor + 1 \le L$ slots.
     Thus `write <= read` holds at every step.
