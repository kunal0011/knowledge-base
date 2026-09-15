---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 68: Text Justification"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - string-formatting
  - greedy
  - google
  - amazon
  - meta
---

# LeetCode 68: Text Justification

**Target Companies:** Google (Signature Hard), Amazon, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Hard  
**Topic:** Greedy Line Packing & Space Distribution

---

### Problem Statement

Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly `maxWidth` characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the **empty slots on the left will be assigned more spaces** than the slots on the right.

For the last line of text, it should be **left-justified**, and no extra space is inserted between words.

**Note:**
- A word is defined as a character sequence consisting of non-space characters only.
- Each word's length is guaranteed to be greater than 0 and not exceed `maxWidth`.
- The input array `words` contains at least one word.

---

### Input & Output Formats & Constraints

- **Input:** `words: List[str]`, `maxWidth: int`
- **Output:** `List[str]`
- **Constraints:**
  - $1 \le \text{words.length} \le 300$
  - $1 \le \text{words}[i].\text{length} \le \text{maxWidth} \le 100$
  - `words[i]` consists of English letters and symbols.

---

### Key Idea & Intuition

The problem is a pure implementation and greedy layout challenge with three distinct formatting modes:

1. **Greedy Word Packing:**
   - How many words fit on the current line?
   - A line with words $w_i, w_{i+1}, \dots, w_{j-1}$ requires at least $\sum_{k=i}^{j-1} |w_k| + (j - 1 - i)$ characters (since each adjacent pair requires at least 1 separating space).
   - Find the maximal $j$ such that this sum is $\le maxWidth$.

2. **Case A: Normal Fully Justified Line ($j < n$ and $num\_words > 1$):**
   - Total spaces needed = $maxWidth - \sum |w_k|$.
   - Number of space slots between words = $num\_words - 1$.
   - Each slot receives a base allocation: $base = \lfloor total\_spaces / slots \rfloor$.
   - The leftmost $rem = total\_spaces \pmod{slots}$ slots receive $base + 1$ spaces.

3. **Case B: Left-Justified Line (Single Word or Last Line):**
   - If a line has only 1 word ($num\_words == 1$), or if it is the final line ($j == n$):
   - Separate adjacent words with exactly **one** single space `' '`.
   - Pad the remaining spaces at the very end of the line on the right until the total length reaches $maxWidth$.

---

### Solution Approach (Step-by-Step)

1. **Outer Loop over Words:**
   - Maintain an index $i = 0$. While $i < n$:
2. **Determine Line Boundary $j$:**
   - Start $j = i$, `line_len = 0`.
   - Advance $j$ while $j < n$ and `line_len + len(words[j]) + (j - i) <= maxWidth`.
   - Add `len(words[j])` to `line_len`.
3. **Format the Line:**
   - Number of words on this line is $num\_words = j - i$.
   - If $num\_words == 1$ or $j == n$ (last line):
     - Join `words[i..j-1]` with a single space `" "`.
     - Pad right with spaces until length equals $maxWidth$.
   - Else (middle line with $\ge 2$ words):
     - Compute $spaces = maxWidth - line\_len$.
     - Compute $slots = num\_words - 1$.
     - $base = spaces // slots$, $extra = spaces \% slots$.
     - For each word from $i$ to $j-2$, append the word followed by $base + (1 \text{ if } (k - i) < extra \text{ else } 0)$ spaces.
     - Append the final word $words[j-1]$.
4. **Append Formatted Line and Update $i = j$.**
5. **Return Result List.**

---

### Visual Algorithm Walkthrough

#### Example: `words = ["This", "is", "an", "example", "of", "text", "justification."]`, `maxWidth = 16`

```
Line 1:
Words that fit: ["This", "is", "an"]
Letter count: 4 + 2 + 2 = 8
Total spaces needed: 16 - 8 = 8 spaces
Slots: 2 slots
base = 8 // 2 = 4 spaces, extra = 8 % 2 = 0
Slot 1: 4 spaces -> "This    "
Slot 2: 4 spaces -> "is    "
Last word: "an"
Result line: "This    is    an" (Length 16)

Line 2:
Words that fit: ["example", "of", "text"]
Letter count: 7 + 2 + 4 = 13
Total spaces needed: 16 - 13 = 3 spaces
Slots: 2 slots
base = 3 // 2 = 1 space, extra = 3 % 2 = 1
Slot 1 (extra + 1): 2 spaces -> "example  "
Slot 2: 1 space -> "of "
Last word: "text"
Result line: "example  of text" (Length 16)

Line 3 (Last line):
Words: ["justification."]
Letter count: 14
Single word on last line -> left justify, right pad:
"justification.  " (Length 16)
```

---

### Solved Examples with Multiple Inputs

| `words` | `maxWidth` | Formatted Output | Notes |
| :--- | :--- | :--- | :--- |
| `["This", "is", "an", "example", "of", "text", "justification."]` | `16` | `["This    is    an", "example  of text", "justification.  "]` | Multi-slot, uneven distribution, and single-word last line |
| `["What","must","be","acknowledgment","shall","be"]` | `16` | `["What   must   be", "acknowledgment  ", "shall be        "]` | Word occupying almost entire line + multi-word last line |
| `["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"]` | `20` | Detailed 6-line justified block | Tests even distribution across many words |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res: List[str] = []
        i, n = 0, len(words)
        
        while i < n:
            # Find how many words fit on this line
            j = i
            line_len = 0
            while j < n and line_len + len(words[j]) + (j - i) <= maxWidth:
                line_len += len(words[j])
                j += 1
                
            num_words = j - i
            total_spaces = maxWidth - line_len
            
            # Case 1: Single word or last line (left-justified)
            if num_words == 1 or j == n:
                line = " ".join(words[i:j])
                line = line + " " * (maxWidth - len(line))
            else:
                # Case 2: Fully justified line
                slots = num_words - 1
                base_spaces = total_spaces // slots
                extra_spaces = total_spaces % slots
                
                parts = []
                for k in range(i, j - 1):
                    parts.append(words[k])
                    # Add base spaces + 1 if within extra_spaces
                    spaces_to_add = base_spaces + (1 if (k - i) < extra_spaces else 0)
                    parts.append(" " * spaces_to_add)
                parts.append(words[j - 1])
                line = "".join(parts)
                
            res.append(line)
            i = j
            
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>

class Solution {
public:
    std::vector<std::string> fullJustify(std::vector<std::string>& words, int maxWidth) {
        std::vector<std::string> result;
        int i = 0, n = words.size();

        while (i < n) {
            int j = i, lineLen = 0;
            while (j < n && lineLen + static_cast<int>(words[j].size()) + (j - i) <= maxWidth) {
                lineLen += words[j].size();
                j++;
            }

            int numWords = j - i;
            int numSpaces = maxWidth - lineLen;
            std::string line = "";

            if (numWords == 1 || j == n) {
                // Left-justified: single space between words, pad remaining on right
                for (int k = i; k < j; ++k) {
                    line += words[k];
                    if (k < j - 1) line += " ";
                }
                line.append(maxWidth - line.size(), ' ');
            } else {
                // Fully justified
                int slots = numWords - 1;
                int baseSpaces = numSpaces / slots;
                int extraSpaces = numSpaces % slots;

                for (int k = i; k < j - 1; ++k) {
                    line += words[k];
                    line.append(baseSpaces + (k - i < extraSpaces ? 1 : 0), ' ');
                }
                line += words[j - 1];
            }

            result.push_back(line);
            i = j;
        }

        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> result = new ArrayList<>();
        int i = 0, n = words.length;

        while (i < n) {
            int j = i, lineLen = 0;
            while (j < n && lineLen + words[j].length() + (j - i) <= maxWidth) {
                lineLen += words[j].length();
                j++;
            }

            int numWords = j - i;
            int numSpaces = maxWidth - lineLen;
            StringBuilder line = new StringBuilder();

            if (numWords == 1 || j == n) {
                // Left-justified
                for (int k = i; k < j; k++) {
                    line.append(words[k]);
                    if (k < j - 1) line.append(" ");
                }
                while (line.length() < maxWidth) {
                    line.append(" ");
                }
            } else {
                // Fully justified
                int slots = numWords - 1;
                int baseSpaces = numSpaces / slots;
                int extraSpaces = numSpaces % slots;

                for (int k = i; k < j - 1; k++) {
                    line.append(words[k]);
                    int spaces = baseSpaces + (k - i < extraSpaces ? 1 : 0);
                    for (int s = 0; s < spaces; s++) {
                        line.append(" ");
                    }
                }
                line.append(words[j - 1]);
            }

            result.add(line.toString());
            i = j;
        }

        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(L)$ where $L$ is the total number of characters across all words and padded spaces. Each word is scanned once to calculate length, and each character is written once to the output buffer.
- **Space Complexity:** $\mathcal{O}(L)$ auxiliary space to construct and return the formatted strings.

---

### Takeaway Pattern & Interview Traps

1. **Division by Zero Protection:**
   - If a line contains only 1 word, `slots = num_words - 1 = 0`. Attempting `total_spaces / slots` triggers a runtime divide-by-zero error! Always check `num_words == 1` first.
2. **Last Line Rule:**
   - Even if the last line contains multiple words, it must be left-justified with single spaces between words and all remaining spaces placed at the end.
3. **Remainder Space Distribution:**
   - Remainder spaces must be allocated to the leftmost slots first: `k - i < extra_spaces`.
