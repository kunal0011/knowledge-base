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
  - google
  - amazon
---

# LeetCode 68: Text Justification

**Target Companies:** Google (Signature Hard), Amazon, Meta, Apple  
**Difficulty:** Hard  
**Topic:** Greedy Line Packing & Space Distribution

---

### Problem Statement

Given an array of strings `words` and a width `maxWidth`, format the text such that each line has exactly `maxWidth` characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces `' '` when necessary so that each line has exactly `maxWidth` characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the **empty slots on the left will be assigned more spaces** than the slots on the right.

For the last line of text, it should be **left-justified**, and no extra space is inserted between words.

---

### Input & Output Formats & Constraints

- **Input:** `words: List[str]`, `maxWidth: int`
- **Output:** `List[str]`
- **Constraints:**
  - $1 \le \text{words.length} \le 300$
  - $1 \le \text{words}[i].\text{length} \le \text{maxWidth} \le 100$
  - `words[i]` consists of English letters and symbols.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res, cur_line, num_letters = [], [], 0
        
        for w in words:
            if num_letters + len(w) + len(cur_line) > maxWidth:
                for i in range(maxWidth - num_letters):
                    cur_line[i % (len(cur_line) - 1 or 1)] += ' '
                res.append("".join(cur_line))
                cur_line, num_letters = [], 0
            cur_line.append(w)
            num_letters += len(w)
            
        res.append(" ".join(cur_line).ljust(maxWidth))
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
            while (j < n && lineLen + words[j].size() + (j - i) <= maxWidth) {
                lineLen += words[j].size();
                j++;
            }

            int numWords = j - i;
            int numSpaces = maxWidth - lineLen;
            std::string line = "";

            if (numWords == 1 || j == n) {
                for (int k = i; k < j; ++k) {
                    line += words[k];
                    if (k < j - 1) line += " ";
                }
                line.append(maxWidth - line.size(), ' ');
            } else {
                int baseSpaces = numSpaces / (numWords - 1);
                int extraSpaces = numSpaces % (numWords - 1);

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
                for (int k = i; k < j; k++) {
                    line.append(words[k]);
                    if (k < j - 1) line.append(" ");
                }
                while (line.length() < maxWidth) line.append(" ");
            } else {
                int baseSpaces = numSpaces / (numWords - 1);
                int extraSpaces = numSpaces % (numWords - 1);

                for (int k = i; k < j - 1; k++) {
                    line.append(words[k]);
                    int spaces = baseSpaces + (k - i < extraSpaces ? 1 : 0);
                    for (int s = 0; s < spaces; s++) line.append(" ");
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

- **Time Complexity:** $O(N \times \text{maxWidth})$ — Linear sweep across all words.
- **Space Complexity:** $O(N \times \text{maxWidth})$ for the formatted lines.
