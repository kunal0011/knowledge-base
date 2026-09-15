---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 937: Reorder Data in Log Files"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - custom-sort
  - amazon
---

# LeetCode 937: Reorder Data in Log Files

**Target Companies:** Amazon (Top #1 Classic Signature Question)  
**Difficulty:** Medium  
**Topic:** Stable Custom Tuple Sort

---

### Problem Statement

You are given an array of `logs`. Each log is a space-delimited string of words, where the first word is the **identifier**.

There are two types of logs:
- **Letter-logs:** All words (except the identifier) consist of lowercase English letters.
- **Digit-logs:** All words (except the identifier) consist of digits.

Reorder these logs so that:
1. The **letter-logs** come before all **digit-logs**.
2. The letter-logs are sorted **lexicographically** by their contents. If their contents are the same, sort them lexicographically by their identifiers.
3. The digit-logs maintain their **relative order**.

Return the final order of the logs.

---

### Input & Output Formats & Constraints

- **Input:** `logs: List[str]`
- **Output:** `List[str]`
- **Constraints:**
  - $1 \le \text{logs.length} \le 100$
  - $3 \le \text{logs}[i].\text{length} \le 100$

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        def get_key(log: str):
            ident, content = log.split(" ", 1)
            if content[0].isalpha():
                return (0, content, ident)  # Priority 0: letter-log
            return (1,)                     # Priority 1: digit-log (stable Timsort)
            
        return sorted(logs, key=get_key)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::vector<std::string> reorderLogFiles(std::vector<std::string>& logs) {
        auto comp = [](const std::string& a, const std::string& b) {
            int aSpace = a.find(' ');
            int bSpace = b.find(' ');

            bool aIsDigit = isdigit(a[aSpace + 1]);
            bool bIsDigit = isdigit(b[bSpace + 1]);

            if (!aIsDigit && !bIsDigit) {
                std::string aContent = a.substr(aSpace + 1);
                std::string bContent = b.substr(bSpace + 1);
                if (aContent != bContent) return aContent < bContent;
                return a.substr(0, aSpace) < b.substr(0, bSpace);
            }
            if (!aIsDigit && bIsDigit) return true;
            return false;
        };

        std::stable_sort(logs.begin(), logs.end(), comp);
        return logs;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.Arrays;

class Solution {
    public String[] reorderLogFiles(String[] logs) {
        Arrays.sort(logs, (a, b) -> {
            int aSpace = a.indexOf(' ');
            int bSpace = b.indexOf(' ');

            char aChar = a.charAt(aSpace + 1);
            char bChar = b.charAt(bSpace + 1);

            boolean aIsDigit = Character.isDigit(aChar);
            boolean bIsDigit = Character.isDigit(bChar);

            if (!aIsDigit && !bIsDigit) {
                int cmp = a.substring(aSpace + 1).compareTo(b.substring(bSpace + 1));
                if (cmp != 0) return cmp;
                return a.substring(0, aSpace).compareTo(b.substring(0, bSpace));
            }

            if (!aIsDigit && bIsDigit) return -1;
            if (aIsDigit && !bIsDigit) return 1;
            return 0; // maintain stable order for digit logs
        });

        return logs;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(M \times N \log N)$ where $M$ is max log length and $N$ is number of logs.
- **Space Complexity:** $O(M \times N)$ for sorting keys / recursion buffer.
