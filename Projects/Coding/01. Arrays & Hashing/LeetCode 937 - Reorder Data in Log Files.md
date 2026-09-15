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
  - string
  - amazon
---

# LeetCode 937: Reorder Data in Log Files

**Target Companies:** Amazon (Top #1 Classic Signature Question), Microsoft, Google  
**Difficulty:** Medium  
**Topic:** Stable Custom Tuple Sort / String Partitioning

---

### Problem Statement

You are given an array of `logs`. Each log is a space-delimited string of words, where the first word is the **identifier**.

There are two types of logs:
- **Letter-logs:** All words (except the identifier) consist of lowercase English letters.
- **Digit-logs:** All words (except the identifier) consist of digits.

Reorder these logs so that:
1. The **letter-logs** come before all **digit-logs**.
2. The letter-logs are sorted **lexicographically** by their contents. If their contents are the same, sort them lexicographically by their identifiers.
3. The digit-logs maintain their **relative original order**.

Return *the final order of the logs*.

---

### Input & Output Formats & Constraints

- **Input:** `logs: List[str]`
- **Output:** `List[str]`
- **Constraints:**
  - $1 \le \text{logs.length} \le 100$
  - $3 \le \text{logs}[i].\text{length} \le 100$
  - `logs[i]` has an identifier and at least one word after the identifier.
  - All words in each log are separated by a single space.

---

### Key Idea & Intuition

The problem demands a customized sorting criterion with stability:

#### 1. Splitting Log into Identifier and Content:
Each log has the format `"<identifier> <content>"`. By splitting at the first space delimiter only (e.g. `log.split(" ", 1)`), we cleanly isolate:
- `ident`: The prefix before the first space.
- `content`: Everything following the first space.

#### 2. Classifying Log Type:
Because a log's words after the identifier are either all letters or all digits, checking `content[0].isdigit()` (or `content[0].isalpha()`) uniquely categorizes the log:
- If first character of content is a digit $\to$ **Digit-log**.
- If first character of content is a letter $\to$ **Letter-log**.

#### 3. Custom Sort Key / Stable Sort:
- We can define a sort key tuple:
  - For letter-logs: `(0, content, ident)`
    - Group identifier `0` ensures all letter-logs precede digit-logs.
    - `content` comes next for lexicographical ordering by content.
    - `ident` serves as tiebreaker if contents are identical.
  - For digit-logs: `(1,)`
    - Group identifier `1` places all digit-logs after letter-logs.
    - Because Python's Timsort and C++'s `std::stable_sort` / Java's `Arrays.sort` (for objects) are **stable**, items with identical keys preserve their relative input order!

---

### Solution Approach (Step-by-Step)

1. **Define Sorting Comparator or Key Function:**
   - For each `log`:
     - Split into `ident, content` at the first space (`split(" ", 1)`).
     - If `content[0]` is alphabetic:
       - Return key `(0, content, ident)`.
     - Else:
       - Return key `(1,)`.
2. **Execute Stable Sort:**
   - Sort the `logs` array using the custom key.
3. **Return Result:**
   - Return the sorted `logs`.

---

### Visual Algorithm Walkthrough

#### Example:
`logs = ["dig1 8 1 5 1", "let1 art can", "dig2 3 6", "let2 own kit dig", "let3 art zero"]`

```
Parse Keys:
Log 1: "dig1 8 1 5 1"   -> Content "8 1 5 1" starts with digit  -> Key: (1,)
Log 2: "let1 art can"   -> Content "art can" starts with letter -> Key: (0, "art can", "let1")
Log 3: "dig2 3 6"       -> Content "3 6" starts with digit      -> Key: (1,)
Log 4: "let2 own kit"   -> Content "own kit" starts with letter -> Key: (0, "own kit", "let2")
Log 5: "let3 art zero"  -> Content "art zero" starts with letter-> Key: (0, "art zero", "let3")

Sorting:
Priority 0 (Letter-logs):
  1. (0, "art can", "let1")   -> "let1 art can"
  2. (0, "art zero", "let3")  -> "let3 art zero"
  3. (0, "own kit", "let2")   -> "let2 own kit dig"

Priority 1 (Digit-logs, maintaining original relative order):
  4. "dig1 8 1 5 1"
  5. "dig2 3 6"

Final Output:
["let1 art can", "let3 art zero", "let2 own kit dig", "dig1 8 1 5 1", "dig2 3 6"]
```

---

### Solved Examples with Multiple Inputs

| Input `logs` | Parsed Letter-logs | Parsed Digit-logs | Output |
| :--- | :--- | :--- | :--- |
| `["a1 9 2 3 1","g1 act car","zo4 4 7","ab1 off key dog","a8 act zoo"]` | `"g1 act car"`, `"a8 act zoo"`, `"ab1 off key dog"` | `"a1 9 2 3 1"`, `"zo4 4 7"` | `["g1 act car","a8 act zoo","ab1 off key dog","a1 9 2 3 1","zo4 4 7"]` |
| `["let1 art can", "let2 art can"]` | Contents identical $\to$ tie-break by ID: `"let1"` before `"let2"` | None | `["let1 art can", "let2 art can"]` |
| `["dig1 1 2 3", "dig2 4 5 6"]` | None | `"dig1 1 2 3"`, `"dig2 4 5 6"` | `["dig1 1 2 3", "dig2 4 5 6"]` (Preserves order) |

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
                return (0, content, ident)  # Priority 0: letter-log (sort by content, then ident)
            return (1,)                     # Priority 1: digit-log (stable Timsort maintains relative order)
            
        return sorted(logs, key=get_key)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

class Solution {
public:
    std::vector<std::string> reorderLogFiles(std::vector<std::string>& logs) {
        auto comp = [](const std::string& a, const std::string& b) {
            size_t aSpace = a.find(' ');
            size_t bSpace = b.find(' ');

            bool aIsDigit = std::isdigit(a[aSpace + 1]);
            bool bIsDigit = std::isdigit(b[bSpace + 1]);

            if (!aIsDigit && !bIsDigit) {
                std::string aContent = a.substr(aSpace + 1);
                std::string bContent = b.substr(bSpace + 1);
                if (aContent != bContent) {
                    return aContent < bContent;
                }
                return a.substr(0, aSpace) < b.substr(0, bSpace);
            }

            // Letter logs come before digit logs
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

            // Both are letter-logs
            if (!aIsDigit && !bIsDigit) {
                int cmp = a.substring(aSpace + 1).compareTo(b.substring(bSpace + 1));
                if (cmp != 0) {
                    return cmp;
                }
                return a.substring(0, aSpace).compareTo(b.substring(0, bSpace));
            }

            // One letter-log and one digit-log
            if (!aIsDigit && bIsDigit) return -1;
            if (aIsDigit && !bIsDigit) return 1;

            // Both are digit-logs: return 0 to preserve relative order (Arrays.sort is stable)
            return 0;
        });

        return logs;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(M \cdot N \log N)$, where $N$ is the number of logs and $M$ is the maximum length of a single log. Comparing two strings takes $\mathcal{O}(M)$ time, and sorting $N$ elements takes $\mathcal{O}(N \log N)$ comparisons.
- **Space Complexity:** $\mathcal{O}(M \cdot N)$ auxiliary space required for key extraction and sorting recursion buffers.

---

### Takeaway Pattern & Interview Traps

1. **Stable Sorting Requirement:**
   - In C++, use `std::stable_sort` instead of `std::sort`. `std::sort` does not guarantee relative ordering preservation of digit logs.
   - In Java, `Arrays.sort(Object[])` is guaranteed to be stable (TimSort).
2. **Limit Split to 1:**
   - In Python, use `split(" ", 1)` so only the first space is used to partition the identifier from the rest of the string, preventing excess splitting of the content words.
