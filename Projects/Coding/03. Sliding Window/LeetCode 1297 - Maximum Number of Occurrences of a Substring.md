---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1297: Maximum Number of Occurrences of a Substring"
tags:
  - leetcode
  - coding
  - sliding-window
  - string
  - hash-table
  - amazon
  - google
---

# LeetCode 1297: Maximum Number of Occurrences of a Substring

**Target Companies:** Google, Amazon, Microsoft  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Frequency Map / Length Optimization  

---

### Problem Statement

Given a string `s`, return the maximum number of occurrences of any substring under the following rules:
- The number of unique characters in the substring must be less than or equal to `maxLetters`.
- The substring size must be between `minSize` and `maxSize` inclusive.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`, `maxLetters: int`, `minSize: int`, `maxSize: int`
- **Output:** `int` (maximum frequency of any valid substring)
- **Constraints:**
  - $1 \le \text{s.length} \le 10^5$
  - $1 \le \text{maxLetters} \le 26$
  - $1 \le \text{minSize} \le \text{maxSize} \le \min(26, \text{s.length})$
  - `s` consists of lowercase English letters.

---

### Key Idea & Intuition

- **The `maxSize` Red-Herring (Crucial Mathematical Proof):**
  - The problem states that the substring length can be anywhere in $[\text{minSize}, \text{maxSize}]$.
  - **Claim:** We only ever need to check substrings of length **`minSize`**!
  - **Proof:**
    - Suppose a substring $T$ of length $L > \text{minSize}$ is valid (meaning $\text{unique\_chars}(T) \le \text{maxLetters}$) and occurs $F$ times in $s$.
    - Consider the prefix of $T$ of length $\text{minSize}$, denoted $P = T[0 \dots \text{minSize} - 1]$.
    - Since $P$ is a subsegment of $T$, $\text{unique\_chars}(P) \le \text{unique\_chars}(T) \le \text{maxLetters}$. Thus, $P$ is also a valid substring.
    - Every occurrence of $T$ in $s$ inevitably contains an occurrence of $P$.
    - Therefore, the frequency of $P$ must be $\ge$ the frequency of $T$:
      $$\text{count}(P) \ge \text{count}(T)$$
    - Thus, searching larger lengths can never yield a strictly higher frequency than searching at length `minSize`.
- **Fixed-Size Sliding Window of Length `minSize`:**
  - Maintain a sliding window of fixed length `minSize`.
  - Track the number of distinct characters in the window using an array or hash map of size 26.
  - If `distinct_count <= maxLetters`, increment the frequency count of the current substring in a hash map.
  - Return the maximum frequency observed.

---

### Solution Approach (Step-by-Step)

1. Initialize `counts = Counter()`, a character frequency map for the current window, and `substring_freq = defaultdict(int)`.
2. Initialize `distinct_chars = 0`.
3. Slide a window of length `minSize` across $s$:
   - Maintain character counts for the window $[i, i + \text{minSize} - 1]$.
   - When sliding from $i - 1$ to $i$:
     - Add incoming `s[i + minSize - 1]`.
     - Remove outgoing `s[i - 1]`.
   - If `len(window_char_map) <= maxLetters`:
     - Substring `sub = s[i : i + minSize]` is valid.
     - `substring_freq[sub] += 1`
4. Return `max(substring_freq.values(), default=0)`.

---

### Visual Algorithm Walkthrough

Let `s = "aababcaab"`, `maxLetters = 2`, `minSize = 3`, `maxSize = 4`.
We only need to evaluate windows of length `minSize = 3`:

```
Indices: 012345678
String:  aababcaab

1. i=0..2: "aab"
   - Distinct chars: {'a', 'b'} -> 2 <= maxLetters (2) -> VALID!
   - freq["aab"] = 1

2. i=1..3: "aba"
   - Distinct chars: {'a', 'b'} -> 2 <= 2 -> VALID!
   - freq["aba"] = 1

3. i=2..4: "bab"
   - Distinct chars: {'a', 'b'} -> 2 <= 2 -> VALID!
   - freq["bab"] = 1

4. i=3..5: "abc"
   - Distinct chars: {'a', 'b', 'c'} -> 3 > 2 -> INVALID (Pruned!)

5. i=4..6: "bca"
   - Distinct chars: {'a', 'b', 'c'} -> 3 > 2 -> INVALID (Pruned!)

6. i=5..7: "caa"
   - Distinct chars: {'a', 'c'} -> 2 <= 2 -> VALID!
   - freq["caa"] = 1

7. i=6..8: "aab"
   - Distinct chars: {'a', 'b'} -> 2 <= 2 -> VALID!
   - freq["aab"] = 2

Maximum frequency = 2 (for substring "aab").
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | `maxLetters` | `minSize` | `maxSize` | Valid Substrings Tested | Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `"aababcaab"` | `2` | `3` | `4` | `"aab"` occurs twice | `2` |
| **Example 2** | `"aaaa"` | `1` | `3` | `3` | `"aaa"` occurs twice | `2` |
| **No Valid Substring** | `"abcde"` | `2` | `3` | `3` | All 3-letter substrings have 3 distinct chars | `0` |
| **Single Character** | `"a"` | `1` | `1` | `1` | `"a"` occurs once | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import Dict
from collections import defaultdict

class Solution:
    def maxFreq(self, s: str, maxLetters: int, minSize: int, maxSize: int) -> int:
        """
        Finds the maximum occurrences of a valid substring.
        Only examines substrings of length minSize, since shorter substrings
        always have equal or higher frequency than their extensions.
        """
        n = len(s)
        substring_freq: Dict[str, int] = defaultdict(int)
        char_counts = [0] * 26
        distinct_count = 0

        # Prime the first window of length minSize
        for i in range(minSize):
            idx = ord(s[i]) - ord('a')
            if char_counts[idx] == 0:
                distinct_count += 1
            char_counts[idx] += 1

        if distinct_count <= maxLetters:
            substring_freq[s[:minSize]] += 1

        # Slide the window across s
        for i in range(minSize, n):
            # Add incoming character
            in_idx = ord(s[i]) - ord('a')
            if char_counts[in_idx] == 0:
                distinct_count += 1
            char_counts[in_idx] += 1

            # Remove outgoing character
            out_idx = ord(s[i - minSize]) - ord('a')
            char_counts[out_idx] -= 1
            if char_counts[out_idx] == 0:
                distinct_count -= 1

            # Check validity
            if distinct_count <= maxLetters:
                sub = s[i - minSize + 1 : i + 1]
                substring_freq[sub] += 1

        return max(substring_freq.values(), default=0)
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    int maxFreq(const std::string& s, int maxLetters, int minSize, int maxSize) {
        int n = static_cast<int>(s.size());
        std::unordered_map<std::string, int> substring_freq;
        std::vector<int> char_counts(26, 0);
        int distinct_count = 0;

        for (int i = 0; i < minSize; ++i) {
            int idx = s[i] - 'a';
            if (char_counts[idx]++ == 0) {
                distinct_count++;
            }
        }

        if (distinct_count <= maxLetters) {
            substring_freq[s.substr(0, minSize)]++;
        }

        for (int i = minSize; i < n; ++i) {
            int in_idx = s[i] - 'a';
            if (char_counts[in_idx]++ == 0) {
                distinct_count++;
            }

            int out_idx = s[i - minSize] - 'a';
            if (--char_counts[out_idx] == 0) {
                distinct_count--;
            }

            if (distinct_count <= maxLetters) {
                substring_freq[s.substr(i - minSize + 1, minSize)]++;
            }
        }

        int max_res = 0;
        for (const auto& [sub, count] : substring_freq) {
            max_res = std::max(max_res, count);
        }

        return max_res;
    }
};
```

#### Java
```java
import java.util.HashMap;
import java.util.Map;

class Solution {
    public int maxFreq(String s, int maxLetters, int minSize, int maxSize) {
        int n = s.length();
        Map<String, Integer> substringFreq = new HashMap<>();
        int[] charCounts = new int[26];
        int distinctCount = 0;

        for (int i = 0; i < minSize; i++) {
            int idx = s.charAt(i) - 'a';
            if (charCounts[idx]++ == 0) {
                distinctCount++;
            }
        }

        if (distinctCount <= maxLetters) {
            substringFreq.put(s.substring(0, minSize), 1);
        }

        for (int i = minSize; i < n; i++) {
            int inIdx = s.charAt(i) - 'a';
            if (charCounts[inIdx]++ == 0) {
                distinctCount++;
            }

            int outIdx = s.charAt(i - minSize) - 'a';
            if (--charCounts[outIdx] == 0) {
                distinctCount--;
            }

            if (distinctCount <= maxLetters) {
                String sub = s.substring(i - minSize + 1, i + 1);
                substringFreq.put(sub, substringFreq.getOrDefault(sub, 0) + 1);
            }
        }

        int maxCount = 0;
        for (int count : substringFreq.values()) {
            if (count > maxCount) {
                maxCount = count;
            }
        }

        return maxCount;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N \cdot \text{minSize})$ where $N = \text{s.length} \le 10^5$ and $\text{minSize} \le 26$.
  - Sliding the window takes $\mathcal{O}(N)$ updates of distinct character counts.
  - Taking a substring slice of length $\text{minSize}$ and inserting into the hash map takes $\mathcal{O}(\text{minSize})$.
  - Total operations $\le 26 \times 10^5 \approx 2.6 \times 10^6$, running in $< 25 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N \cdot \text{minSize})$ in the worst case to store the substrings in the frequency hash map.

---

### Takeaway Pattern & Interview Traps

- **Detecting Redundant Parameters:** Recognizing that `maxSize` is completely irrelevant is the core test of this interview problem. Trying to check all window lengths from `minSize` to `maxSize` leads to unnecessary complexity and slower runtime.
- **Substring Slice Minimization:** Only extract `s[i - minSize + 1 : i + 1]` when `distinct_count <= maxLetters`. This avoids creating substring objects for invalid windows.