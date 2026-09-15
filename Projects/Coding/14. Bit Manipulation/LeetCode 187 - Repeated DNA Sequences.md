---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 187: Repeated DNA Sequences"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - hash-table
  - rolling-hash
  - google
  - amazon
---

# LeetCode 187: Repeated DNA Sequences

**Target Companies:** Google, Amazon, LinkedIn, Apple  
**Difficulty:** Medium  
**Topic:** 2-Bit Rolling Bitmask Hash (Bitwise Rabin-Karp)  

---

### Problem Statement

The **DNA sequence** is composed of a series of nucleotides abbreviated as `'A'`, `'C'`, `'G'`, and `'T'`.

For example, `"ACGAATTCCG"` is a DNA sequence.

When studying DNA, it is useful to identify repeated sequences within the DNA.

Given a string `s` that represents a **DNA sequence**, return all the **10-letter-long sequences** (substrings) that occur more than once in a DNA molecule. You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `List[str]` (all 10-letter repeated sequences)
- **Constraints:**
  - $1 \le \text{s.length} \le 10^5$
  - `s[i]` is either `'A'`, `'C'`, `'G'`, or `'T'`.

---

### Key Idea & Intuition

A naive approach extracts every 10-character substring and inserts it into a hash set. Each substring slice of length 10 requires string hashing and memory allocation, costing $O(10 \cdot N)$ time and high memory overhead for $N = 10^5$.

#### 2-Bit Integer Rolling Encoding:
Notice that the alphabet size $|\Sigma| = 4$:
- `'A'` $\to 00_2$ (0)
- `'C'` $\to 01_2$ (1)
- `'G'` $\to 10_2$ (2)
- `'T'` $\to 11_2$ (3)

A sequence of 10 nucleotides requires:
$$10 \times 2 \text{ bits} = 20 \text{ bits}$$
Since 20 bits easily fits within a standard 32-bit integer ($2^{20} - 1 = 1,048,575$), we can represent **any 10-letter DNA sequence as a single integer**!

#### Rolling Bitmask Window:
As the 10-character window slides right by 1 character:
1. Shift the existing 20-bit mask left by 2 positions: `mask << 2`.
2. Append the incoming nucleotide's 2-bit code: `| char_code`.
3. Discard the oldest nucleotide that slid out of the 20-bit window using a bitmask of 20 ones: `& ((1 << 20) - 1)` (or `& 0xFFFFF`).
4. Lookups and insertions into a hash set of integers take strict $O(1)$ time with zero string copies!

---

### Solution Approach (Step-by-Step)

1. If `len(s) < 10`, return `[]`.
2. Map characters to 2-bit values: `{'A': 0, 'C': 1, 'G': 2, 'T': 3}`.
3. Compute bitmask for the first 10 characters `s[0...9]`.
4. Maintain `seen` (set of seen integer bitmasks) and `res` (set of repeated string substrings).
5. For index $i$ from $10$ to $\text{len}(s) - 1$:
   - Roll bitmask: `curr_mask = ((curr_mask << 2) | mapping[s[i]]) & 0xFFFFF`.
   - If `curr_mask` is already in `seen`:
     - Add `s[i - 9 : i + 1]` to `res`.
   - Else:
     - Add `curr_mask` to `seen`.
6. Return `list(res)`.

---

### Visual Algorithm Walkthrough

```
Character Encoding:
  'A' = 00, 'C' = 01, 'G' = 10, 'T' = 11
20-bit Mask: 0xFFFFF (twenty 1s: 11111111111111111111_2)

Sequence: "AAAAACCCCCAAAAACCCCC"

Window 0: "AAAAACCCCC"
  Binary: 00 00 00 00 00  01 01 01 01 01
  Integer value = 341
  seen = {341}

Next char arrives: 'A' (00):
  1. Shift left 2: (341 << 2) = 00 00 00 00 01 01 01 01 01 00
  2. OR new char (00): ...0100
  3. Mask with 0xFFFFF: Keeps lower 20 bits.
  New Window: "AAAACCCCCA" -> Integer = 1364
  seen = {341, 1364}

When window reaches "AAAAACCCCC" again:
  Integer value calculated = 341.
  341 is already in seen!
  Add "AAAAACCCCC" to output set.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Repeating Sequences
- **Input:** `s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"`
- **Output:** `["AAAAACCCCC", "CCCCCAAAAA"]`

#### Example 2: Uniform Repeating Monomer
- **Input:** `s = "AAAAAAAAAAAAA"` (13 'A's)
- **Trace:**
  - Length 10 substring `"AAAAAAAAAA"` appears 4 times.
- **Output:** `["AAAAAAAAAA"]`

#### Example 3: String Length Less Than 10
- **Input:** `s = "ACGT"`
- **Output:** `[]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        n = len(s)
        if n < 10:
            return []
            
        to_2bit = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        mask_20bit = (1 << 20) - 1  # 0xFFFFF
        
        seen = set()
        res = set()
        
        # Build first 10-char integer bitmask
        curr_mask = 0
        for i in range(10):
            curr_mask = (curr_mask << 2) | to_2bit[s[i]]
        seen.add(curr_mask)
        
        # Slide window of size 10 across the remaining characters
        for i in range(10, n):
            curr_mask = ((curr_mask << 2) | to_2bit[s[i]]) & mask_20bit
            if curr_mask in seen:
                res.add(s[i - 9 : i + 1])
            else:
                seen.add(curr_mask)
                
        return list(res)
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <unordered_set>

class Solution {
public:
    std::vector<std::string> findRepeatedDnaSequences(std::string s) {
        int n = s.size();
        if (n < 10) return {};

        auto to2Bit = [](char c) -> int {
            switch (c) {
                case 'A': return 0;
                case 'C': return 1;
                case 'G': return 2;
                case 'T': return 3;
            }
            return 0;
        };

        const int mask20Bit = (1 << 20) - 1; // 0xFFFFF
        std::unordered_set<int> seen;
        std::unordered_set<std::string> res;

        int currMask = 0;
        for (int i = 0; i < 10; ++i) {
            currMask = (currMask << 2) | to2Bit(s[i]);
        }
        seen.insert(currMask);

        for (int i = 10; i < n; ++i) {
            currMask = ((currMask << 2) | to2Bit(s[i])) & mask20Bit;
            if (seen.count(currMask)) {
                res.insert(s.substr(i - 9, 10));
            } else {
                seen.insert(currMask);
            }
        }

        return std::vector<std::string>(res.begin(), res.end());
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class Solution {
    public List<String> findRepeatedDnaSequences(String s) {
        int n = s.length();
        if (n < 10) return new ArrayList<>();

        int[] to2Bit = new int[26];
        to2Bit['A' - 'A'] = 0;
        to2Bit['C' - 'A'] = 1;
        to2Bit['G' - 'A'] = 2;
        to2Bit['T' - 'A'] = 3;

        final int mask20Bit = (1 << 20) - 1; // 0xFFFFF
        Set<Integer> seen = new HashSet<>();
        Set<String> res = new HashSet<>();

        int currMask = 0;
        for (int i = 0; i < 10; i++) {
            currMask = (currMask << 2) | to2Bit[s.charAt(i) - 'A'];
        }
        seen.add(currMask);

        for (int i = 10; i < n; i++) {
            currMask = ((currMask << 2) | to2Bit[s.charAt(i) - 'A']) & mask20Bit;
            if (seen.contains(currMask)) {
                res.add(s.substring(i - 9, i + 1));
            } else {
                seen.add(currMask);
            }
        }

        return new ArrayList<>(res);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Rolling hash updates occur in $O(1)$ constant time bit operations. Adding matched substrings to the output takes $O(10)$ only when duplicates are encountered.
- **Space Complexity:** $O(N)$ — The hash set stores up to $N - 9$ integer bitmasks (4 bytes each) rather than strings (which would take $> 24$ bytes per entry in C++/Java), yielding a 6x memory reduction.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Fixed-Alphabet Rolling Bitmask Hash (Bitwise Rabin-Karp).
- **Trap:** Forgetting the `& 0xFFFFF` bitmask: without masking, the integer continually grows larger until overflowing 32-bit integer limits, corrupting the 10-character window.