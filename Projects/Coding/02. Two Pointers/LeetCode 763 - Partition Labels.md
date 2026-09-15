---
date: "2026-09-15"
type: leetcode-solution
category: "Two Pointers"
folder: "02. Two Pointers"
title: "LeetCode 763: Partition Labels"
tags:
  - leetcode
  - coding
  - two-pointers
  - greedy
  - string
  - amazon
  - google
  - meta
---

# LeetCode 763: Partition Labels

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy Two Pointers / Last Occurrence Sweep

---

### Problem Statement

You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be `s`.

Return *a list of integers representing the size of these parts*.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `List[int]` (sizes of partitions)
- **Constraints:**
  - $1 \le \text{s.length} \le 500$
  - `s` consists of lowercase English letters.

---

### Key Idea & Intuition

#### 1. Invariant:
If a partition contains a character $c$, it **must** extend at least until the **last occurrence** of $c$ anywhere in the string. Otherwise, character $c$ would appear in another partition, violating the problem constraint.

#### 2. Greedy Right Boundary Expansion:
1. First pass: Record the last index of each character `'a'` to `'z'` in a lookup array or hash map `last`.
2. Second pass: Maintain the current partition boundary `[start, end]`:
   - Initialize `start = 0`, `end = 0`.
   - As we scan each index $i$:
     - The partition must accommodate the last occurrence of $s[i]$, so update:
       $$\text{end} = \max(\text{end}, \text{last}[s[i]])$$
     - If $i == \text{end}$:
       - Every character visited so far has its last occurrence within $[start, end]$!
       - No character inside this window appears anywhere to the right of $end$.
       - Because we want to maximize the number of partitions (making each partition as small as possible), we greedily cut here:
         $$\text{size} = \text{end} - \text{start} + 1$$
       - Set `start = i + 1` for the next partition.

---

### Solution Approach (Step-by-Step)

1. **Record Last Occurrences:**
   - Create a 26-element array `last` where `last[c - 'a'] = i` for every character in $s$.
2. **Greedy Scan:**
   - Initialize `res = []`, `start = 0`, `end = 0`.
   - For $i$ from $0$ to $len(s) - 1$:
     - `end = max(end, last[s[i] - 'a'])`
     - If $i == end$:
       - Append `end - start + 1` to `res`.
       - `start = i + 1`
3. **Return Partition Sizes:**
   - Return `res`.

---

### Visual Algorithm Walkthrough

#### Example: `s = "ababcbacadefegdehijhklij"` ($N = 24$)

```
Last occurrences:
  'a': 8, 'b': 5, 'c': 7, 'd': 14, 'e': 15, 'f': 11, 'g': 13, 'h': 19, 'i': 22, 'j': 23, 'k': 20, 'l': 21

Scan Phase:
i=0, ch='a': end = max(0, 8) = 8
i=1, ch='b': end = max(8, 5) = 8
i=2, ch='a': end = 8
i=3, ch='b': end = 8
i=4, ch='c': end = max(8, 7) = 8
i=5, ch='b': end = 8
i=6, ch='a': end = 8
i=7, ch='c': end = 8
i=8, ch='a': end = 8.
  Here i == end (8 == 8)!
  Partition 1 found: "ababcbaca", size = 8 - 0 + 1 = 9.
  res.append(9). start = 9.

i=9,  ch='d': end = max(0, 14) = 14
i=10, ch='e': end = max(14, 15) = 15
i=11, ch='f': end = max(15, 11) = 15
...
i=15, ch='e': end = 15.
  Here i == end (15 == 15)!
  Partition 2 found: "defegde", size = 15 - 9 + 1 = 7.
  res.append(7). start = 16.

i=16..23: "hijhklij"
  end expands to 23.
  At i=23: i == end.
  Partition 3 found: "hijhklij", size = 23 - 16 + 1 = 8.
  res.append(8).

Final Output: [9, 7, 8]
```

---

### Solved Examples with Multiple Inputs

| Input `s` | Partitions Formed | Substring Segments | Output Sizes |
| :--- | :--- | :--- | :--- |
| `"ababcbacadefegdehijhklij"` | 3 | `"ababcbaca"`, `"defegde"`, `"hijhklij"` | `[9, 7, 8]` |
| `"eccbbbbdec"` | 1 | Entire string contains `'e'` and `'c'` spanning across | `[10]` |
| `"a"` | 1 | `"a"` | `[1]` |
| `"abc"` | 3 | `"a"`, `"b"`, `"c"` (all distinct) | `[1, 1, 1]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {ch: i for i, ch in enumerate(s)}
        res: List[int] = []
        start, end = 0, 0
        
        for i, ch in enumerate(s):
            end = max(end, last[ch])
            if i == end:
                res.append(end - start + 1)
                start = i + 1
                
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::vector<int> partitionLabels(std::string s) {
        int last[26] = {0};
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            last[s[i] - 'a'] = i;
        }

        std::vector<int> result;
        int start = 0, end = 0;
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            end = std::max(end, last[s[i] - 'a']);
            if (i == end) {
                result.push_back(end - start + 1);
                start = i + 1;
            }
        }
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<Integer> partitionLabels(String s) {
        int[] last = new int[26];
        for (int i = 0; i < s.length(); i++) {
            last[s.charAt(i) - 'a'] = i;
        }

        List<Integer> result = new ArrayList<>();
        int start = 0, end = 0;

        for (int i = 0; i < s.length(); i++) {
            end = Math.max(end, last[s.charAt(i) - 'a']);
            if (i == end) {
                result.add(end - start + 1);
                start = i + 1;
            }
        }
        return result;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of string $s$. The first pass populates the last occurrence array in $\mathcal{O}(n)$ time. The second pass evaluates the partitions in a single scan of $\mathcal{O}(n)$ time.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. The `last` array has a constant size of 26 integers for lowercase English letters.

---

### Takeaway Pattern & Interview Traps

1. **Greedy Maximization of Partitions:**
   - By cutting at the earliest possible index where `i == end`, we mathematically maximize the number of partitions. Any later cut would only merge valid partitions together, decreasing the total partition count.
2. **Interval Merging Analogy:**
   - This problem is equivalent to Merge Intervals: each distinct character defines an interval `[first_seen[c], last_seen[c]]`. Merging all overlapping intervals and returning their lengths yields the identical result.
