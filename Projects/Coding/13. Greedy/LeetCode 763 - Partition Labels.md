---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 763: Partition Labels"
tags:
  - leetcode
  - coding
  - greedy
  - two-pointers
  - string
  - amazon
  - google
  - meta
  - microsoft
---

# LeetCode 763: Partition Labels

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Uber  
**Difficulty:** Medium  
**Topic:** Greedy / Two Pointers / String  

---

### Problem Statement

You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be `s`.

Return a list of integers representing the size of these parts.

---

### Input & Output Formats & Constraints

- **Input:** A string `s` of length $1 \le |s| \le 500$.
- **Output:** A list of integers `List[int]` representing the lengths of each partition in order.
- **Constraints:**
  - `1 <= s.length <= 500`
  - `s` consists of lowercase English letters.

---

### Key Idea & Intuition

#### Interval Expansion & Invariant
For each unique character $c$ present in `s`, all occurrences of $c$ must be confined to the exact same partition. Thus, if a partition includes the first occurrence of $c$, it must extend **at least** to the last occurrence of $c$.

Let $\text{last}[c]$ be the highest index where character $c$ appears in $s$.
As we iterate through $s$ from left to right:
1. Every time we encounter character $s[i]$, the right boundary of the current partition must be at least $\text{last}[s[i]]$.
2. We maintain a running boundary:
   $$\text{end} = \max(\text{end}, \text{last}[s[i]])$$
3. If our current index $i$ reaches $\text{end}$ ($i == \text{end}$), then every character encountered in the current segment $[ \text{start}, \text{end} ]$ has all of its occurrences contained completely within this range. No character within this window appears beyond index $i$.
4. Therefore, index $\text{end}$ represents the earliest valid partition cut point! By cutting here, we maximize the number of partitions.

#### Why Greedy is Globally Optimal
A partition cannot terminate before $\text{end}$. Terminating at the exact moment $i == \text{end}$ creates a valid partition using the minimum necessary length, leaving the largest remaining suffix for subsequent partitions. This choice satisfies the greedy-choice property and optimal substructure.

---

### Solution Approach (Step-by-Step)

1. **Precompute Last Occurrences:**
   - Create an array `last` of size 26 (for `'a'` through `'z'`).
   - Iterate $i$ from $0$ to $|s|-1$, setting $\text{last}[s[i] - \text{'a'}] = i$.
2. **Greedy Traversal:**
   - Initialize `start = 0`, `end = 0`, and an empty list `result`.
   - Iterate $i$ from $0$ to $|s|-1$:
     - Update $\text{end} = \max(\text{end}, \text{last}[s[i] - \text{'a'}])$.
     - If $i == \text{end}$:
       - Append the partition length $\text{end} - \text{start} + 1$ to `result`.
       - Update $\text{start} = i + 1$.
3. **Return:**
   - Return `result`.

---

### Visual Algorithm Walkthrough

#### Trace of $s = \text{"ababcbacadefegdehijhklij"}$
```
Precomputed last occurrence for each character:
a: 8,  b: 5,  c: 7,  d: 14, e: 15, f: 11, g: 13,
h: 19, i: 22, j: 23, k: 20, l: 21

Index:    0 1 2 3 4 5 6 7 8 | 9 10 11 12 13 14 15 | 16 17 18 19 20 21 22 23
Char:     a b a b c b a c a | d  e  f  e  g  d  e |  h  i  j  h  k  l  i  j
                      
Partition 1:
i = 0, char 'a': end = max(0, 8) = 8
i = 1, char 'b': end = max(8, 5) = 8
...
i = 8, char 'a': end = 8. Since i == end (8 == 8):
  -> Cut Partition 1!
  -> Size: 8 - 0 + 1 = 9
  -> New start = 9

Partition 2:
i = 9,  char 'd': end = max(8, 14) = 14
i = 10, char 'e': end = max(14, 15) = 15
...
i = 15, char 'e': end = 15. Since i == end (15 == 15):
  -> Cut Partition 2!
  -> Size: 15 - 9 + 1 = 7
  -> New start = 16

Partition 3:
i = 16, char 'h': end = max(15, 19) = 19
i = 17, char 'i': end = max(19, 22) = 22
i = 18, char 'j': end = max(22, 23) = 23
...
i = 23, char 'j': end = 23. Since i == end (23 == 23):
  -> Cut Partition 3!
  -> Size: 23 - 16 + 1 = 8
  -> New start = 24

Result: [9, 7, 8]
```

---

### Solved Examples with Multiple Inputs

| Input String $s$ | Last Occurrences | Partitions Identified | Slices | Output Lengths |
|---|---|---|---|---|
| `"ababcbacadefegdehijhklij"` | a:8, b:5, c:7, d:14, e:15, ... | [0..8], [9..15], [16..23] | `"ababcbaca"`, `"defegde"`, `"hijhklij"` | `[9, 7, 8]` |
| `"eccbbbbdec"` | e:8, c:9, b:6, d:7 | [0..9] | `"eccbbbbdec"` | `[10]` |
| `"abcdef"` | a:0, b:1, c:2, d:3, e:4, f:5 | [0..0], [1..1], [2..2], [3..3], [4..4], [5..5] | 6 single-character partitions | `[1, 1, 1, 1, 1, 1]` |
| `"a"` | a:0 | [0..0] | `"a"` | `[1]` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def partitionLabels(self, s: str) -> list[int]:
        # Precompute last occurrence of each character
        last: dict[str, int] = {ch: i for i, ch in enumerate(s)}
        
        result: list[int] = []
        start: int = 0
        end: int = 0
        
        # Greedy scan to extend partition boundaries
        for i, ch in enumerate(s):
            end = max(end, last[ch])
            if i == end:
                result.append(end - start + 1)
                start = i + 1
                
        return result
```

#### C++17
```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::vector<int> partitionLabels(const std::string& s) {
        int last[26] = {0};
        int n = static_cast<int>(s.size());
        
        // Record the last seen index for each character
        for (int i = 0; i < n; ++i) {
            last[s[i] - 'a'] = i;
        }
        
        std::vector<int> result;
        int start = 0;
        int end = 0;
        
        // Linear scan to identify earliest safe partition cuts
        for (int i = 0; i < n; ++i) {
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

#### Java 17
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<Integer> partitionLabels(String s) {
        int[] last = new int[26];
        int n = s.length();
        
        // Record the last occurrence of each character
        for (int i = 0; i < n; i++) {
            last[s.charAt(i) - 'a'] = i;
        }
        
        List<Integer> result = new ArrayList<>();
        int start = 0;
        int end = 0;
        
        // Greedily expand boundaries until current index meets furthest bound
        for (int i = 0; i < n; i++) {
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

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |s|$. We iterate through the string twice: once to populate the `last` array and once to compute partition boundaries. Both passes take $\mathcal{O}(N)$ time.
- **Space Complexity:** $\mathcal{O}(|\Sigma|) = \mathcal{O}(1)$ auxiliary space, where $\Sigma$ is the alphabet size ($|\Sigma| = 26$ lowercase English letters). The output list occupies $\mathcal{O}(K)$ space where $K \le 26$.

---

### Takeaway Pattern & Interview Traps

1. **Equivalence to Interval Merging:** This problem can also be viewed as merging overlapping intervals: each distinct character defines an interval $[\text{first}(c), \text{last}(c)]$. Finding non-overlapping connected components is equivalent to interval merging, but tracking running `end` achieves this in $\mathcal{O}(N)$ without sorting!
2. **Fixed Alphabet Space Optimization:** Use a direct array `int last[26]` rather than a hash map in C++/Java for cache locality and guaranteed $\mathcal{O}(1)$ lookups.
3. **Partition Cut Condition:** The partition is finalized precisely when $i == \text{end}$. Remember to reset $\text{start} = i + 1$ immediately after adding the length $\text{end} - \text{start} + 1$.