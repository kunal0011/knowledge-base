---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 49: Group Anagrams"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 49: Group Anagrams

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Hash Map with Canonical Frequency Key / Sorted String Hashing

---

### Problem Statement

Given an array of strings `strs`, group the **anagrams** together. You can return the answer in **any order**.

An **Anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

---

### Input & Output Formats & Constraints

- **Input:** `strs: List[str]`
- **Output:** `List[List[str]]`
- **Constraints:**
  - $1 \le \text{strs.length} \le 10^4$
  - $0 \le \text{strs}[i].\text{length} \le 100$
  - `strs[i]` consists of lowercase English letters.

---

### Key Idea & Intuition

#### Canonical Key Invariant:
Two words are anagrams if and only if they map to the exact same canonical signature:
1. **Approach A — Sorted String as Key:**
   - Sorting the characters of a word produces a canonical string. For example, `"eat"`, `"tea"`, and `"ate"` all sort to `"aet"`.
   - Time per word of length $K$: $\mathcal{O}(K \log K)$.
2. **Approach B — Character Frequency Array as Key:**
   - Since words consist only of lowercase English letters (`'a'` to `'z'`), we can count character frequencies into an array of size 26.
   - For example, `"abb"` maps to $(1, 2, 0, \dots, 0)$.
   - Time per word of length $K$: $\mathcal{O}(K)$.

#### Hash Map Grouping:
We maintain a hash map where:
- Key = Canonical representation (either sorted string or frequency tuple)
- Value = List of original words sharing that key
After iterating through all strings, the values of the hash map yield the grouped anagram lists.

---

### Solution Approach (Step-by-Step)

1. **Initialize Hash Map:**
   - Create a hash map mapping each canonical key to a list of strings (`defaultdict(list)` in Python, `unordered_map<string, vector<string>>` in C++, `HashMap<String, List<String>>` in Java).
2. **Iterate Through Strings:**
   - For each word $s$ in `strs`:
     - Sort its characters to form a canonical key (or compute its 26-element character count signature).
     - Append $s$ to `map[canonical_key]`.
3. **Collect Results:**
   - Extract and return all value lists from the hash map.

---

### Visual Algorithm Walkthrough

#### Example: `strs = ["eat", "tea", "tan", "ate", "nat", "bat"]`

```
Word        Sorted Key      Hash Map State
----------------------------------------------------------------------
"eat"   ->  "aet"       ->  {"aet": ["eat"]}
"tea"   ->  "aet"       ->  {"aet": ["eat", "tea"]}
"tan"   ->  "ant"       ->  {"aet": ["eat", "tea"], "ant": ["tan"]}
"ate"   ->  "aet"       ->  {"aet": ["eat", "tea", "ate"], "ant": ["tan"]}
"nat"   ->  "ant"       ->  {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}
"bat"   ->  "abt"       ->  {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"], "abt": ["bat"]}

Final Output:
[
  ["eat", "tea", "ate"],
  ["tan", "nat"],
  ["bat"]
]
```

---

### Solved Examples with Multiple Inputs

| Input `strs` | Grouping Process | Output Groups |
| :--- | :--- | :--- |
| `["eat","tea","tan","ate","nat","bat"]` | Keys: `"aet"` (3), `"ant"` (2), `"abt"` (1) | `[["bat"],["nat","tan"],["ate","eat","tea"]]` |
| `[""]` | Key: `""` | `[[""]]` |
| `["a"]` | Key: `"a"` | `[["a"]]` |
| `["ab", "ba", "abc", "cba", "bca"]` | Keys: `"ab"` (2), `"abc"` (3) | `[["ab", "ba"], ["abc", "cba", "bca"]]` |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # Using 26-element frequency tuple for O(N * K) time
        groups = defaultdict(list)
        
        for s in strs:
            count = [0] * 26
            for ch in s:
                count[ord(ch) - ord('a')] += 1
            groups[tuple(count)].append(s)
            
        return list(groups.values())
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        std::unordered_map<std::string, std::vector<std::string>> groups;
        
        for (const std::string& s : strs) {
            std::string key = s;
            std::sort(key.begin(), key.end());
            groups[key].push_back(s);
        }

        std::vector<std::vector<std::string>> result;
        result.reserve(groups.size());
        for (auto& pair : groups) {
            result.push_back(std::move(pair.second));
        }
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> groups = new HashMap<>();

        for (String s : strs) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);

            groups.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
        }

        return new ArrayList<>(groups.values());
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - **Sorting Approach:** $\mathcal{O}(N \cdot K \log K)$, where $N$ is the number of strings and $K$ is the maximum length of a string in `strs`. Sorting each string takes $\mathcal{O}(K \log K)$.
  - **Frequency Count Approach:** $\mathcal{O}(N \cdot K)$, where counting frequencies of 26 characters takes $\mathcal{O}(K)$ per string.
- **Space Complexity:** $\mathcal{O}(N \cdot K)$ total auxiliary space required to store all strings inside the hash map groups.

---

### Takeaway Pattern & Interview Traps

1. **Choosing the Key Representation:**
   - In Python, a tuple of 26 integers is hashable and avoids string sorting overhead ($\mathcal{O}(K)$ vs $\mathcal{O}(K \log K)$).
   - In Java and C++, sorting a string of length $K \le 100$ is extremely fast in cache memory and simpler to write without building delimiter-separated strings (e.g. `"1#2#0#..."`).
2. **Delimiters in Count Strings:**
   - If serializing frequency arrays to strings (e.g. in Java or C++), remember to insert delimiters (e.g. `'#'`). Otherwise, counts like `1` followed by `11` would collide with `11` followed by `1`.
