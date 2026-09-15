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
---

# LeetCode 49: Group Anagrams

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple  
**Difficulty:** Medium  
**Topic:** Hash Map with Canonical Frequency Key / Sorted Tuple

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

- **Canonical Key Invariant:**
  - Two words are anagrams if and only if they map to the exact same canonical representation.
  - **Representation 1 (Character Count Tuple):** A 26-element tuple of counts `(count('a'), count('b'), ..., count('z'))`. Takes $O(K)$ time per word of length $K$.
  - **Representation 2 (Sorted String):** `"".join(sorted(s))`. Takes $O(K \log K)$ per word.
- **Hash Table Grouping:**
  - Map `canonical_key -> list of original words`.
  - Values of the hash table represent the final grouped anagrams.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
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
        
        for (const auto& s : strs) {
            std::string key = s;
            std::sort(key.begin(), key.end());
            groups[key].push_back(s);
        }

        std::vector<std::vector<std::string>> result;
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

- **Time Complexity:** $O(N \times K \log K)$ where $N$ is string count and $K$ is max length (or $O(N \times K)$ with 26-element counting key).
- **Space Complexity:** $O(N \times K)$ to store grouped strings in the hash map.
