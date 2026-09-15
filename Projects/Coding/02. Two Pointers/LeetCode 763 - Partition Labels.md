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
  - amazon
  - google
---

# LeetCode 763: Partition Labels

**Target Companies:** Amazon (Top #1 Classic), Google, Meta  
**Difficulty:** Medium  
**Topic:** Greedy Two Pointers / Last Occurrence Sweep

---

### Problem Statement

You are given a string `s`. We want to partition the string into as many parts as possible so that each letter appears in at most one part.

Note that the partition is done so that after concatenating all the parts in order, the resultant string should be `s`.

Return a list of integers representing the **size of these parts**.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {ch: i for i, ch in enumerate(s)}
        res = []
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
        for (int i = 0; i < s.size(); ++i) {
            last[s[i] - 'a'] = i;
        }

        std::vector<int> result;
        int start = 0, end = 0;
        for (int i = 0; i < s.size(); ++i) {
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

- **Time Complexity:** $O(N)$ — One pass to find last indices, one pass to slice partitions.
- **Space Complexity:** $O(1)$ — 26-element array for alphabet last positions.
