---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 217: Contains Duplicate"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - apple
---

# LeetCode 217: Contains Duplicate

**Target Companies:** Amazon, Apple, Google, Microsoft  
**Difficulty:** Easy  
**Topic:** Hash Set Membership / Early Exit

---

### Problem Statement

Given an integer array `nums`, return `true` if any value appears **at least twice** in the array, and return `false` if every element is distinct.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `bool`
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-10^9 \le \text{nums}[i] \le 10^9$

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <unordered_set>

class Solution {
public:
    bool containsDuplicate(std::vector<int>& nums) {
        std::unordered_set<int> seen;
        for (int num : nums) {
            if (seen.count(num)) return true;
            seen.insert(num);
        }
        return false;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.HashSet;
import java.util.Set;

class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> seen = new HashSet<>();
        for (int num : nums) {
            if (seen.contains(num)) return true;
            seen.add(num);
        }
        return false;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ with early exit on the first detected duplicate.
- **Space Complexity:** $O(N)$ for the hash set.
