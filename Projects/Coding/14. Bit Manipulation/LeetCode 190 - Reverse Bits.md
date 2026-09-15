---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 190: Reverse Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 190: Reverse Bits

**Target Companies:** Google, Amazon, Apple  
**Difficulty:** Easy  
**Topic:** Bitwise Extraction and Shifting

---

### Problem Statement

Reverse bits of a given 32 bits unsigned integer.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <cstdint>

class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        for (int i = 0; i < 32; ++i) {
            res = (res << 1) | (n & 1);
            n >>= 1;
        }
        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res = (res << 1) | (n & 1);
            n >>>= 1;
        }
        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(1)$ — Exactly 32 bit operations.
- **Space Complexity:** $O(1)$ constant auxiliary space.
