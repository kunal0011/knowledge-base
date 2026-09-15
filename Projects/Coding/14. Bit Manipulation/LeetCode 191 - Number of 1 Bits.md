---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 191: Number of 1 Bits"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - amazon
  - google
---

# LeetCode 191: Number of 1 Bits (Hamming Weight)

**Target Companies:** Amazon, Google, Microsoft, Apple  
**Difficulty:** Easy  
**Topic:** Brian Kernighan's Algorithm (`n & (n - 1)`)

---

### Problem Statement

Given a positive integer `n`, write a function that returns the number of set bits (1s) in its binary representation (also known as the **Hamming weight**).

---

### Key Idea & Intuition

- **Brian Kernighan's Bit Trick:**
  - $n \ \& \ (n - 1)$ always clears the **lowest set bit** (rightmost 1) of $n$ to 0!
  - Instead of looping 32 times, loop only as many times as there are 1-bits in $n$.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= (n - 1)
            count += 1
        return count
```

#### 2. C++ (C++17 / STL)
```cpp
class Solution {
public:
    int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= (n - 1);
            count++;
        }
        return count;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            n &= (n - 1);
            count++;
        }
        return count;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(K)$ where $K$ is the number of 1-bits ($\le 32$).
- **Space Complexity:** $O(1)$ constant space.
