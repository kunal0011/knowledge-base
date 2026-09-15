---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 268: Missing Number"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - xor
  - math
  - amazon
  - google
---

# LeetCode 268: Missing Number

**Target Companies:** Amazon (All-Time Classic), Microsoft, Google, Apple, Meta  
**Difficulty:** Easy  
**Topic:** XOR Index-Value Annihilation & Gauss Summation  

---

### Problem Statement

Given an array `nums` containing $n$ distinct numbers in the range $[0, n]$, return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only $O(1)$ extra space complexity and $O(n)$ runtime complexity?

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (missing number)
- **Constraints:**
  - $n == \text{nums.length}$
  - $1 \le n \le 10^4$
  - $0 \le \text{nums}[i] \le n$
  - All the numbers of `nums` are **unique**.

---

### Key Idea & Intuition

#### Method 1: XOR Index-Value Annihilation (Overflow-Safe)
Consider the full expected sequence of numbers from $0$ to $n$, and the actual sequence of numbers present in `nums`:
- The expected sequence has $n + 1$ numbers: $[0, 1, 2, \dots, n]$.
- The array `nums` has $n$ numbers.
- If we XOR every index $i \in [0, n - 1]$ and the final number $n$ with every value `nums[i]`:
  $$\text{ans} = n \oplus \bigoplus_{i=0}^{n-1} (i \oplus \text{nums}[i])$$
- Every number that exists in the array appears twice (once as a value, once as an index) and annihilates to $0$ ($x \oplus x = 0$).
- The only number that appears exactly once is the missing number!
- **Key Advantage:** XOR is immune to arithmetic integer overflow.

#### Method 2: Gauss Summation Formula
$$\text{Expected Sum} = \sum_{i=0}^n i = \frac{n(n + 1)}{2}$$
$$\text{Missing Number} = \text{Expected Sum} - \sum \text{nums}$$
Takes $O(N)$ time and $O(1)$ space, but requires 64-bit integer handling when $n$ is very large to avoid sum overflow.

---

### Solution Approach (Step-by-Step)

1. Initialize `ans = len(nums)`.
2. For index `i` and value `num` in enumerate(`nums`):
   - `ans ^= (i ^ num)`.
3. Return `ans`.

---

### Visual Algorithm Walkthrough

```
nums = [3, 0, 1], n = len(nums) = 3
Expected set: {0, 1, 2, 3}
Actual set:   {0, 1, 3}

Index (i):   0   1   2   3 (n)
Value (num): 3   0   1

Cumulative XOR:
ans = 3 ^ (0 ^ 3) ^ (1 ^ 0) ^ (2 ^ 1)
Rearranging by values:
ans = (0 ^ 0) ^ (1 ^ 1) ^ (3 ^ 3) ^ 2
    =    0    ^    0    ^    0    ^ 2
    = 2

Missing Number = 2!
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Missing Value
- **Input:** `nums = [3, 0, 1]`
- **Output:** `2`

#### Example 2: Missing First Number (Zero)
- **Input:** `nums = [1, 2]` ($n = 2$)
- **Trace:** $2 \oplus (0 \oplus 1) \oplus (1 \oplus 2) = (1 \oplus 1) \oplus (2 \oplus 2) \oplus 0 = 0$.
- **Output:** `0`

#### Example 3: Missing Last Number ($n$)
- **Input:** `nums = [0, 1]` ($n = 2$)
- **Trace:** $2 \oplus (0 \oplus 0) \oplus (1 \oplus 1) = 2$.
- **Output:** `2`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        ans = len(nums)
        for i, num in enumerate(nums):
            ans ^= (i ^ num)
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int missingNumber(std::vector<int>& nums) {
        int n = nums.size();
        int ans = n;
        for (int i = 0; i < n; ++i) {
            ans ^= (i ^ nums[i]);
        }
        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int ans = n;
        for (int i = 0; i < n; i++) {
            ans ^= (i ^ nums[i]);
        }
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass through the array performing $O(1)$ XOR operations per element.
- **Space Complexity:** $O(1)$ auxiliary space — Only a single accumulator variable.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Dual XOR Cancellation between Index Space and Value Space.
- **Trap:** Forgetting that the expected numbers go up to $n$ (while indices only go up to $n - 1$). You must initialize `ans = n` so that number $n$ is included in the XOR pool.