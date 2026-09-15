---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 136: Single Number"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - xor
  - fundamentals
  - amazon
  - google
---

# LeetCode 136: Single Number

**Target Companies:** Amazon (All-Time Most Asked), Google, Microsoft, Apple, Meta  
**Difficulty:** Easy  
**Topic:** XOR Self-Inverse Cancellation Invariant  

---

### Problem Statement

Given a **non-empty** array of integers `nums`, every element appears *twice* except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (the element that appears only once)
- **Constraints:**
  - $1 \le \text{nums.length} \le 3 \times 10^4$
  - $-3 \times 10^4 \le \text{nums}[i] \le 3 \times 10^4$
  - Each element in the array appears twice except for one element which appears only once.

---

### Key Idea & Intuition

The bitwise **XOR** ($\oplus$) operation possesses four fundamental algebraic properties:
1. **Identity Element:** $x \oplus 0 = x$
2. **Self-Inverse (Nilpotence):** $x \oplus x = 0$
3. **Commutativity:** $x \oplus y = y \oplus x$
4. **Associativity:** $(x \oplus y) \oplus z = x \oplus (y \oplus z)$

Because XOR is commutative and associative, we can reorder the cumulative XOR of all elements in `nums` arbitrarily:
$$\text{XOR}_{\text{all}} = (a_1 \oplus a_1) \oplus (a_2 \oplus a_2) \oplus \dots \oplus (a_k \oplus a_k) \oplus u$$
Every duplicate pair reduces to $0$:
$$\text{XOR}_{\text{all}} = 0 \oplus 0 \oplus \dots \oplus 0 \oplus u = u$$
The cumulative XOR reduces all paired duplicates to zero, isolating the single unique element $u$ in $O(N)$ time and $O(1)$ space.

---

### Solution Approach (Step-by-Step)

1. Initialize accumulator `ans = 0`.
2. For each number `num` in `nums`:
   - `ans ^= num`.
3. Return `ans`.

---

### Visual Algorithm Walkthrough

```
nums = [4, 1, 2, 1, 2]

Initial: ans = 0

Step 1: ans = 0 ^ 4 = 4              (Binary: 100)
Step 2: ans = 4 ^ 1 = 5              (Binary: 101)
Step 3: ans = 5 ^ 2 = 7              (Binary: 111)
Step 4: ans = 7 ^ 1 = 6 (1 cancels!) (Binary: 110)
Step 5: ans = 6 ^ 2 = 4 (2 cancels!) (Binary: 100)

Rearranged proof:
ans = 4 ^ (1 ^ 1) ^ (2 ^ 2)
    = 4 ^ 0 ^ 0
    = 4

Result: 4
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Array
- **Input:** `nums = [2, 2, 1]`
- **Trace:** $0 \oplus 2 \oplus 2 \oplus 1 = (2 \oplus 2) \oplus 1 = 0 \oplus 1 = 1$.
- **Output:** `1`

#### Example 2: Negative Numbers
- **Input:** `nums = [-1, -1, -2]`
- **Trace:** $(-1 \oplus -1) \oplus -2 = 0 \oplus -2 = -2$.
- **Output:** `-2`

#### Example 3: Single Element Array
- **Input:** `nums = [1]`
- **Output:** `1`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            ans ^= num
        return ans
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    int singleNumber(std::vector<int>& nums) {
        int ans = 0;
        for (int num : nums) {
            ans ^= num;
        }
        return ans;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int num : nums) {
            ans ^= num;
        }
        return ans;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Single pass through array of $N$ integers.
- **Space Complexity:** $O(1)$ auxiliary space — Only one integer accumulator variable.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** XOR Pair Annihilation. Whenever elements appear in pairs with exactly one exception, XOR provides an instant $O(N)$ time and $O(1)$ space solution.
- **Trap:** Trying to use a Hash Set or Sorting: Sorting takes $O(N \log N)$ time, and a Hash Set requires $O(N)$ memory, which violates the strict $O(1)$ memory requirement.