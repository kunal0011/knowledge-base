---
date: "2026-09-15"
type: leetcode-solution
category: "Bit Manipulation"
folder: "14. Bit Manipulation"
title: "LeetCode 260: Single Number III"
tags:
  - leetcode
  - coding
  - bit-manipulation
  - xor
  - partitioning
  - amazon
  - google
---

# LeetCode 260: Single Number III

**Target Companies:** Google, Amazon, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** Differential Bit Partitioning via Lowest Set Bit (`x & (-x)`)  

---

### Problem Statement

Given an integer array `nums`, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once. You can return the answer in **any order**.

You must write an algorithm that runs in linear runtime complexity and uses only constant extra space.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[int]` (the two unique elements)
- **Constraints:**
  - $2 \le \text{nums.length} \le 3 \times 10^4$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$
  - Each integer in `nums` will appear twice, only two integers will appear once.

---

### Key Idea & Intuition

Let the two unique numbers be $A$ and $B$.
1. If we XOR all elements in the array:
   $$\text{XOR}_{\text{all}} = A \oplus B$$
   Because $A \ne B$, $\text{XOR}_{\text{all}} \ne 0$.
2. In $\text{XOR}_{\text{all}}$, any bit that is `1` represents a position where **$A$ and $B$ differ** (one has a `1` at that position, and the other has a `0`).
3. We can isolate the **lowest set bit** (rightmost 1) of $\text{XOR}_{\text{all}}$ using the two's complement identity:
   $$\text{diff\_bit} = \text{XOR}_{\text{all}} \ \& \ (-\text{XOR}_{\text{all}})$$
4. **Two-Group Partition:**
   We divide all elements in `nums` into two independent subsets based on whether they have this bit set:
   - **Group 1:** All numbers with `num & diff_bit != 0`. (Contains $A$, and duplicate pairs).
   - **Group 2:** All numbers with `num & diff_bit == 0`. (Contains $B$, and duplicate pairs).
5. XORing each group independently cancels out all duplicate pairs within each group, cleanly isolating $A$ in Group 1 and $B$ in Group 2!

---

### Solution Approach (Step-by-Step)

1. Compute `xor_all = 0`. For `x` in `nums`: `xor_all ^= x`.
2. Extract lowest set bit: `diff_bit = xor_all & (-xor_all)`.
   *(Note: In C++/Java, cast to unsigned/64-bit to prevent overflow when `xor_all == INT_MIN`).*
3. Initialize `a = 0`, `b = 0`.
4. For each `x` in `nums`:
   - If `x & diff_bit`: `a ^= x`
   - Else: `b ^= x`
5. Return `[a, b]`.

---

### Visual Algorithm Walkthrough

```
nums = [1, 2, 1, 3, 2, 5]
Unique elements are 3 and 5.

Step 1: Compute cumulative XOR of all elements
  xor_all = 1 ^ 2 ^ 1 ^ 3 ^ 2 ^ 5
          = (1 ^ 1) ^ (2 ^ 2) ^ (3 ^ 5)
          = 0 ^ 0 ^ (011_2 ^ 101_2)
          = 110_2  (Decimal 6)

Step 2: Isolate lowest set bit of 6
  xor_all       = 0110_2
  -xor_all      = 1010_2 (Two's complement)
  diff_bit = 6 & (-6) = 0010_2 (Bit 1 is set!)

Step 3: Partition into two subsets based on Bit 1:
  Bit 1 is set (num & 2 != 0):
    Numbers: [2, 2, 3]
    Group 1 XOR = 2 ^ 2 ^ 3 = 3 -> a = 3!

  Bit 1 is NOT set (num & 2 == 0):
    Numbers: [1, 1, 5]
    Group 2 XOR = 1 ^ 1 ^ 5 = 5 -> b = 5!

Output: [3, 5]
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed Input
- **Input:** `nums = [1, 2, 1, 3, 2, 5]`
- **Output:** `[3, 5]`

#### Example 2: Negative Numbers
- **Input:** `nums = [-1, 0]`
- **Trace:**
  - $A = -1, B = 0$. $\text{XOR} = -1$.
  - Lowest set bit of $-1$ is $1$.
  - Group 1: `[-1]`, Group 2: `[0]`.
- **Output:** `[-1, 0]`

#### Example 3: Two Elements Only
- **Input:** `nums = [0, 1]`
- **Output:** `[0, 1]`

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        xor_all = 0
        for num in nums:
            xor_all ^= num
            
        # Isolate the rightmost set bit
        diff_bit = xor_all & (-xor_all)
        
        a = 0
        b = 0
        for num in nums:
            if num & diff_bit:
                a ^= num
            else:
                b ^= num
                
        return [a, b]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> singleNumber(std::vector<int>& nums) {
        int xorAll = 0;
        for (int num : nums) {
            xorAll ^= num;
        }

        // Prevent INT_MIN overflow when negating
        long long diffBit = static_cast<long long>(xorAll) & (-static_cast<long long>(xorAll));

        int a = 0, b = 0;
        for (int num : nums) {
            if (num & diffBit) {
                a ^= num;
            } else {
                b ^= num;
            }
        }

        return {a, b};
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] singleNumber(int[] nums) {
        int xorAll = 0;
        for (int num : nums) {
            xorAll ^= num;
        }

        // Isolate lowest set bit using unsigned subtraction
        int diffBit = xorAll & (-xorAll);

        int a = 0;
        int b = 0;
        for (int num : nums) {
            if ((num & diffBit) != 0) {
                a ^= num;
            } else {
                b ^= num;
            }
        }

        return new int[]{a, b};
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(N)$ — Exactly two linear passes: one to compute total XOR, and one to partition and accumulate the answers.
- **Space Complexity:** $O(1)$ auxiliary space — Only constant scalar variables.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Two-Group XOR Disambiguation via Lowest Set Bit (`x & (-x)`).
- **Trap:** Signed 32-bit Integer Overflow in C++: If `xorAll == INT_MIN` ($-2^{31}$), evaluating `-xorAll` in 32-bit signed arithmetic invokes undefined behavior (overflows $2^{31} - 1$). Always cast `xorAll` to `long long` (or `unsigned int`) before negation.