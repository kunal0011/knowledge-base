---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 238: Product of Array Except Self"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - prefix-sum
  - amazon
  - google
  - meta
  - apple
---

# LeetCode 238: Product of Array Except Self

**Target Companies:** Amazon (Top #1 Classic), Google, Meta, Apple, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Prefix & Suffix Accumulation in $\mathcal{O}(1)$ Auxiliary Space

---

### Problem Statement

Given an integer array `nums`, return an array `answer` such that `answer[i]` is equal to the product of all the elements of `nums` except `nums[i]`.

The product of any prefix or suffix of `nums` is **guaranteed** to fit in a **32-bit** integer.

You must write an algorithm that runs in **$\mathcal{O}(n)$ time** and **without using the division operation**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `List[int]`
- **Constraints:**
  - $2 \le \text{nums.length} \le 10^5$
  - $-30 \le \text{nums}[i] \le 30$
  - The product of any prefix or suffix of `nums` is guaranteed to fit in a 32-bit signed integer.
- **Follow up:** Can you solve the problem in $\mathcal{O}(1)$ extra memory space? (The output array does not count as extra space for space complexity analysis.)

---

### Key Idea & Intuition

#### 1. Why Division is Prohibited
If division were allowed, we could compute the product of all elements $P = \prod \text{nums}[j]$ and set $\text{ans}[i] = P / \text{nums}[i]$. However:
- The problem explicitly forbids division.
- Furthermore, handling zeros (`nums[i] == 0`) requires complicated branching (single zero vs. multiple zeros).

#### 2. Prefix and Suffix Decomposition
For any index $i$:
$$\text{ans}[i] = \underbrace{(\text{nums}[0] \times \text{nums}[1] \times \dots \times \text{nums}[i-1])}_{\text{Prefix product strictly to the left}} \times \underbrace{(\text{nums}[i+1] \times \dots \times \text{nums}[n-1])}_{\text{Suffix product strictly to the right}}$$

#### 3. In-Place Output Re-use for $\mathcal{O}(1)$ Auxiliary Space
Instead of allocating two separate arrays `prefix[n]` and `suffix[n]`:
1. **Pass 1 (Left-to-Right):**
   - Populate `res[i]` with the product of all numbers to the left of index $i$.
2. **Pass 2 (Right-to-Left):**
   - Maintain a running scalar variable `suffix = 1`.
   - Multiply `res[i]` by `suffix`, then update `suffix *= nums[i]`.
This computes the exact solution in two passes using $\mathcal{O}(1)$ additional memory beyond the returned answer array.

---

### Solution Approach (Step-by-Step)

1. **Initialize Output Array:**
   - Allocate `res` of size $n$, initialized with $1$.
2. **Left Pass (Prefix Accumulation):**
   - Initialize `prefix = 1`.
   - For $i$ from $0$ to $n - 1$:
     - `res[i] = prefix`
     - `prefix *= nums[i]`
3. **Right Pass (Suffix Accumulation):**
   - Initialize `suffix = 1`.
   - For $i$ from $n - 1$ down to $0$:
     - `res[i] *= suffix`
     - `suffix *= nums[i]`
4. **Return Result:**
   - Return `res`.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [1, 2, 3, 4]`

```
Step 1: Left Pass (res[i] = product of elements to left of i)
i=0: res[0] = 1,          prefix becomes 1 * 1 = 1
i=1: res[1] = 1,          prefix becomes 1 * 2 = 2
i=2: res[2] = 2,          prefix becomes 2 * 3 = 6
i=3: res[3] = 6,          prefix becomes 6 * 4 = 24
Array after Left Pass: res = [1, 1, 2, 6]

Step 2: Right Pass (multiply by running suffix from right)
suffix = 1
i=3: res[3] = 6 * 1 = 6,   suffix becomes 1 * 4 = 4
i=2: res[2] = 2 * 4 = 8,   suffix becomes 4 * 3 = 12
i=1: res[1] = 1 * 12 = 12, suffix becomes 12 * 2 = 24
i=0: res[0] = 1 * 24 = 24, suffix becomes 24 * 1 = 24

Final Output: [24, 12, 8, 6]
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Prefix Pass `res` | Suffix Multiplications | Output | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 2, 3, 4]` | `[1, 1, 2, 6]` | $\times [24, 12, 4, 1]$ | `[24, 12, 8, 6]` | Standard all-positive array |
| `[-1, 1, 0, -3, 3]` | `[1, -1, -1, 0, 0]` | $\times [0, 0, -9, 3, 1]$ | `[0, 0, 9, 0, 0]` | Single zero: only zero's index gets non-zero product |
| `[0, 0]` | `[1, 0]` | $\times [0, 1]$ | `[0, 0]` | Multiple zeros: all outputs are 0 |
| `[2, 3]` | `[1, 2]` | $\times [3, 1]$ | `[3, 2]` | Minimum length $N = 2$ |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        
        # Left pass: prefix products
        prefix = 1
        for i in range(n):
            res[i] = prefix
            prefix *= nums[i]
            
        # Right pass: multiply by suffix products
        suffix = 1
        for i in range(n - 1, -1, -1):
            res[i] *= suffix
            suffix *= nums[i]
            
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class Solution {
public:
    std::vector<int> productExceptSelf(std::vector<int>& nums) {
        int n = nums.size();
        std::vector<int> res(n, 1);

        int prefix = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = prefix;
            prefix *= nums[i];
        }

        int suffix = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] *= suffix;
            suffix *= nums[i];
        }

        return res;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];

        int prefix = 1;
        for (int i = 0; i < n; i++) {
            res[i] = prefix;
            prefix *= nums[i];
        }

        int suffix = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= suffix;
            suffix *= nums[i];
        }

        return res;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$. The algorithm performs exactly two linear passes over the array of size $n$, each doing $\mathcal{O}(1)$ arithmetic assignments per element.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. The output array `res` does not count towards auxiliary memory per problem specifications, and only scalar variables `prefix` and `suffix` are used.

---

### Takeaway Pattern & Interview Traps

1. **The Division Trap:**
   - Division is disqualified explicitly by the problem prompt. Even if allowed, zero handling creates pitfalls:
     - 0 zeros: simple division.
     - 1 zero: all elements are 0 except the index with the zero.
     - $\ge 2$ zeros: all elements are 0.
   - The prefix/suffix two-pass approach handles all zero cases uniformly without special conditionals.
2. **Off-by-One in Accumulation:**
   - Always assign `res[i] = prefix` *before* multiplying `prefix *= nums[i]`, ensuring that `nums[i]` itself is excluded from `res[i]`.
