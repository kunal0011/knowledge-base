---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 402: Remove K Digits"
tags:
  - leetcode
  - coding
  - greedy
  - monotonic-stack
  - string
  - amazon
  - google
---

# LeetCode 402: Remove K Digits

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Medium  
**Topic:** Greedy / Monotonic Stack / String  

---

### Problem Statement

Given string `num` representing a non-negative integer, and an integer `k`, return the smallest possible integer after removing `k` digits from `num`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `num`: `str` / `string` ($1 \le |num| \le 10^5$, digits only).
  - `k`: `int` ($0 \le k \le |num|$).
- **Output:**
  - `str` / `string` — the smallest integer represented as a string without leading zeros, or `"0"` if empty.
- **Constraints:**
  - $1 \le k \le \text{num.length} \le 10^5$
  - `num` consists of only digits.
  - `num` does not have any leading zeros except for the zero itself.

---

### Key Idea & Intuition

In positional decimal notation, the most significant digits are on the **far left**.
Comparing two numbers of the same length:
$$d_1 d_2 \dots d_m \quad \text{vs} \quad d'_1 d'_2 \dots d'_m$$
Their relative magnitude is entirely determined by the **first index from the left where their digits differ**.

Therefore:
- If we have two adjacent digits $d_i$ and $d_{i+1}$ where $d_i > d_{i+1}$:
  - Removing $d_i$ shifts $d_{i+1}$ into the higher place value position, making the entire number strictly smaller.
- If $d_i \le d_{i+1}$, removing $d_i$ would promote $d_{i+1}$ (which is larger or equal) to a higher place value, making the number larger or the same.

#### Monotonic Increasing Stack:
1. Maintain a stack of digits in monotonically increasing order.
2. For each digit $c$ in `num`:
   - While $k > 0$, `stack` is not empty, and `stack[-1] > c`:
     - Pop `stack` (greedy deletion of the larger predecessor) and decrement $k \mathrel{-}= 1$.
   - Push $c$ to `stack`.
3. **If $k > 0$ after scanning all digits:**
   - The remaining digits in the stack are already monotonically increasing (e.g. `"12345"`).
   - In a monotonically increasing sequence, the largest digits are at the tail.
   - Pop the last $k$ digits from the stack: `stack = stack[:-k]`.
4. **Strip Leading Zeros:**
   - Remove any leading `'0'` characters from the resulting string.
   - If the resulting string is empty, return `"0"`.

---

### Solution Approach (Step-by-Step)

1. If $k == \text{len}(num)$, return `"0"`.
2. Initialize an empty stack `stack = []`.
3. For each digit `digit` in `num`:
   - While `k > 0` and `stack` and `stack[-1] > digit`:
     - `stack.pop()`
     - `k -= 1`
   - `stack.append(digit)`
4. If $k > 0$:
   - Truncate stack: `stack = stack[:-k]`.
5. Join into string and strip leading zeros: `result = "".join(stack).lstrip('0')`.
6. Return `result if result else "0"`.

---

### Visual Algorithm Walkthrough

For `num = "1432219"`, `k = 3`:

```
i=0 ('1'): stack = ['1']
i=1 ('4'): 4 > 1 -> stack = ['1', '4']
i=2 ('3'): 4 > 3 and k > 0 -> Pop 4! k becomes 2.
           1 < 3 -> push 3.
           stack = ['1', '3']
i=3 ('2'): 3 > 2 and k > 0 -> Pop 3! k becomes 1.
           1 < 2 -> push 2.
           stack = ['1', '2']
i=4 ('2'): 2 <= 2 -> push 2.
           stack = ['1', '2', '2']
i=5 ('1'): 2 > 1 and k > 0 -> Pop 2! k becomes 0.
           k is now 0 -> no more pops possible!
           push 1.
           stack = ['1', '2', '1']
i=6 ('9'): push 9.
           stack = ['1', '2', '1', '9']

k = 0, no trailing truncation needed.
Result string: "1219"
```

For `num = "10200"`, `k = 1`:

```
i=0 ('1'): stack = ['1']
i=1 ('0'): 1 > 0 and k > 0 -> Pop 1! k becomes 0.
           push 0.
           stack = ['0']
i=2 ('2'): push 2. stack = ['0', '2']
i=3 ('0'): push 0. stack = ['0', '2', '0']  (k is 0, cannot pop 2)
i=4 ('0'): push 0. stack = ['0', '2', '0', '0']

Result after stripping leading zeros from "0200":
"200".
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `num = "1432219"`, `k = 3`
- **Output:** `"1219"`

#### Example 2:
- **Input:** `num = "10200"`, `k = 1`
- **Output:** `"200"`

#### Example 3 (All Digits Removed):
- **Input:** `num = "10"`, `k = 2`
- **Output:** `"0"`

#### Example 4 (Monotonically Increasing Array):
- **Input:** `num = "123456"`, `k = 3`
- **Tracing:** No pop during loop. Truncate trailing 3 digits.
- **Output:** `"123"`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        if k >= len(num):
            return "0"
            
        stack = []
        for digit in num:
            while k > 0 and stack and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)
            
        # If k remaining, trim from the tail
        if k > 0:
            stack = stack[:-k]
            
        # Strip leading zeros
        result = "".join(stack).lstrip('0')
        return result if result else "0"
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::string removeKdigits(std::string num, int k) {
        int n = static_cast<int>(num.size());
        if (k >= n) return "0";
        
        std::string stack;
        
        for (char digit : num) {
            while (k > 0 && !stack.empty() && stack.back() > digit) {
                stack.pop_back();
                k--;
            }
            stack.push_back(digit);
        }
        
        // If k removals still remain, pop from the end
        while (k > 0 && !stack.empty()) {
            stack.pop_back();
            k--;
        }
        
        // Remove leading zeros
        int start = 0;
        while (start < stack.size() && stack[start] == '0') {
            start++;
        }
        
        std::string result = stack.substr(start);
        return result.empty() ? "0" : result;
    }
};
```

#### Java 17
```java
class Solution {
    public String removeKdigits(String num, int k) {
        int n = num.length();
        if (k >= n) return "0";
        
        char[] stack = new char[n];
        int top = 0;
        
        for (int i = 0; i < n; i++) {
            char c = num.charAt(i);
            while (k > 0 && top > 0 && stack[top - 1] > c) {
                top--;
                k--;
            }
            stack[top++] = c;
        }
        
        // If k > 0, remove trailing elements
        top -= k;
        
        // Find first non-zero character
        int start = 0;
        while (start < top && stack[start] == '0') {
            start++;
        }
        
        if (start == top) {
            return "0";
        }
        
        return new String(stack, start, top - start);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Each digit is pushed to the stack once and popped at most once.
  - Stripping leading zeros and slicing takes $\mathcal{O}(n)$ time.
  - Overall time is $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(n)$ auxiliary space
  - The stack stores at most $n$ characters.

---

### Takeaway Pattern & Interview Traps

- **The Remaining $K$ Trap:** If the input string is already sorted (`"12345"`), no digit will ever be popped during the loop. Forgetting `stack = stack[:-k]` or `top -= k` is the most common bug!
- **Leading Zeros Trap:** When removing `'1'` from `"10200"`, the result is `"0200"`. Must strip leading zeros to return `"200"`.
- **All Zeros or Empty String:** If all digits vanish or become `'0'`, return `"0"`, never `""`.