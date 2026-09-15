---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 402: Remove K Digits"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - greedy
  - amazon
  - google
---

# LeetCode 402: Remove K Digits

**Target Companies:** Amazon, Google, Microsoft, Bloomberg, Meta  
**Difficulty:** Medium  
**Topic:** Monotonic Stack / Greedy / String Processing

---

### Problem Statement

Given string `num` representing a non-negative integer, and an integer `k`, return *the smallest possible integer after removing `k` digits from `num`*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `num`: `str`, where $1 \le \text{len}(num) \le 10^5$. `num` consists of only digits (`'0'`–`'9'`).
  - `k`: `int`, where $0 \le k \le \text{len}(num)$.
- **Output:**
  - `str`: The smallest possible number represented as a string, with leading zeros stripped. If the number is zero or all digits are removed, return `"0"`.
- **Constraints:**
  - `num` does not have any leading zeros except for the number `'0'` itself.
  - $1 \le \text{num.length} \le 10^5$.
  - $0 \le k \le \text{num.length}$.

---

### Key Idea & Intuition

In positional base-10 numbers, the most significant digits (the leftmost digits) dominate the value of the number:
$$43000 > 39999$$
Even if a single leading digit is slightly smaller, the resulting number is drastically smaller overall.

Therefore, we greedily want each digit from left to right to be as small as possible:
- If a digit $d_i$ is followed by a smaller digit $d_{i+1}$ (i.e. $d_i > d_{i+1}$), removing $d_i$ shifts $d_{i+1}$ into the higher place value, immediately yielding a smaller number.
- Conversely, if digits are monotonically increasing ($12345$), the leftmost digits are already minimal, so the optimal digits to remove are at the tail (the least significant digits).

This requirement—popping larger preceding digits in favor of a smaller incoming digit—is the textbook definition of a **Monotonic Increasing Stack**:
1. Iterate through each digit $d$ in `num`.
2. While `k > 0`, `stack` is non-empty, and `stack[-1] > d`:
   - Pop `stack[-1]` and decrement `k` by 1.
3. Push $d$ to the stack.
4. If $k > 0$ after processing all digits (e.g., input was already sorted like `"12345"`):
   - Truncate the last $k$ digits from the stack.
5. Remove all leading zeros and return `"0"` if the string is empty.

---

### Solution Approach (Step-by-Step)

1. **Stack Traversal:**
   - Initialize an empty list/array `stack`.
   - For each char `d` in `num`:
     - While `k > 0` and `stack` and `stack[-1] > d`:
       - `stack.pop()`
       - `k -= 1`
     - `stack.append(d)`
2. **Handle Remaining Removals:**
   - If `k > 0`: `stack = stack[:-k]` (pop $k$ elements from the back).
3. **Leading Zero Cleanup:**
   - Convert `stack` to a string, strip leading `'0'` characters.
   - If the resulting string is empty, return `"0"`.

---

### Visual Algorithm Walkthrough

Let `num = "1432219"`, $k = 3$:

```
Digit: '1' -> Stack: ['1']
Digit: '4' -> 4 > 1 -> Stack: ['1', '4']
Digit: '3' -> 3 < 4 AND k > 0:
              Pop '4', k becomes 2.
              Push '3'. Stack: ['1', '3']
Digit: '2' -> 2 < 3 AND k > 0:
              Pop '3', k becomes 1.
              Push '2'. Stack: ['1', '2']
Digit: '2' -> 2 == 2 (non-decreasing) -> Stack: ['1', '2', '2']
Digit: '1' -> 1 < 2 AND k > 0:
              Pop '2', k becomes 0.
              Push '1'. Stack: ['1', '2', '1']
Digit: '9' -> k == 0 (no more removals allowed!) -> Stack: ['1', '2', '1', '9']

Result: "1219"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Decreasing Digits

- **Input:** `num = "1432219"`, `k = 3`
- **Output:** `"1219"`

#### Example 2: Leading Zeros Exposed

- **Input:** `num = "10200"`, `k = 1`
- **Tracing:**
  - $d = \text{'1'}$: stack = `['1']`
  - $d = \text{'0'}$: $0 < 1 \implies$ pop `1`, $k = 0$. Push `0`. Stack = `['0']`.
  - Remaining digits pushed: `['0', '2', '0', '0']`.
  - Stripping leading zero: `"200"`.
- **Output:** `"200"`

#### Example 3: All Digits Removed

- **Input:** `num = "10"`, `k = 2`
- **Tracing:**
  - $k=2$ removes all digits. Stack becomes empty.
  - Return fallback `"0"`.
- **Output:** `"0"`

#### Example 4: Monotonically Increasing Input

- **Input:** `num = "123456"`, `k = 3`
- **Tracing:**
  - No digits popped during traversal.
  - Remaining $k=3$ truncates last 3 digits $\implies$ `"123"`.
- **Output:** `"123"`

---

### Multi-Language Implementations

#### Python 3

```python
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = []

        for digit in num:
            # Pop larger previous digits if we still have deletions available
            while stack and k > 0 and stack[-1] > digit:
                stack.pop()
                k -= 1
            stack.append(digit)

        # If k deletions still remain, drop from the least significant end
        if k > 0:
            stack = stack[:-k]

        # Strip leading zeros and handle empty result
        result = "".join(stack).lstrip('0')
        return result if result else "0"
```

#### C++17

```cpp
#include <string>

class Solution {
public:
    std::string removeKdigits(std::string num, int k) {
        std::string stack = ""; // Use std::string directly as a stack
        
        for (char d : num) {
            while (!stack.empty() && k > 0 && stack.back() > d) {
                stack.pop_back();
                k--;
            }
            stack.push_back(d);
        }

        // Pop remaining k digits from the back
        while (k > 0 && !stack.empty()) {
            stack.pop_back();
            k--;
        }

        // Find first non-zero character
        int start = 0;
        int n = static_cast<int>(stack.size());
        while (start < n && stack[start] == '0') {
            start++;
        }

        std::string result = stack.substr(start);
        return result.empty() ? "0" : result;
    }
};
```

#### Java

```java
public class Solution {
    public String removeKdigits(String num, int k) {
        int n = num.length();
        if (k == n) return "0";

        char[] stack = new char[n];
        int top = 0; // stack pointer

        for (int i = 0; i < n; i++) {
            char d = num.charAt(i);
            while (top > 0 && k > 0 && stack[top - 1] > d) {
                top--;
                k--;
            }
            stack[top++] = d;
        }

        // If k > 0, remove from the back
        top -= k;

        // Skip leading zeros
        int start = 0;
        while (start < top && stack[start] == '0') {
            start++;
        }

        if (start == top) return "0";

        return new String(stack, start, top - start);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Each digit in `num` is pushed onto the stack once and popped at most once.
  - Stripping leading zeros takes $\mathcal{O}(N)$.
  - Total Time: $\mathcal{O}(N)$, where $N$ is the length of `num`.
- **Space Complexity:** $\mathcal{O}(N)$
  - In C++, using `std::string` as the stack uses $\mathcal{O}(N)$ auxiliary space.
  - In Java, the primitive `char[]` buffer takes $\mathcal{O}(N)$ auxiliary space.

---

### Takeaway Pattern & Interview Traps

1. **Greedy Principle of Digit Places:**
   - Minimizing numbers from left to right is always globally optimal because higher place values ($10^k$) outweigh any sum of lower place values ($10^{k-1} + \dots$).
2. **Remaining Removals ($k > 0$):**
   - For already non-decreasing inputs like `"12345"`, no characters are popped in the main loop. You must explicitly pop the remaining $k$ digits from the tail.
3. **Leading Zeros & Empty String:**
   - `"10200"` with $k=1$ produces `"0200"`. Forgetting `.lstrip('0')` is the most common test failure.
   - If all digits are removed or the string is all zeros (e.g. `"0000"`), return `"0"`, never `""`.