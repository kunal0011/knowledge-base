---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 32: Longest Valid Parentheses"
tags:
  - leetcode
  - coding
  - stack
  - dynamic-programming
  - string
  - amazon
  - google
---

# LeetCode 32: Longest Valid Parentheses

**Target Companies:** Amazon, Google, Meta, Microsoft, ByteDance, Bloomberg  
**Difficulty:** Hard  
**Topic:** Stack / String / Two-Pass Counter / Dynamic Programming

---

### Problem Statement

Given a string containing just the characters `'('` and `')'`, return *the length of the longest valid (well-formed) parentheses substring*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $0 \le \text{len}(s) \le 3 \times 10^4$. `s[i]` is either `'('` or `')'`.
- **Output:**
  - `int`: The maximum length of a contiguous substring of well-formed parentheses.
- **Constraints:**
  - $0 \le s.\text{length} \le 3 \times 10^4$.
  - An empty string returns `0`.

---

### Key Idea & Intuition

A valid parentheses substring must satisfy two invariants:
1. At no prefix does the count of `')'` exceed the count of `'('`.
2. Total count of `'('` equals total count of `')'`.

The challenge is finding the longest **contiguous** valid slice.

#### Approach 1: Stack with Index Boundary Sentinel ($\mathcal{O}(N)$ Time, $\mathcal{O}(N)$ Space)
Instead of pushing characters, push **indices** of characters onto the stack:
- Push `-1` initially onto the stack. This acts as the anchor/base index before the start of any valid substring.
- When encountering `'('`:
  - Push its index $i$ onto the stack.
- When encountering `')'`:
  - Pop the top index from the stack.
  - If the stack is now empty: this `')'` is unmatched and acts as a boundary barrier. Push its index $i$ as the **new base/anchor**.
  - If the stack is not empty: the current valid substring extends from `stack.top()` to $i$. Its length is $i - \text{stack.top()}$. We update `max_len = max(max_len, i - stack.top())`.

#### Approach 2: Two-Pass Counter ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space - Optimal Space)
We can eliminate the stack entirely using two passes:
1. **Left-to-Right Pass:**
   - Maintain `left` and `right` counters.
   - For each character: if `'('`, increment `left`; if `')'`, increment `right`.
   - If `left == right`: update `max_len = max(max_len, 2 * right)`.
   - If `right > left`: invalid closing parenthesis encountered; reset `left = right = 0`.
2. **Right-to-Left Pass:**
   - The forward pass misses substrings where `left > right` (e.g. `"(()"` where `left=2, right=1`).
   - A reverse pass from right-to-left symmetrically handles this: if `left > right`, reset `left = right = 0`.

---

### Solution Approach (Step-by-Step)

#### Algorithm 1: Stack Approach
1. Initialize `stack = [-1]`, `max_len = 0`.
2. For each index $i$ from $0$ to $N - 1$:
   - If $s[i] == \text{'('}$: push $i$.
   - If $s[i] == \text{')'}$:
     - `stack.pop()`
     - If `len(stack) == 0`:
       - `stack.append(i)` (reset base)
     - Else:
       - `max_len = max(max_len, i - stack[-1])`
3. Return `max_len`.

#### Algorithm 2: Two-Pass Counter ($\mathcal{O}(1)$ Extra Space)
1. Initialize `left = 0, right = 0, max_len = 0`.
2. Scan left to right:
   - Increment `left` or `right`.
   - If `left == right`: `max_len = max(max_len, 2 * right)`.
   - If `right > left`: `left = right = 0`.
3. Reset `left = right = 0`.
4. Scan right to left:
   - Increment `left` or `right`.
   - If `left == right`: `max_len = max(max_len, 2 * left)`.
   - If `left > right`: `left = right = 0`.
5. Return `max_len`.

---

### Visual Algorithm Walkthrough

Let $s = \text{")()())"}$:

```
Initial: stack = [-1], max_len = 0

i = 0, char = ')'
  pop() -> stack empty -> push 0 as new base!
  Stack: [0]

i = 1, char = '('
  push 1
  Stack: [0, 1]

i = 2, char = ')'
  pop() -> popped 1, stack top is 0.
  Valid length: 2 - 0 = 2.
  max_len = max(0, 2) = 2.
  Stack: [0]

i = 3, char = '('
  push 3
  Stack: [0, 3]

i = 4, char = ')'
  pop() -> popped 3, stack top is 0.
  Valid length: 4 - 0 = 4.
  max_len = max(2, 4) = 4.
  Stack: [0]

i = 5, char = ')'
  pop() -> stack empty -> push 5 as new base!
  Stack: [5]

Result: max_len = 4 (Substring "()()")
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Mixed String

- **Input:** `s = ")()())"`
- **Execution:** Valid length 4 from index 1 to 4 (`"()()"`).
- **Output:** `4`

#### Example 2: Incomplete Opening String

- **Input:** `s = "(()"`
- **Stack Trace:**
  - Start: `[-1]`
  - $i=0$ `'('`: `[-1, 0]`
  - $i=1$ `'('`: `[-1, 0, 1]`
  - $i=2$ `')'`: pop `1`. Stack top is `0`. Valid length = $2 - 0 = 2$.
- **Output:** `2`

#### Example 3: Empty String

- **Input:** `s = ""`
- **Output:** `0`

---

### Multi-Language Implementations

#### Python 3

##### Stack Approach ($\mathcal{O}(N)$ Space)
```python
from typing import List

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack: List[int] = [-1]  # Sentinel base index
        max_len = 0

        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    # New base boundary after invalid ')'
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])

        return max_len
```

##### Optimal $\mathcal{O}(1)$ Space Approach (Two-Pass Counter)
```python
class SolutionTwoPass:
    def longestValidParentheses(self, s: str) -> int:
        left = right = max_len = 0

        # Left to right pass
        for ch in s:
            if ch == '(':
                left += 1
            else:
                right += 1

            if left == right:
                max_len = max(max_len, 2 * right)
            elif right > left:
                left = right = 0

        # Right to left pass
        left = right = 0
        for ch in reversed(s):
            if ch == '(':
                left += 1
            else:
                right += 1

            if left == right:
                max_len = max(max_len, 2 * left)
            elif left > right:
                left = right = 0

        return max_len
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int longestValidParentheses(const std::string& s) {
        // Two-Pass O(1) space optimal method
        int left = 0, right = 0, max_len = 0;
        int n = static_cast<int>(s.size());

        // Pass 1: Left to right
        for (int i = 0; i < n; ++i) {
            if (s[i] == '(') left++;
            else right++;

            if (left == right) {
                max_len = std::max(max_len, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }

        // Pass 2: Right to left
        left = right = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] == '(') left++;
            else right++;

            if (left == right) {
                max_len = std::max(max_len, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }

        return max_len;
    }
};
```

#### Java

```java
public class Solution {
    public int longestValidParentheses(String s) {
        int left = 0, right = 0, maxLen = 0;
        int n = s.length();

        // Pass 1: Left to right
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }

            if (left == right) {
                maxLen = Math.max(maxLen, 2 * right);
            } else if (right > left) {
                left = right = 0;
            }
        }

        // Pass 2: Right to left
        left = right = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }

            if (left == right) {
                maxLen = Math.max(maxLen, 2 * left);
            } else if (left > right) {
                left = right = 0;
            }
        }

        return maxLen;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Stack Approach: Pushes and pops each index at most once $\implies \mathcal{O}(N)$ time.
  - Two-Pass Counter Approach: Two linear scans through the string of length $N \implies 2 \times \mathcal{O}(N) = \mathcal{O}(N)$ time.
- **Space Complexity:**
  - Stack Approach: $\mathcal{O}(N)$ to store indices in the stack.
  - Two-Pass Counter Approach: $\mathcal{O}(1)$ auxiliary space using only primitive scalar counters.

---

### Takeaway Pattern & Interview Traps

1. **The Base Sentinel Index (`-1`):**
   - Without the initial `-1`, when the stack becomes empty after matching an opening parenthesis at index $0$ with a closing parenthesis at index $1$ (e.g. `"()"`), there is no previous boundary index to subtract from. `-1` makes the length calculation $1 - (-1) = 2$ seamless.
2. **Why Single Left-to-Right Counter Fails:**
   - Consider `s = "(()"`. In the forward pass, `left` reaches $2$ and `right` reaches $1$. `left == right` is never triggered, so `max_len` would wrongly stay `0`. The reverse right-to-left pass catches this symmetric case without any extra memory.