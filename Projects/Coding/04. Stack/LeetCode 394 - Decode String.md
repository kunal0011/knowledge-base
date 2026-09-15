---
date: "2026-08-29"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 394: Decode String"
tags:
  - leetcode
  - coding
  - stack
  - string
  - recursion
  - amazon
  - google
---

# LeetCode 394: Decode String

**Target Companies:** Google, Amazon, Meta, Bloomberg, Microsoft, Cisco  
**Difficulty:** Medium  
**Topic:** Stack / String Parsing / Recursion

---

### Problem Statement

Given an encoded string, return its decoded string.

The encoding rule is: `k[encoded_string]`, where the `encoded_string` inside the square brackets is being repeated exactly `k` times. Note that `k` is guaranteed to be a positive integer.

You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, `k`. For example, there will not be input like `3a` or `2[4]`.

The test cases are generated so that the length of the output will never exceed $10^5$.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $1 \le \text{len}(s) \le 30$.
  - `s` consists of lowercase English letters, digits, `'['`, and `']'`.
  - All integers in `s` are in the range $[1, 300]$.
- **Output:**
  - `str`: The decoded string.
- **Constraints:**
  - Brackets are always properly matched and nested.
  - Length of output string $\le 10^5$.

---

### Key Idea & Intuition

Encoded expressions can be nested arbitrarily, such as `3[a2[c]]` $\to$ `3[acc]` $\to$ `accaccacc`.
Because inner brackets must be evaluated and expanded before outer brackets, this naturally forms a hierarchical context tree, perfectly suited for a **Stack**:

1. **When encountering digits (`'0'`–`'9'`):**
   - Accumulate them into `curr_k = curr_k * 10 + int(digit)` to handle numbers with multiple digits (e.g. `100[a]`).
2. **When encountering `'['`:**
   - A new nested context starts.
   - Push the current prefix string `curr_str` and the repeat multiplier `curr_k` onto the stack: `stack.append((curr_str, curr_k))`.
   - Reset `curr_str = ""` and `curr_k = 0` to start collecting the nested substring.
3. **When encountering `']'`:**
   - The current nested substring is complete.
   - Pop the parent context `(prev_str, k)` from the stack.
   - Expand the current substring: `curr_str = prev_str + curr_str * k`.
4. **When encountering regular letters:**
   - Append to `curr_str`.

---

### Solution Approach (Step-by-Step)

1. Initialize `stack = []` (holding tuples of `(str, int)`), `curr_str = ""`, `curr_k = 0`.
2. Iterate through each character `ch` in `s`:
   - If `ch.isdigit()`:
     - `curr_k = curr_k * 10 + int(ch)`
   - Else if `ch == '['`:
     - Push `(curr_str, curr_k)` onto `stack`.
     - Reset `curr_str = ""`, `curr_k = 0`.
   - Else if `ch == ']'`:
     - Pop `(prev_str, k)` from `stack`.
     - `curr_str = prev_str + curr_str * k`.
   - Else:
     - `curr_str += ch`
3. Return `curr_str`.

---

### Visual Algorithm Walkthrough

Trace $s = \text{"3[a2[c]]"}$:

```
Initial: stack = [], curr_str = "", curr_k = 0

Read '3': curr_k = 3
Read '[': Push ("", 3) to stack.
          Stack: [("", 3)], curr_str = "", curr_k = 0

Read 'a': curr_str = "a"
Read '2': curr_k = 2
Read '[': Push ("a", 2) to stack.
          Stack: [("", 3), ("a", 2)], curr_str = "", curr_k = 0

Read 'c': curr_str = "c"
Read ']': Close innermost bracket!
          Pop ("a", 2).
          curr_str = "a" + ("c" * 2) = "acc"
          Stack: [("", 3)]

Read ']': Close outermost bracket!
          Pop ("", 3).
          curr_str = "" + ("acc" * 3) = "accaccacc"
          Stack: []

End of string. Final result = "accaccacc"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Nested Encodings

- **Input:** `s = "3[a2[c]]"`
- **Output:** `"accaccacc"`

#### Example 2: Adjacent Encodings

- **Input:** `s = "2[abc]3[cd]ef"`
- **Step Tracing:**
  - `2[abc]` expands to `"abcabc"`
  - `3[cd]` expands to `"cdcdcd"`
  - `"ef"` appends directly
- **Output:** `"abcabccdcdcdef"`

#### Example 3: Multi-digit Counts

- **Input:** `s = "10[a]"`
- **Output:** `"aaaaaaaaaa"` (10 'a's)

---

### Multi-Language Implementations

#### Python 3

```python
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []  # stores tuples of (previous_str, multiplier_k)
        curr_str = ""
        curr_k = 0

        for ch in s:
            if ch.isdigit():
                curr_k = curr_k * 10 + int(ch)
            elif ch == '[':
                stack.append((curr_str, curr_k))
                curr_str = ""
                curr_k = 0
            elif ch == ']':
                prev_str, k = stack.pop()
                curr_str = prev_str + curr_str * k
            else:
                curr_str += ch

        return curr_str
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <cctype>

class Solution {
public:
    std::string decodeString(const std::string& s) {
        std::vector<std::string> str_stack;
        std::vector<int> count_stack;

        std::string curr_str = "";
        int curr_k = 0;

        for (char ch : s) {
            if (std::isdigit(ch)) {
                curr_k = curr_k * 10 + (ch - '0');
            } else if (ch == '[') {
                str_stack.push_back(curr_str);
                count_stack.push_back(curr_k);
                curr_str = "";
                curr_k = 0;
            } else if (ch == ']') {
                std::string prev_str = str_stack.back();
                str_stack.pop_back();
                int k = count_stack.back();
                count_stack.pop_back();

                std::string expanded = "";
                expanded.reserve(prev_str.size() + curr_str.size() * k);
                expanded += prev_str;
                for (int i = 0; i < k; ++i) {
                    expanded += curr_str;
                }
                curr_str = std::move(expanded);
            } else {
                curr_str.push_back(ch);
            }
        }

        return curr_str;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public String decodeString(String s) {
        Deque<StringBuilder> strStack = new ArrayDeque<>();
        Deque<Integer> countStack = new ArrayDeque<>();

        StringBuilder currStr = new StringBuilder();
        int currK = 0;

        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);

            if (Character.isDigit(ch)) {
                currK = currK * 10 + (ch - '0');
            } else if (ch == '[') {
                strStack.push(currStr);
                countStack.push(currK);
                currStr = new StringBuilder();
                currK = 0;
            } else if (ch == ']') {
                StringBuilder prevStr = strStack.pop();
                int k = countStack.pop();

                for (int count = 0; count < k; count++) {
                    prevStr.append(currStr);
                }
                currStr = prevStr;
            } else {
                currStr.append(ch);
            }
        }

        return currStr.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(L)$
  - Where $L$ is the length of the final decoded string ($L \le 10^5$).
  - Every character of the output string is generated and concatenated through string additions.
- **Space Complexity:** $\mathcal{O}(L + D)$
  - Where $D$ is the maximum depth of nested brackets (stack frames).
  - The stack stores intermediate strings that total at most the length of the output string.

---

### Takeaway Pattern & Interview Traps

1. **Multi-Digit Multipliers:**
   - Multipliers can exceed $9$ (e.g., `100[leetcode]`). Always parse iteratively using `curr_k = curr_k * 10 + digit`.
2. **Context Preservation:**
   - When entering `'['`, always save both `curr_str` (the prefix built before this bracket) and `curr_k`. Popping on `']'` restores the outer world seamlessly.
3. **Memory Optimization in Java / C++:**
   - Using `StringBuilder` or reserving memory on `std::string` prevents quadratic repeated copies during string concatenation.