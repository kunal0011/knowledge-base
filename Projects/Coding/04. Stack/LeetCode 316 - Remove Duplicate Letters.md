---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 316: Remove Duplicate Letters"
tags:
  - leetcode
  - coding
  - stack
  - monotonic-stack
  - greedy
  - string
  - amazon
  - google
---

# LeetCode 316: Remove Duplicate Letters

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Monotonic Stack / Greedy / String Processing

---

### Problem Statement

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the **smallest in lexicographical order** among all possible results.

*(Note: This problem is identical to LeetCode 1081: Smallest Subsequence of Distinct Characters.)*

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $1 \le \text{len}(s) \le 10^4$.
  - `s` consists of lowercase English letters (`'a'` through `'z'`).
- **Output:**
  - `str`: The lexicographically smallest subsequence containing every distinct character of `s` exactly once.
- **Constraints:**
  - Every unique character in `s` must appear exactly once in the output string.
  - The relative order must form a valid subsequence of `s`.

---

### Key Idea & Intuition

To make the resulting string **lexicographically smallest**, we want smaller characters (like `'a'`, `'b'`) to appear as early as possible.
However, we have two hard constraints:
1. Every unique character must appear **at least and at most once**.
2. Characters can only be selected in the left-to-right order they appear in `s`.

#### The Greedy Choice with Reversibility: Monotonic Stack
Suppose our current prefix in the stack ends with `'c'`, and the incoming character is `'a'`:
- `'a'` is smaller than `'c'`, so having `'a'` precede `'c'` would make the string lexicographically smaller.
- Can we discard `'c'` from the stack right now?
  - **Yes**, IF `'c'` appears again somewhere later in the string (`last_occurrence['c'] > current_index`). We can safely pop `'c'` now, knowing we can pick it up later!
  - **No**, IF `'c'` never appears again in the remainder of the string. We are forced to keep `'c'` to satisfy the requirement that all unique letters must be present.

Hence, we maintain:
1. **`last_occurrence` array / map**: records the highest index where each character appears in `s`.
2. **`in_stack` boolean array / set**: tracks which characters are already selected in the stack. If character `ch` is already in the stack, we **skip** it because its earlier placement has already been greedily optimized.
3. **Monotonic Stack**: while `stack` is not empty, `stack.top() > ch`, and `last_occurrence[stack.top()] > i`, pop `stack.top()` and unmark it from `in_stack`.

---

### Solution Approach (Step-by-Step)

1. **Precompute Last Indices:**
   - Scan $s$ once to map each character to its rightmost index: `last_idx[c] = i`.
2. **Iterate Through Characters:**
   - For each index $i$ and character $c = s[i]$:
     - If $c$ is already in `in_stack`, continue to next character.
     - While `stack` is not empty AND `stack[-1] > c` AND `last_idx[stack[-1]] > i`:
       - `removed = stack.pop()`
       - `in_stack.remove(removed)`
     - `stack.append(c)`
     - `in_stack.add(c)`
3. **Assemble Result:**
   - Join all characters in `stack` from bottom to top to produce the final string.

---

### Visual Algorithm Walkthrough

Let $s = \text{"cbacdcbc"}$:
- Unique characters: `{a, b, c, d}`
- Last occurrences:
  - `c`: index 7
  - `b`: index 6
  - `a`: index 2
  - `d`: index 4

```
Index:    0   1   2   3   4   5   6   7
Char:     c   b   a   c   d   c   b   c

i = 0 ('c'):
  Stack is empty -> push 'c'.
  Stack: ['c'], in_stack: {c}

i = 1 ('b'):
  'b' < 'c' AND last_idx['c'] (7) > 1 -> Pop 'c'!
  Push 'b'.
  Stack: ['b'], in_stack: {b}

i = 2 ('a'):
  'a' < 'b' AND last_idx['b'] (6) > 2 -> Pop 'b'!
  Push 'a'.
  Stack: ['a'], in_stack: {a}

i = 3 ('c'):
  'c' > 'a' -> Push 'c'.
  Stack: ['a', 'c'], in_stack: {a, c}

i = 4 ('d'):
  'd' > 'c' -> Push 'd'.
  Stack: ['a', 'c', 'd'], in_stack: {a, c, d}

i = 5 ('c'):
  'c' already in in_stack -> SKIP!

i = 6 ('b'):
  'b' < 'd', BUT last_idx['d'] (4) < 6 (cannot appear again)!
  Stop popping!
  Push 'b'.
  Stack: ['a', 'c', 'd', 'b'], in_stack: {a, c, d, b}

i = 7 ('c'):
  'c' already in in_stack -> SKIP!

Final Output: "acdb"
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Reverse Alphabetical String with Duplicates

- **Input:** `s = "bcabc"`
- **Step Tracing:**
  - `i=0 ('b')`: stack = `['b']`
  - `i=1 ('c')`: stack = `['b', 'c']`
  - `i=2 ('a')`: `'a' < 'c'` and `'c'` appears at index 4 $\to$ pop `'c'`. `'a' < 'b'` and `'b'` appears at index 3 $\to$ pop `'b'`. Push `'a'`. stack = `['a']`.
  - `i=3 ('b')`: stack = `['a', 'b']`
  - `i=4 ('c')`: stack = `['a', 'b', 'c']`
- **Output:** `"abc"`

#### Example 2: Non-Removable Middle Characters

- **Input:** `s = "cbacdcbc"`
- **Output:** `"acdb"` (Demonstrated in walkthrough above).

#### Example 3: Already Lexicographically Sorted

- **Input:** `s = "abcd"`
- **Output:** `"abcd"` (No pops occur).

---

### Multi-Language Implementations

#### Python 3

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # Step 1: Precompute the last occurrence of each character
        last_idx = {ch: i for i, ch in enumerate(s)}
        in_stack = set()
        stack = []

        # Step 2: Monotonic stack traversal
        for i, ch in enumerate(s):
            if ch in in_stack:
                continue

            # Pop characters that are lexicographically greater than ch
            # AND still appear later in the string
            while stack and stack[-1] > ch and last_idx[stack[-1]] > i:
                popped = stack.pop()
                in_stack.remove(popped)

            stack.append(ch)
            in_stack.add(ch)

        return "".join(stack)
```

#### C++17

```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::string removeDuplicateLetters(const std::string& s) {
        std::vector<int> last_idx(26, 0);
        std::vector<bool> in_stack(26, false);
        int n = static_cast<int>(s.size());

        for (int i = 0; i < n; ++i) {
            last_idx[s[i] - 'a'] = i;
        }

        std::string result; // Use std::string directly as our stack

        for (int i = 0; i < n; ++i) {
            int ch_idx = s[i] - 'a';
            if (in_stack[ch_idx]) continue;

            while (!result.empty() && result.back() > s[i] && last_idx[result.back() - 'a'] > i) {
                in_stack[result.back() - 'a'] = false;
                result.pop_back();
            }

            result.push_back(s[i]);
            in_stack[ch_idx] = true;
        }

        return result;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public String removeDuplicateLetters(String s) {
        int[] lastIdx = new int[26];
        boolean[] inStack = new boolean[26];
        int n = s.length();

        for (int i = 0; i < n; i++) {
            lastIdx[s.charAt(i) - 'a'] = i;
        }

        Deque<Character> stack = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);
            int chIdx = ch - 'a';

            if (inStack[chIdx]) continue;

            while (!stack.isEmpty() && stack.peek() > ch && lastIdx[stack.peek() - 'a'] > i) {
                char removed = stack.pop();
                inStack[removed - 'a'] = false;
            }

            stack.push(ch);
            inStack[chIdx] = true;
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pollLast()); // Poll from bottom to top
        }

        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Scanning $s$ to compute `last_idx` takes $\mathcal{O}(N)$.
  - Each character in $s$ is pushed onto the stack at most once and popped at most once.
  - Overall Time: $\mathcal{O}(N)$, where $N$ is the length of string $s$.
- **Space Complexity:** $\mathcal{O}(1)$ (or $\mathcal{O}(\Sigma)$ where $\Sigma = 26$)
  - The stack and boolean arrays store at most $26$ distinct lowercase English characters.
  - Auxiliary space is strictly bounded by constant $\mathcal{O}(26) = \mathcal{O}(1)$.

---

### Takeaway Pattern & Interview Traps

1. **Monotonic Stack with Future-Looking Condition:**
   - Standard monotonic stacks pop unconditionally when monotonicity is violated. Here, popping is **conditional** on future availability: `last_idx[stack.top()] > i`.
2. **Why Skip if `ch in in_stack`?**
   - If a character is already present in the stack, keeping its earlier position gives a lexicographically superior prefix than discarding it and placing it further to the right.
3. **Direct String Buffer as Stack:**
   - In C++, using `std::string` directly as a stack with `push_back()` and `pop_back()` avoids any secondary string reconstruction and eliminates heap allocations.