---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 316: Remove Duplicate Letters"
tags:
  - leetcode
  - coding
  - greedy
  - stack
  - monotonic-stack
  - string
  - amazon
  - google
---

# LeetCode 316: Remove Duplicate Letters

**Target Companies:** Google, Amazon, Meta, Microsoft, Apple, ByteDance  
**Difficulty:** Medium  
**Topic:** Greedy / Monotonic Stack / String  

---

### Problem Statement

Given a string `s`, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the **smallest in lexicographical order** among all possible results.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str` / `string` ($1 \le |s| \le 10^4$).
- **Output:**
  - `str` / `string` — the lexicographically smallest subsequence containing every distinct character of `s` exactly once.
- **Constraints:**
  - $1 \le \text{s.length} \le 10^4$
  - `s` consists of lowercase English letters.
  *(Note: This problem is completely identical to LeetCode 1081: Smallest Subsequence of Distinct Characters).*

---

### Key Idea & Intuition

To minimize the resulting string lexicographically:
1. We want smaller characters to appear as early as possible (i.e. we prefer `'a'` before `'b'`, `'b'` before `'c'`).
2. However, every character must appear **at least once** and **at most once**.
3. Therefore, if the current character $c$ is smaller than the character on top of our stack, we can only safely pop the top character if that character **appears again later** in the remaining part of string $s$.
4. If the character on top of our stack does **not** appear again later, we are strictly forbidden from popping it, because popping it would mean it never appears in our final result!

#### Monotonic Stack with Last Occurrence Lookup:
- Precompute the **last occurrence index** of each character in `s`: `last_pos[c]`.
- Maintain a stack and a `seen` set (or boolean array of size 26):
  - If character $c$ is already in `seen`, skip it (an earlier placement of $c$ was already optimal).
  - While `stack` is not empty, `stack[-1] > c`, and `last_pos[stack[-1]] > i`:
    - The top character is larger than $c$, and we know it will appear again at index $> i$.
    - Pop it from `stack` and remove it from `seen`.
  - Push $c$ onto `stack` and add $c$ to `seen`.
- The characters remaining in the stack form the optimal answer.

---

### Solution Approach (Step-by-Step)

1. Compute `last_pos = {c: i for i, c in enumerate(s)}`.
2. Initialize an empty stack `stack = []` and a set `seen = set()`.
3. For index `i` and character `c` in `enumerate(s)`:
   - If `c` is in `seen`: continue.
   - While `stack` and `stack[-1] > c` and `last_pos[stack[-1]] > i`:
     - `popped = stack.pop()`
     - `seen.remove(popped)`
   - `stack.append(c)`
   - `seen.add(c)`
4. Return `"".join(stack)`.

---

### Visual Algorithm Walkthrough

For `s = "cbacdcbc"`:
Last occurrences:
`'a': 2, 'b': 7, 'c': 6, 'd': 4`

```
i=0 ('c'): stack = ['c'], seen = {'c'}
i=1 ('b'):
  'b' < 'c' and last_pos['c'] (6) > 1 -> Pop 'c'
  stack = ['b'], seen = {'b'}
i=2 ('a'):
  'a' < 'b' and last_pos['b'] (7) > 2 -> Pop 'b'
  stack = ['a'], seen = {'a'}
i=3 ('c'):
  'c' > 'a' -> push 'c'
  stack = ['a', 'c'], seen = {'a', 'c'}
i=4 ('d'):
  'd' > 'c' -> push 'd'
  stack = ['a', 'c', 'd'], seen = {'a', 'c', 'd'}
i=5 ('c'):
  'c' is already in seen -> SKIP!
i=6 ('b'):
  'b' < 'd' and last_pos['d'] (4) < 6: Cannot pop 'd'!
  Push 'b'
  stack = ['a', 'c', 'd', 'b'], seen = {'a', 'c', 'd', 'b'}
i=7 ('c'):
  'c' is already in seen -> SKIP!

Result = "acdb".
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `s = "bcabc"`
- **Tracing:**
  - Push 'b'
  - Push 'c'
  - 'a' arrives: pops 'c' (c appears at 4), pops 'b' (b appears at 3), pushes 'a'
  - Pushes 'b', pushes 'c'
- **Output:** `"abc"`

#### Example 2:
- **Input:** `s = "cbacdcbc"`
- **Output:** `"acdb"`

#### Example 3 (Already Sorted):
- **Input:** `s = "abc"`
- **Output:** `"abc"`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        last_pos = {c: i for i, c in enumerate(s)}
        stack = []
        seen = set()
        
        for i, c in enumerate(s):
            if c in seen:
                continue
                
            # Pop larger elements if they appear again in the future
            while stack and stack[-1] > c and last_pos[stack[-1]] > i:
                seen.remove(stack.pop())
                
            stack.append(c)
            seen.add(c)
            
        return "".join(stack)
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::string removeDuplicateLetters(const std::string& s) {
        std::vector<int> last_pos(26, 0);
        int n = static_cast<int>(s.size());
        for (int i = 0; i < n; ++i) {
            last_pos[s[i] - 'a'] = i;
        }
        
        std::vector<bool> in_stack(26, false);
        std::string result;
        
        for (int i = 0; i < n; ++i) {
            char c = s[i];
            int idx = c - 'a';
            
            if (in_stack[idx]) continue;
            
            while (!result.empty() && result.back() > c && last_pos[result.back() - 'a'] > i) {
                in_stack[result.back() - 'a'] = false;
                result.pop_back();
            }
            
            result.push_back(c);
            in_stack[idx] = true;
        }
        
        return result;
    }
};
```

#### Java 17
```java
import java.util.ArrayDeque;
import java.util.Deque;

class Solution {
    public String removeDuplicateLetters(String s) {
        int[] lastPos = new int[26];
        int n = s.length();
        for (int i = 0; i < n; i++) {
            lastPos[s.charAt(i) - 'a'] = i;
        }
        
        boolean[] inStack = new boolean[26];
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            int idx = c - 'a';
            
            if (inStack[idx]) continue;
            
            while (sb.length() > 0 && 
                   sb.charAt(sb.length() - 1) > c && 
                   lastPos[sb.charAt(sb.length() - 1) - 'a'] > i) {
                inStack[sb.charAt(sb.length() - 1) - 'a'] = false;
                sb.deleteCharAt(sb.length() - 1);
            }
            
            sb.append(c);
            inStack[idx] = true;
        }
        
        return sb.toString();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$
  - Building `last_pos` takes $\mathcal{O}(n)$ time.
  - Each character enters the stack at most once and is popped at most once.
  - Character lookups and comparisons are $\mathcal{O}(1)$.
  - Total time: $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The stack and `seen` array contain at most 26 lowercase English characters.

---

### Takeaway Pattern & Interview Traps

- **Future Existence Invariant:** In greedy monotonic stack problems where every character must be preserved, never pop an element unless its last occurrence index exceeds the current index (`last_pos[top] > i`).
- **Skip Already Present Characters:** Always check `if c in seen: continue`. If $c$ is already in the stack, putting another copy or evaluating it again can only make the prefix lexicographically worse or violate the distinct letters requirement.