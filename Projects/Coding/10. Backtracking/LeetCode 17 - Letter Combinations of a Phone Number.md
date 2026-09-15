---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 17: Letter Combinations of a Phone Number"
tags:
  - leetcode
  - coding
  - backtracking
  - hash-table
  - string
  - amazon
  - google
---

# LeetCode 17: Letter Combinations of a Phone Number

**Target Companies:** Amazon, Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Backtracking / Cartesian Product Tree Traversal  

---

### Problem Statement

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

```
2 -> "abc"     3 -> "def"
4 -> "ghi"     5 -> "jkl"     6 -> "mno"
7 -> "pqrs"    8 -> "tuv"     9 -> "wxyz"
```

---

### Input & Output Formats & Constraints

- **Input:** `digits: str`
- **Output:** `List[str]`
- **Constraints:**
  - $0 \le \text{digits.length} \le 4$
  - `digits[i]` is a digit in the range `['2', '9']`.

---

### Key Idea & Intuition

- **Cartesian Product of Character Sets:**
  - Each digit in `digits` corresponds to an alphabet set of size 3 (or 4 for digits `'7'` and `'9'`).
  - Generating all possible combinations is equivalent to finding the Cartesian product:
    $$\text{Keys}[\text{digit}_0] \times \text{Keys}[\text{digit}_1] \times \dots \times \text{Keys}[\text{digit}_{N-1}]$$
- **Fixed-Depth Backtracking:**
  - The recursion depth is exactly $N = \text{len(digits)}$.
  - At depth `idx`, we iterate over all candidate letters mapped to `digits[idx]`:
    1. Append letter to current buffer.
    2. Recurse to `idx + 1`.
    3. Pop letter (backtrack).
  - When `idx == len(digits)`, the current buffer contains a complete word of length $N$; append it to our results.
- **Edge Case (Empty Input):**
  - If `digits == ""`, the problem requires returning `[]`, NOT `[""]`.

---

### Solution Approach (Step-by-Step)

1. If `len(digits) == 0`, immediately return `[]`.
2. Map digits `'2'` through `'9'` to their corresponding strings.
3. Initialize `results = []` and a mutable character buffer `path = []`.
4. Define `backtrack(index)`:
   - If `index == len(digits)`:
     - Append `"".join(path)` to `results`.
     - Return.
   - For each character `ch` in `MAPPING[digits[index]]`:
     - `path.append(ch)`
     - `backtrack(index + 1)`
     - `path.pop()` (backtrack)
5. Call `backtrack(0)` and return `results`.

---

### Visual Algorithm Walkthrough

For `digits = "23"`:
- Digit `'2'` $\implies$ `'a', 'b', 'c'`
- Digit `'3'` $\implies$ `'d', 'e', 'f'`

```
                          backtrack(0, "")
                  /               |               \
             Pick 'a'          Pick 'b'          Pick 'c'
            backtrack(1)      backtrack(1)      backtrack(1)
           /   |   \         /   |   \         /   |   \
         'd'  'e'  'f'     'd'  'e'  'f'     'd'  'e'  'f'
          |    |    |       |    |    |       |    |    |
        "ad" "ae" "af"    "bd" "be" "bf"    "cd" "ce" "cf"
```

Each leaf at depth 2 contributes a valid 2-letter combination to the output.

---

### Solved Examples with Multiple Inputs

| Test Case | `digits` | Key Mappings | Valid Combinations | Output Count |
| :--- | :--- | :--- | :--- | :--- |
| **Standard** | `"23"` | `2: abc`, `3: def` | `"ad","ae","af","bd","be","bf","cd","ce","cf"` | `9` |
| **Single Digit** | `"2"` | `2: abc` | `"a","b","c"` | `3` |
| **Four Digits (4-letter keys)** | `"79"` | `7: pqrs`, `9: wxyz` | Cartesian product $4 \times 4$ | `16` |
| **Empty Input** | `""` | None | None | `0` (`[]`) |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        """
        Returns all letter combinations for the given phone digits.
        """
        if not digits:
            return []

        digit_map = {
            '2': "abc", '3': "def", '4': "ghi", '5': "jkl",
            '6': "mno", '7': "pqrs", '8': "tuv", '9': "wxyz"
        }

        results: List[str] = []
        path: List[str] = []

        def backtrack(idx: int) -> None:
            if idx == len(digits):
                results.append("".join(path))
                return

            letters = digit_map[digits[idx]]
            for char in letters:
                path.append(char)
                backtrack(idx + 1)
                path.pop()  # Backtrack

        backtrack(0)
        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::string> letterCombinations(const std::string& digits) {
        if (digits.empty()) return {};

        const std::vector<std::string> digit_map = {
            "",     "",     "abc",  "def", // 0, 1, 2, 3
            "ghi",  "jkl",  "mno",         // 4, 5, 6
            "pqrs", "tuv",  "wxyz"         // 7, 8, 9
        };

        std::vector<std::string> results;
        std::string path;
        backtrack(0, digits, digit_map, path, results);
        return results;
    }

private:
    void backtrack(int idx, const std::string& digits, const std::vector<std::string>& digit_map,
                   std::string& path, std::vector<std::string>& results) {
        if (idx == static_cast<int>(digits.size())) {
            results.push_back(path);
            return;
        }

        const std::string& letters = digit_map[digits[idx] - '0'];
        for (char c : letters) {
            path.push_back(c);
            backtrack(idx + 1, digits, digit_map, path, results);
            path.pop_back(); // Backtrack
        }
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    private static final String[] DIGIT_MAP = {
        "",     "",     "abc",  "def", // 0, 1, 2, 3
        "ghi",  "jkl",  "mno",         // 4, 5, 6
        "pqrs", "tuv",  "wxyz"         // 7, 8, 9
    };

    public List<String> letterCombinations(String digits) {
        List<String> results = new ArrayList<>();
        if (digits == null || digits.isEmpty()) {
            return results;
        }

        StringBuilder path = new StringBuilder();
        backtrack(0, digits, path, results);
        return results;
    }

    private void backtrack(int idx, String digits, StringBuilder path, List<String> results) {
        if (idx == digits.length()) {
            results.add(path.toString());
            return;
        }

        String letters = DIGIT_MAP[digits.charAt(idx) - '0'];
        for (int i = 0; i < letters.length(); i++) {
            path.append(letters.charAt(i));
            backtrack(idx + 1, digits, path, results);
            path.deleteCharAt(path.length() - 1); // Backtrack
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(4^N \cdot N)$ where $N$ is the number of digits ($0 \le N \le 4$).
  - For $N$ digits, each digit branches into at most 4 letters ($3^a \times 4^b \le 4^N$).
  - For each of the $4^N$ combinations, copying the string of length $N$ takes $\mathcal{O}(N)$ time.
  - Since $N \le 4$, $4^4 \times 4 = 256 \times 4 = 1024$ operations $\implies < 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space for the recursion call stack and current string buffer `path`.

---

### Takeaway Pattern & Interview Traps

- **Empty String Pitfall:** Returning `[""]` instead of `[]` for `digits == ""` is the most common bug in this problem. Always check `if not digits: return []` right at the start.
- **Buffer Backtracking vs String Copying:** In C++ and Java, appending and popping from a mutable buffer (`std::string` or `StringBuilder`) avoids allocating new intermediate string objects on every recursive call.