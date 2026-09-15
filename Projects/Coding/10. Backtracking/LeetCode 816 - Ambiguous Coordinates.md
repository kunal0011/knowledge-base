---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 816: Ambiguous Coordinates"
tags:
  - leetcode
  - coding
  - backtracking
  - string
  - math
  - amazon
  - google
---

# LeetCode 816: Ambiguous Coordinates

**Target Companies:** Google, Amazon  
**Difficulty:** Medium  
**Topic:** Backtracking / String Splitting / Decimal Validation  

---

### Problem Statement

We had some 2-dimensional coordinates, like `"(1, 3)"` or `"(2, 0.5)"`. Then, we removed all commas, decimal points, and spaces and ended up with the string `s`.

- For example, `"(1, 3)"` becomes `s = "(13)"` and `"(2, 0.5)"` becomes `s = "(205)"`.

Return *a list of strings representing all possibilities for what our original coordinates could have been*.

Our original representation never had extraneous zeroes, so we never started with numbers like `"00"`, `"07"`, `"00.0"`, `"0.0"`, `"0.00"`, or `"1.0"`. A number like `"0"` or `"0.5"` is allowed, but `"0.0"` or `"1.0"` is not.

You may return the answer in **any order**.

---

### Input & Output Formats & Constraints

- **Input:** `s: str` of the format `"([0-9]+)"`
- **Output:** `List[str]` containing formatted valid coordinates `"(x, y)"`
- **Constraints:**
  - $4 \le \text{s.length} \le 12$
  - `s[0] == '('` and `s[s.length - 1] == ')'`
  - The remaining characters in `s` are digits.

---

### Key Idea & Intuition

- **Decomposing the Problem:**
  1. **Coordinate Split:** Strip the outer parentheses `digits = s[1:-1]`. Length $N \le 10$. Partition `digits` into two non-empty substrings: $X = \text{digits}[:i]$ and $Y = \text{digits}[i:]$ for $1 \le i < N$.
  2. **Valid Number Generation:** For each substring $T$, generate all valid numbers that can be formed by inserting an optional decimal point.
  3. **Cartesian Product:** Combine each valid $x \in \text{valid}(X)$ with each $y \in \text{valid}(Y)$ to format `f"({x}, {y})"`.
- **Mathematical Invariants for Valid Number $T$:**
  - **As an Integer (no decimal point):**
    - $T$ cannot have leading zeros unless $T == \text{"0"}$.
    - Valid if: $\text{len}(T) == 1 \lor T[0] \ne \text{'0'}$.
  - **With a Decimal Point ($T[:d] + \text{'.' } + T[d:]$ for $1 \le d < \text{len}(T)$):**
    - **Integer part $T[:d]$:** Cannot have leading zeros unless length is 1 ($\text{len}(T[:d]) == 1 \lor T[0] \ne \text{'0'}$).
    - **Fractional part $T[d:]$:** Cannot have trailing zeros ($T[-1] \ne \text{'0'}$).

---

### Solution Approach (Step-by-Step)

1. Extract core digits `S = s[1:-1]`.
2. Define helper function `get_valid(segment)`:
   - Initialize `valid = []`.
   - Length $m = \text{len}(segment)$.
   - **Integer option (no dot):**
     - If $m == 1$ or $segment[0] \ne \text{'0'}$:
       - Append `segment` to `valid`.
   - **Decimal point options:**
     - For $d$ from $1$ to $m - 1$:
       - `left = segment[:d]`, `right = segment[d:]`.
       - Check `left`: valid if `left == '0'` or `left[0] != '0'`.
       - Check `right`: valid if `right[-1] != '0'`.
       - If both are valid:
         - Append `left + "." + right` to `valid`.
   - Return `valid`.
3. Initialize `results = []`.
4. Loop split index $i$ from $1$ to $\text{len}(S) - 1$:
   - `x_candidates = get_valid(S[:i])`
   - `y_candidates = get_valid(S[i:])`
   - For `x` in `x_candidates`:
     - For `y` in `y_candidates`:
       - Append `f"({x}, {y})"` to `results`.
5. Return `results`.

---

### Visual Algorithm Walkthrough

Let `s = "(123)"`, digits `S = "123"`:

```
Possible Splits:
1. Split at i = 1: X = "1", Y = "23"
   - get_valid("1"):
     - No dot: "1" (valid)
     -> ["1"]
   - get_valid("23"):
     - No dot: "23" (valid)
     - Dot at 1: "2.3" (left="2", right="3" != '0' -> valid)
     -> ["23", "2.3"]
   Cross product:
     "(1, 23)", "(1, 2.3)"

2. Split at i = 2: X = "12", Y = "3"
   - get_valid("12"):
     - No dot: "12"
     - Dot at 1: "1.2"
     -> ["12", "1.2"]
   - get_valid("3"):
     -> ["3"]
   Cross product:
     "(12, 3)", "(1.2, 3)"

Final Results: ["(1, 23)", "(1, 2.3)", "(12, 3)", "(1.2, 3)"]
```

---

### Solved Examples with Multiple Inputs

| Test Case | `s` | Substring Splits | Valid Filtered Outputs |
| :--- | :--- | :--- | :--- |
| **Standard** | `"(123)"` | `("1", "23")`, `("12", "3")` | `["(1, 23)", "(1, 2.3)", "(12, 3)", "(1.2, 3)"]` |
| **With Zeroes** | `"(00011)"` | Splitting around zeros | `["(0, 0.011)", "(0.001, 1)"]` |
| **Double Zero** | `"(0123)"` | Leading zeros in prefix | `["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", ...]` |
| **All Zeros** | `"(00)"` | Both must be `"0"` | `["(0, 0)"]` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        """
        Returns all possible original coordinate representations for string s.
        """
        digits = s[1:-1]
        n = len(digits)
        results: List[str] = []

        def get_valid_numbers(segment: str) -> List[str]:
            m = len(segment)
            valid = []

            # Option 1: Whole integer without decimal point
            if m == 1 or segment[0] != '0':
                valid.append(segment)

            # Option 2: Number with decimal point segment[:d] . segment[d:]
            for d in range(1, m):
                left, right = segment[:d], segment[d:]
                # Left integer part cannot have leading zeros unless it is "0"
                if (left == '0' or left[0] != '0') and right[-1] != '0':
                    valid.append(f"{left}.{right}")

            return valid

        # Partition digits into X and Y coordinates
        for i in range(1, n):
            x_vals = get_valid_numbers(digits[:i])
            y_vals = get_valid_numbers(digits[i:])
            for x in x_vals:
                for y in y_vals:
                    results.append(f"({x}, {y})")

        return results
```

#### C++17
```cpp
#include <string>
#include <vector>

class Solution {
public:
    std::vector<std::string> ambiguousCoordinates(std::string s) {
        std::string digits = s.substr(1, s.size() - 2);
        int n = static_cast<int>(digits.size());
        std::vector<std::string> results;

        for (int i = 1; i < n; ++i) {
            std::vector<std::string> x_vals = getValid(digits.substr(0, i));
            std::vector<std::string> y_vals = getValid(digits.substr(i));

            for (const std::string& x : x_vals) {
                for (const std::string& y : y_vals) {
                    results.push_back("(" + x + ", " + y + ")");
                }
            }
        }

        return results;
    }

private:
    std::vector<std::string> getValid(const std::string& segment) {
        int m = static_cast<int>(segment.size());
        std::vector<std::string> valid;

        // Option 1: No decimal point
        if (m == 1 || segment[0] != '0') {
            valid.push_back(segment);
        }

        // Option 2: With decimal point
        for (int d = 1; d < m; ++d) {
            std::string left = segment.substr(0, d);
            std::string right = segment.substr(d);

            if ((left == "0" || left[0] != '0') && right.back() != '0') {
                valid.push_back(left + "." + right);
            }
        }

        return valid;
    }
};
```

#### Java
```java
import java.util.ArrayList;
import java.util.List;

class Solution {
    public List<String> ambiguousCoordinates(String s) {
        String digits = s.substring(1, s.length() - 1);
        int n = digits.length();
        List<String> results = new ArrayList<>();

        for (int i = 1; i < n; i++) {
            List<String> xVals = getValid(digits.substring(0, i));
            List<String> yVals = getValid(digits.substring(i));

            for (String x : xVals) {
                for (String y : yVals) {
                    results.add("(" + x + ", " + y + ")");
                }
            }
        }

        return results;
    }

    private List<String> getValid(String segment) {
        int m = segment.length();
        List<String> valid = new ArrayList<>();

        // No decimal point
        if (m == 1 || segment.charAt(0) != '0') {
            valid.add(segment);
        }

        // With decimal point
        for (int d = 1; d < m; d++) {
            String left = segment.substring(0, d);
            String right = segment.substring(d);

            if ((left.equals("0") || left.charAt(0) != '0') && right.charAt(right.length() - 1) != '0') {
                valid.add(left + "." + right);
            }
        }

        return valid;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N^3)$ where $N = \text{digits.length} \le 10$.
  - Splitting $S$ into two parts takes $\mathcal{O}(N)$ iterations.
  - Generating valid numbers for each segment requires testing up to $N$ decimal positions, with substring creations of length $\le N$ taking $\mathcal{O}(N^2)$.
  - Cartesian product produces at most $\mathcal{O}(N^2)$ coordinate pairs of length $\mathcal{O}(N)$.
  - Since $N \le 10$, $10^3 = 1000$ operations $\implies < 1 \text{ ms}$.
- **Space Complexity:** $\mathcal{O}(N^3)$ to store all output formatted coordinate strings.

---

### Takeaway Pattern & Interview Traps

- **Decomposing Coordinate Splitting:** Breaking the problem into two orthogonal subproblems—splitting $S \to (X, Y)$ and then independently finding valid numbers for a single string—makes what could be a messy recursion trivial.
- **Decimal Invariant Traps:** Remember both constraints:
  - Integer part: cannot start with `'0'` unless length is 1 (`"0"` is ok, `"05"` is invalid).
  - Fractional part: cannot end with `'0'` (`"1.50"` is invalid, `"1.05"` is ok).