---
date: "2026-09-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 91: Decode Ways"
tags:
  - leetcode
  - coding
  - dynamic-programming
  - string
  - meta
  - google
  - amazon
  - microsoft
  - bloomberg
---

# LeetCode 91: Decode Ways

**Target Companies:** Meta, Google, Amazon, Microsoft, Bloomberg, Apple, Uber  
**Difficulty:** Medium  
**Topic:** Dynamic Programming / String Partitioning

---

### Problem Statement

A message containing letters from `A-Z` can be encoded into numbers using the following mapping:
- `'A' -> "1"`
- `'B' -> "2"`
- ...
- `'Z' -> "26"`

To decode an encoded message, all the digits must be grouped and then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, `"11106"` can be mapped into:
- `"AAJF"` with the grouping `(1, 1, 10, 6)`
- `"KJF"` with the grouping `(11, 10, 6)`

Note that the grouping `(1, 11, 06)` is invalid because `"06"` cannot be mapped into `'F'` since `"6"` is different from `"06"`.

Given a string `s` containing only digits, return the **number of ways** to decode it.

The test cases are generated so that the answer fits in a **32-bit** integer.

---

### Input & Output Formats & Constraints

- **Input:** `s: str`
- **Output:** `int`
- **Constraints:**
  - `1 <= s.length <= 100`
  - `s` consists of digits and may contain leading zero(s).
  - The answer fits in a 32-bit signed integer (`<= 2^31 - 1`).

---

### Key Idea & Intuition

This problem is a variant of the **Climbing Stairs / Fibonacci** recurrence with dynamic constraints imposed by character validity:

At each position $i$ (considering prefixes of length $i$ from $1$ to $n$):
1. **Single-Digit Decode ($s[i-1]$):**
   - The single digit $s[i-1]$ can decode to a letter if and only if $s[i-1] \ne '0'$ (i.e., '1' through '9' correspond to 'A' through 'I').
   - If valid, it contributes $dp[i-1]$ ways.
2. **Two-Digit Decode ($s[i-2 \dots i-1]$):**
   - The two-digit substring formed by $s[i-2]$ and $s[i-1]$ can decode to a letter if and only if it represents an integer in the range $[10, 26]$ ('J' through 'Z').
   - Notice that values with leading zeros (e.g., "01" through "09") or values $\ge 27$ (e.g., "27", "30", "99") are invalid.
   - If valid, it contributes $dp[i-2]$ ways.

Thus, the recurrence relation is:
$$dp[i] = \underbrace{(dp[i-1] \text{ if } s[i-1] \in ['1'..'9'])}_{\text{1-digit branch}} + \underbrace{(dp[i-2] \text{ if } 10 \le \text{int}(s[i-2..i-1]) \le 26)}_{\text{2-digit branch}}$$

#### Space Optimization:
Since $dp[i]$ only depends on $dp[i-1]$ and $dp[i-2]$, we do not need a full array of size $n+1$. We can maintain just two rolling integer variables (`prev2` and `prev1`), reducing the auxiliary space complexity from $\mathcal{O}(n)$ to $\mathcal{O}(1)$.

---

### Solution Approach (Step-by-Step)

1. **Handle Initial Invalid Edge Case:**
   - If $s[0] == '0'$, no valid decoding can start with '0'. Return `0` immediately.
2. **Initialize Rolling Variables:**
   - `prev2 = 1`: Represents $dp[0] = 1$ (the empty prefix has exactly 1 valid way).
   - `prev1 = 1`: Represents $dp[1] = 1$ (since $s[0] \ne '0'$, a single non-zero digit has 1 valid way).
3. **Iterate from Index $i = 2$ to $n$ ($1$-based index):**
   - Let `curr = 0`.
   - **Check single digit ($s[i-1]$):**
     - If $s[i-1] \ne '0'$, `curr += prev1`.
   - **Check two digits ($s[i-2 \dots i-1]$):**
     - Compute the integer value $val = (s[i-2] - '0') \times 10 + (s[i-1] - '0')$.
     - If $10 \le val \le 26$, `curr += prev2`.
   - **Early Exit:**
     - If `curr == 0`, no valid decodings can continue from this point onward (e.g., "30" or "00"). Return `0`.
   - **Shift State:**
     - `prev2 = prev1`
     - `prev1 = curr`
4. **Return Final Result:**
   - After scanning the entire string, `prev1` holds the answer for the full string of length $n$.

---

### Visual Algorithm Walkthrough

#### Example 1: `s = "226"`
- Initial: `prev2 = 1` ($dp[0]$), `prev1 = 1` ($dp[1]$, digit `'2'`)

```
Index 2 (digit '2', substring "22"):
  1-digit check: '2' != '0'       --> + prev1 (1)
  2-digit check: "22" in [10, 26] --> + prev2 (1)
  curr = 1 + 1 = 2
  Shift: prev2 = 1, prev1 = 2

Index 3 (digit '6', substring "26"):
  1-digit check: '6' != '0'       --> + prev1 (2)
  2-digit check: "26" in [10, 26] --> + prev2 (1)
  curr = 2 + 1 = 3
  Shift: prev2 = 2, prev1 = 3

Result: prev1 = 3 (Decodings: "BBF", "BZ", "VF")
```

#### Example 2: Invalid Sequence `s = "230"`
```
Index 1: '2' != '0' -> prev2 = 1, prev1 = 1
Index 2: '3' != '0', "23" in [10, 26] -> curr = 1 + 1 = 2 -> prev2 = 1, prev1 = 2
Index 3: '0' == '0' (1-digit invalid)
         "30" not in [10, 26] (2-digit invalid)
         curr = 0 -> early exit return 0!
```

---

### Solved Examples with Multiple Inputs

| Input String `s` | Step-by-Step Decodings Evaluated | Output | Notes |
| :--- | :--- | :--- | :--- |
| `"12"` | $i=1$: '1' $\to$ 1; $i=2$: '2' (+1) or "12" (+1) $\to 2$ ("AB", "L") | `2` | Standard 2-digit branching |
| `"226"` | $i=2 \to 2$; $i=3$: '6' (+2) or "26" (+1) $\to 3$ | `3` | Valid single and double digits |
| `"06"` | $s[0] == '0' \to 0$ | `0` | Leading zero is completely invalid |
| `"10"` | $i=1$: '1'; $i=2$: '0' invalid as single, "10" valid $\to 1$ ("J") | `1` | Zero must be absorbed by preceding '1' |
| `"27"` | $i=2$: '7' valid (+1), "27" > 26 (invalid) $\to 1$ ("BG") | `1` | Numbers > 26 cannot form 2-digit letter |
| `"2101"` | "2" $\to$ "21" $\to$ "2 10" $\to$ "2 10 1" $\to 1$ ("BJA") | `1` | '0' forces merge with '1', single path |
| `"1001"` | $i=2$: "10" $\to 1$; $i=3$: '0' single invalid, "00" two-digit invalid $\to 0$ | `0` | Consecutive uncombinable zeros |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
        
        # prev2 represents dp[i-2], prev1 represents dp[i-1]
        prev2: int = 1  # dp[0]
        prev1: int = 1  # dp[1]
        
        for i in range(1, len(s)):
            curr: int = 0
            
            # Single-digit decode: s[i]
            if s[i] != '0':
                curr += prev1
            
            # Two-digit decode: s[i-1:i+1]
            two_digit = int(s[i-1:i+1])
            if 10 <= two_digit <= 26:
                curr += prev2
            
            # Early exit if string becomes undecodable
            if curr == 0:
                return 0
            
            prev2 = prev1
            prev1 = curr
            
        return prev1
```

#### C++17
```cpp
#include <string>

class Solution {
public:
    int numDecodings(const std::string& s) {
        if (s.empty() || s[0] == '0') {
            return 0;
        }

        int prev2 = 1; // dp[i-2]
        int prev1 = 1; // dp[i-1]

        for (size_t i = 1; i < s.length(); ++i) {
            int curr = 0;

            // Single digit decode: s[i]
            if (s[i] != '0') {
                curr += prev1;
            }

            // Two digit decode: s[i-1..i]
            int twoDigit = (s[i - 1] - '0') * 10 + (s[i] - '0');
            if (twoDigit >= 10 && twoDigit <= 26) {
                curr += prev2;
            }

            if (curr == 0) {
                return 0; // Cannot decode further
            }

            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
};
```

#### Java 17
```java
class Solution {
    public int numDecodings(String s) {
        if (s == null || s.isEmpty() || s.charAt(0) == '0') {
            return 0;
        }

        int prev2 = 1; // dp[i-2]
        int prev1 = 1; // dp[i-1]

        for (int i = 1; i < s.length(); i++) {
            int curr = 0;

            // Single-digit decode
            if (s.charAt(i) != '0') {
                curr += prev1;
            }

            // Two-digit decode
            int twoDigit = (s.charAt(i - 1) - '0') * 10 + (s.charAt(i) - '0');
            if (twoDigit >= 10 && twoDigit <= 26) {
                curr += prev2;
            }

            if (curr == 0) {
                return 0; // Invalid encoding detected
            }

            prev2 = prev1;
            prev1 = curr;
        }

        return prev1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$, where $n$ is the length of the string $s$. We perform a single linear scan from index $1$ to $n-1$, performing $\mathcal{O}(1)$ arithmetic and comparison operations at each step.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. Only three integer state variables (`prev2`, `prev1`, `curr`) are maintained throughout the iteration.

---

### Takeaway Pattern & Interview Traps

1. **The '0' Character Trap:**
   - A single `'0'` can NEVER be decoded independently. It can only be decoded as part of `"10"` ('J') or `"20"` ('T').
   - Any `'0'` preceded by anything other than `'1'` or `'2'` (e.g., `"30"`, `"70"`, or `"00"`) makes the entire message undecodable.
2. **Leading Zero on Multi-Digit Decodes:**
   - `"06"` is invalid. When checking two digits, ensure the value is $\ge 10$, which inherently excludes leading zero pairs like `"01"` through `"09"`.
3. **Space Optimization Invariant:**
   - When recurrence transitions $dp[i]$ strictly depend on a fixed window of preceding states (here $dp[i-1]$ and $dp[i-2]$), always eliminate the array allocation in favor of $\mathcal{O}(1)$ rolling variables.