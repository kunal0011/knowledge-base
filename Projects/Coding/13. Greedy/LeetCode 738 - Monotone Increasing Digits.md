---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 738: Monotone Increasing Digits"
tags:
  - leetcode
  - coding
  - greedy
  - math
  - string
  - amazon
  - google
  - microsoft
  - meta
---

# LeetCode 738: Monotone Increasing Digits

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg, Adobe  
**Difficulty:** Medium  
**Topic:** Greedy / Math / String  

---

### Problem Statement

An integer has **monotone increasing digits** if and only if each pair of adjacent digits `x` and `y` satisfy `x <= y`.

Given an integer `n`, return the largest number that is less than or equal to `n` and has **monotone increasing digits**.

---

### Input & Output Formats & Constraints

- **Input:** An integer `n` (`0 <= n <= 10^9`).
- **Output:** An integer representing the maximum monotone increasing number $\le n$.
- **Constraints:**
  - `0 <= n <= 10^9` (At most 10 digits in base 10; fits within standard 32-bit signed integer).

---

### Key Idea & Intuition

#### Mathematical Invariant & Greedy Principle
A number $N$ represented by digits $[d_0, d_1, \dots, d_{k-1}]$ is monotone increasing if and only if:
$$d_0 \le d_1 \le d_2 \le \dots \le d_{k-1}$$

Suppose scanning left-to-right we find the first violation where $d_{i-1} > d_i$. To make the prefix up to $i$ monotone while remaining $\le n$, our best move is to decrement $d_{i-1}$ by 1:
$$d_{i-1} \leftarrow d_{i-1} - 1$$
To make the overall number as large as possible, all trailing digits $j \ge i$ should be set to the maximum possible digit, which is `'9'`.

#### The Cascading Borrow Hazard
However, decrementing $d_{i-1}$ may cause a new violation with $d_{i-2}$ if $d_{i-2} == d_{i-1}$ initially.
For example, consider $n = 332$:
- Comparing $d_1 = 3$ and $d_2 = 2$: violation! Decrementing $d_1$ gives `[3, 2, 2]`.
- But now $d_0 = 3 > d_1 = 2$, which violates the monotone condition!
- Decrementing $d_0$ gives $d_0 = 2$, and now all digits from index 1 onward must be replaced by `'9'`, producing `299`.

#### Right-to-Left Traversal Invariant
To handle cascades seamlessly without complex lookbacks, we traverse **right-to-left** from index $k-1$ down to 1:
1. Maintain a pointer `mark = k`, indicating that all indices in `[mark, k-1]` will be overwritten with `'9'`.
2. Whenever $d_{i-1} > d_i$:
   - Decrement $d_{i-1} \leftarrow d_{i-1} - 1$.
   - Set `mark = i`.
3. After the traversal finishes, overwrite all digits from index `mark` up to $k-1$ with `'9'`.
4. Convert the character array back to an integer.

This right-to-left pass guarantees that any decrement will be checked against its predecessor in the very next step, perfectly resolving cascades in a single pass.

---

### Solution Approach (Step-by-Step)

1. **Convert to Mutable String / Character Array:**
   - Convert $n$ to string $s$, and length $k = |s|$.
   - Set `mark = k`.
2. **Right-to-Left Monotonicity Correction:**
   - For $i$ from $k-1$ down to $1$:
     - If $s[i-1] > s[i]$:
       - Decrement $s[i-1]$: `s[i-1] = char(s[i-1] - 1)`.
       - Update `mark = i`.
3. **Suffix 9-Fill:**
   - For $j$ from `mark` to $k-1$:
     - Set $s[j] = '9'$.
4. **Return Result:**
   - Parse $s$ into a 32-bit integer and return.

---

### Visual Algorithm Walkthrough

#### Trace of $n = 332$
```
Initial: s = ['3', '3', '2'], mark = 3
Indices:       0    1    2

Step 1 (i = 2):
Compare s[1] ('3') and s[2] ('2')
'3' > '2' -> VIOLATION!
Action: Decrement s[1] from '3' to '2'
        Set mark = 2
Array state: ['3', '2', '2'], mark = 2

Step 2 (i = 1):
Compare s[0] ('3') and s[1] ('2')
'3' > '2' -> VIOLATION!
Action: Decrement s[0] from '3' to '2'
        Set mark = 1
Array state: ['2', '2', '2'], mark = 1

Step 3 (Suffix Fill from mark = 1 to 2):
s[1] = '9'
s[2] = '9'
Array state: ['2', '9', '9']

Result: 299 <= 332, monotone increasing, maximized!
```

#### Trace of $n = 1234$
```
Initial: s = ['1', '2', '3', '4'], mark = 4
i = 3: s[2] ('3') <= s[3] ('4') -> valid
i = 2: s[1] ('2') <= s[2] ('3') -> valid
i = 1: s[0] ('1') <= s[1] ('2') -> valid

mark remains 4 -> No suffix fill.
Result: 1234
```

---

### Solved Examples with Multiple Inputs

| Input $n$ | Digits Array | Right-to-Left Traversal Decisions | Suffix 9-Fill Range | Output | Explanation |
|---|---|---|---|---|---|
| `10` | `['1', '0']` | $i=1$: '1' > '0' $\to s[0]='0', mark=1$ | $[1..1] \to s[1]='9'$ | `9` | Largest monotone $\le 10$ is 9 |
| `1234` | `['1','2','3','4']` | No violations; $mark=4$ | None | `1234` | Already monotone |
| `332` | `['3','3','2']` | $i=2$: $s[1]='2', mark=2$; $i=1$: $s[0]='2', mark=1$ | $[1..2] \to '9'$ | `299` | Cascading decrement |
| `100` | `['1','0','0']` | $i=2$: '0' $\le$ '0'; $i=1$: $s[0]='0', mark=1$ | $[1..2] \to '9'$ | `99` | Leading zero safely parsed |
| `0` | `['0']` | Loop doesn't run | None | `0` | Single digit base case |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        digits: list[str] = list(str(n))
        k: int = len(digits)
        mark: int = k
        
        # Traverse from right to left to propagate decrements correctly
        for i in range(k - 1, 0, -1):
            if digits[i - 1] > digits[i]:
                digits[i - 1] = str(int(digits[i - 1]) - 1)
                mark = i
                
        # Fill all digits from mark onward with '9'
        for j in range(mark, k):
            digits[j] = '9'
            
        return int("".join(digits))
```

#### C++17
```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        std::string s = std::to_string(n);
        int k = static_cast<int>(s.size());
        int mark = k;
        
        // Scan right to left
        for (int i = k - 1; i > 0; --i) {
            if (s[i - 1] > s[i]) {
                s[i - 1]--;
                mark = i;
            }
        }
        
        // Fill trailing suffix with '9'
        for (int j = mark; j < k; ++j) {
            s[j] = '9';
        }
        
        return std::stoi(s);
    }
};
```

#### Java 17
```java
class Solution {
    public int monotoneIncreasingDigits(int n) {
        char[] s = String.valueOf(n).toCharArray();
        int k = s.length;
        int mark = k;
        
        // Scan right to left
        for (int i = k - 1; i > 0; i--) {
            if (s[i - 1] > s[i]) {
                s[i - 1]--;
                mark = i;
            }
        }
        
        // Fill trailing suffix with '9'
        for (int j = mark; j < k; j++) {
            s[j] = '9';
        }
        
        return Integer.parseInt(new String(s));
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(d)$, where $d = \lfloor \log_{10} n \rfloor + 1$ is the number of decimal digits in $n$. Since $n \le 10^9$, $d \le 10$. The two linear sweeps (right-to-left scan and suffix fill) take at most $2 \times 10$ operations, which is strictly $\mathcal{O}(1)$ runtime in practice.
- **Space Complexity:** $\mathcal{O}(d)$ auxiliary space to store the string/character array representation of the digits ($d \le 10$ bytes).

---

### Takeaway Pattern & Interview Traps

1. **Left-to-Right Pitfall:** Scanning left-to-right and greedily setting $s[i-1] -= 1$ fails for duplicate digits like $332$ where changing $s[1]$ to $2$ violates $s[0] \le s[1]$. Scanning right-to-left handles the cascading carry-back naturally.
2. **Lazy Suffix Filling:** Rather than filling with `'9'` during the loop, merely tracking `mark` avoids repeated array overwrites and keeps the code clean and optimal.
3. **Leading Zero Handling:** Standard integer parsers (`int()` in Python, `std::stoi` in C++, `Integer.parseInt` in Java) automatically handle numbers like `"099"` converting correctly to `99`.