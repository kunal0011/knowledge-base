---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 670: Maximum Swap"
tags:
  - leetcode
  - coding
  - greedy
  - math
  - string
  - amazon
  - google
---

# LeetCode 670: Maximum Swap

**Target Companies:** Meta (Signature Top 1), Amazon, Google, Microsoft, Apple  
**Difficulty:** Medium  
**Topic:** Greedy / Math / String  

---

### Problem Statement

You are given an integer `num`. You can swap two digits at most once to get the maximum valued number.

Return the **maximum valued number** you can get.

---

### Input & Output Formats & Constraints

- **Input:**
  - `num`: `int` ($0 \le num \le 10^8$).
- **Output:**
  - `int` — the maximum integer achievable after at most one swap.
- **Constraints:**
  - $0 \le num \le 10^8$ (at most 9 digits).

---

### Key Idea & Intuition

In positional decimal notation, digits to the **left** have vastly higher significance than digits to the right.
To maximize the number with at most **one swap**:
1. We want to find the **leftmost** digit $s[i]$ that can be made larger.
2. What digit should we swap $s[i]$ with?
   - We want to replace $s[i]$ with the **largest possible digit** that appears to the right of $i$ (i.e. strictly greater than $s[i]$).
   - If there are multiple occurrences of this maximum digit to the right of $i$, which one do we choose?
   - We must choose the **rightmost occurrence**!
     - Swapping with the rightmost occurrence pushes the smaller digit $s[i]$ as far right (to the least significant place) as possible, keeping earlier occurrences of the large digit in their high-value positions.

#### Precomputing Last Seen Positions:
- Store the last index of each digit $0 \dots 9$ in an array `last_pos` of size 10.
- Iterate from left to right ($i = 0 \dots n - 1$):
  - Check digits $d$ from $9$ down to $s[i] + 1$:
    - If `last_pos[d] > i`:
      - Swap $s[i]$ with $s[\text{last\_pos}[d]]$.
      - Immediately return the resulting number!
- If no swap occurs, the number is already maximized (digits are non-increasing).

---

### Solution Approach (Step-by-Step)

1. Convert `num` to a list of characters `digits`.
2. Record the last occurrence of each digit:
   `last = {int(d): i for i, d in enumerate(digits)}`.
3. Loop $i$ from $0$ to $\text{len}(digits) - 1$:
   - For candidate digit $d$ from $9$ down to $\text{int}(digits[i]) + 1$:
     - If $d$ in `last` and `last[d] > i`:
       - Swap `digits[i]` and `digits[last[d]]`.
       - Return `int("".join(digits))`.
4. Return `num`.

---

### Visual Algorithm Walkthrough

For `num = 2736`:
Digits: `['2', '7', '3', '6']`
`last` map:
`2: 0, 7: 1, 3: 2, 6: 3`

```
i = 0 (digit '2'):
  Check candidate d = 9: not in last
  Check candidate d = 8: not in last
  Check candidate d = 7: last[7] = 1 > 0!
  -> Optimal swap found!
  Swap digits[0] ('2') with digits[1] ('7').
  Array becomes: ['7', '2', '3', '6']
  Stop and return 7236.
```

For `num = 9973`:
Digits: `['9', '9', '7', '3']`

```
i = 0 ('9'): candidate > 9: none
i = 1 ('9'): candidate > 9: none
i = 2 ('7'): candidates > 7:
  Check 9: last[9] = 1 < 2 (appears before, not after!)
  Check 8: not present
i = 3 ('3'): last digit, no digits to right.

No swap performed. Return 9973.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `num = 2736`
- **Output:** `7236`

#### Example 2:
- **Input:** `num = 9973`
- **Output:** `9973`

#### Example 3 (Duplicate Max Values):
- **Input:** `num = 1993`
- **Tracing:**
  - $i=0$ ('1'): largest digit to right is '9', with occurrences at index 1 and index 2.
  - Rightmost occurrence is index 2.
  - Swapping index 0 with index 2 yields `9913` (better than `9193` from swapping with index 1).
- **Output:** `9913`

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maximumSwap(self, num: int) -> int:
        digits = list(str(num))
        last = {int(d): i for i, d in enumerate(digits)}
        
        for i, ch in enumerate(digits):
            cur_digit = int(ch)
            # Look for a larger digit appearing after index i
            for d in range(9, cur_digit, -1):
                if last.get(d, -1) > i:
                    digits[i], digits[last[d]] = digits[last[d]], digits[i]
                    return int("".join(digits))
                    
        return num
```

#### C++17
```cpp
#include <string>
#include <vector>
#include <algorithm>

class Solution {
public:
    int maximumSwap(int num) {
        std::string s = std::to_string(num);
        std::vector<int> last(10, -1);
        
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            last[s[i] - '0'] = i;
        }
        
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            int cur_digit = s[i] - '0';
            for (int d = 9; d > cur_digit; --d) {
                if (last[d] > i) {
                    std::swap(s[i], s[last[d]]);
                    return std::stoi(s);
                }
            }
        }
        
        return num;
    }
};
```

#### Java 17
```java
class Solution {
    public int maximumSwap(int num) {
        char[] digits = Integer.toString(num).toCharArray();
        int[] last = new int[10];
        
        for (int i = 0; i < digits.length; i++) {
            last[digits[i] - '0'] = i;
        }
        
        for (int i = 0; i < digits.length; i++) {
            int curDigit = digits[i] - '0';
            for (int d = 9; d > curDigit; d--) {
                if (last[d] > i) {
                    char temp = digits[i];
                    digits[i] = digits[last[d]];
                    digits[last[d]] = temp;
                    return Integer.parseInt(new String(digits));
                }
            }
        }
        
        return num;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(L)$ where $L \le 9$ is the number of digits in `num`
  - String conversion takes $\mathcal{O}(L)$.
  - Outer loop runs $L$ times, inner digit loop runs at most 10 times.
  - Total operations $\le 10 \times 9 = 90$, which is $\mathcal{O}(1)$ practically.
- **Space Complexity:** $\mathcal{O}(L) = \mathcal{O}(1)$ auxiliary space
  - The character buffer and last-seen table of size 10 take negligible memory.

---

### Takeaway Pattern & Interview Traps

- **Why the Rightmost Occurrence Matters:** When multiple identical maximum digits exist to the right of $i$ (e.g. `1993`), swapping with the **rightmost** occurrence leaves all preceding copies of 9 untouched in higher place values, yielding `9913` instead of `9193`.
- **At Most One Swap:** Return immediately after performing the first valid swap; do not continue swapping.