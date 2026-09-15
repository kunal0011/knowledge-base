---
date: "2026-09-15"
type: leetcode-solution
category: "Backtracking"
folder: "10. Backtracking"
title: "LeetCode 282: Expression Add Operators"
tags:
  - leetcode
  - coding
  - backtracking
  - amazon
  - google
---

# LeetCode 282: Expression Add Operators

**Target Companies:** Google (Signature Hard), Amazon, Meta  
**Difficulty:** Hard  
**Topic:** Backtracking / Operator Precedence Maintenance

---

### Problem Statement

Given a string `num` that contains only digits and an integer `target`, return **all possibilities** to insert the binary operators `'+'`, `'-'`, and/or `'*'` between the digits of `num` so that the resultant expression evaluates to the `target` value.

Note that operands in the returned expressions **should not contain leading zeros**.

---

### Input & Output Formats & Constraints

- **Input:** `num: str`, `target: int`
- **Output:** `List[str]` containing all valid arithmetic expressions.
- **Constraints:**
  - $1 \le \text{num.length} \le 10$
  - `num` consists of only digits.
  - $-2^{31} \le \text{target} \le 2^{31} - 1$

---

### Key Idea & Intuition

- **The Operator Precedence Problem with Multiplication:**
  - Addition and subtraction are straightforward:
    - $a + b \implies \text{eval} + b$
    - $a - b \implies \text{eval} - b$
  - Multiplication breaks linear left-to-right evaluation due to precedence:
    - In $2 + 3 * 4$, when we reach $* 4$, the previously evaluated result was $2 + 3 = 5$.
    - We cannot compute $5 * 4 = 20$! The correct evaluation is $2 + (3 * 4) = 14$.
- **The "Previous Operand" Tracker:**
  - Maintain two values along the recursion tree:
    1. `current_val`: The running evaluated total of the expression so far.
    2. `prev_operand`: The last added or subtracted term.
  - For `+ curr`: `current_val + curr`, `prev_operand = curr`
  - For `- curr`: `current_val - curr`, `prev_operand = -curr`
  - For `* curr`: `current_val - prev_operand + (prev_operand * curr)`, `prev_operand = prev_operand * curr`!
- **Edge Case (Leading Zeros):**
  - If a sliced operand begins with `'0'` and has length $> 1$ (e.g. `'05'`), it is invalid and must be pruned immediately.

---

### Solution Approach (Step-by-Step)

1. Define recursive helper `backtrack(index, prev_operand, current_val, expression_parts)`:
   - Base Case: When `index == len(num)`:
     - If `current_val == target`: join expression and append to results.
   - For `end` from `index + 1` to `len(num) + 1`:
     - Extract substring `part = num[index:end]`.
     - If `len(part) > 1 and part[0] == '0'`: `break` (prune leading zero).
     - `val = int(part)`.
     - If `index == 0` (first number):
       - `backtrack(end, val, val, [part])`
     - Else:
       - Addition: `backtrack(end, val, current_val + val, expr + ['+', part])`
       - Subtraction: `backtrack(end, -val, current_val - val, expr + ['-', part])`
       - Multiplication: `backtrack(end, prev_operand * val, current_val - prev_operand + prev_operand * val, expr + ['*', part])`
2. Return collected expressions.

---

### Visual Algorithm Walkthrough

```
num = "123", target = 6

Level 0: Pick "1"
  + "2" -> eval = 3, prev = 2
    + "3" -> eval = 6 == target -> "1+2+3" MATCH!
    * "3" -> eval = 3 - 2 + (2*3) = 7 != target
  * "2" -> eval = 2, prev = 2
    + "3" -> eval = 5 != target
    * "3" -> eval = 2 - 2 + (2*3) = 6 == target -> "1*2*3" MATCH!

Results: ["1+2+3", "1*2*3"]
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        res = []
        n = len(num)
        
        def backtrack(idx: int, prev: int, curr_val: int, path: List[str]):
            if idx == n:
                if curr_val == target:
                    res.append("".join(path))
                return
                
            for end in range(idx + 1, n + 1):
                part = num[idx:end]
                if len(part) > 1 and part[0] == '0':
                    break  # Disallow leading zero
                    
                val = int(part)
                
                if idx == 0:
                    path.append(part)
                    backtrack(end, val, val, path)
                    path.pop()
                else:
                    # Addition
                    path.append('+')
                    path.append(part)
                    backtrack(end, val, curr_val + val, path)
                    path.pop()
                    path.pop()
                    
                    # Subtraction
                    path.append('-')
                    path.append(part)
                    backtrack(end, -val, curr_val - val, path)
                    path.pop()
                    path.pop()
                    
                    # Multiplication
                    path.append('*')
                    path.append(part)
                    backtrack(end, prev * val, curr_val - prev + (prev * val), path)
                    path.pop()
                    path.pop()
                    
        backtrack(0, 0, 0, [])
        return res
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <string>

class Solution {
private:
    std::vector<std::string> result;
    std::string numStr;
    long targetVal;

    void backtrack(int idx, long prev, long currVal, std::string path) {
        if (idx == numStr.size()) {
            if (currVal == targetVal) {
                result.push_back(path);
            }
            return;
        }

        for (int len = 1; idx + len <= numStr.size(); ++len) {
            std::string part = numStr.substr(idx, len);
            if (part.size() > 1 && part[0] == '0') break;

            long val = std::stol(part);

            if (idx == 0) {
                backtrack(idx + len, val, val, part);
            } else {
                backtrack(idx + len, val, currVal + val, path + "+" + part);
                backtrack(idx + len, -val, currVal - val, path + "-" + part);
                backtrack(idx + len, prev * val, currVal - prev + (prev * val), path + "*" + part);
            }
        }
    }

public:
    std::vector<std::string> addOperators(std::string num, int target) {
        numStr = num;
        targetVal = target;
        backtrack(0, 0, 0, "");
        return result;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.*;

class Solution {
    private List<String> result;
    private String num;
    private long target;

    public List<String> addOperators(String num, int target) {
        this.result = new ArrayList<>();
        this.num = num;
        this.target = target;
        backtrack(0, 0, 0, new StringBuilder());
        return result;
    }

    private void backtrack(int idx, long prev, long currVal, StringBuilder path) {
        if (idx == num.length()) {
            if (currVal == target) {
                result.add(path.toString());
            }
            return;
        }

        int len = path.length();
        for (int end = idx + 1; end <= num.length(); end++) {
            String part = num.substring(idx, end);
            if (part.length() > 1 && part.charAt(0) == '0') break;

            long val = Long.parseLong(part);

            if (idx == 0) {
                path.append(part);
                backtrack(end, val, val, path);
                path.setLength(len);
            } else {
                // Add
                path.append("+").append(part);
                backtrack(end, val, currVal + val, path);
                path.setLength(len);

                // Subtract
                path.append("-").append(part);
                backtrack(end, -val, currVal - val, path);
                path.setLength(len);

                // Multiply
                path.append("*").append(part);
                backtrack(end, prev * val, currVal - prev + (prev * val), path);
                path.setLength(len);
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(4^N)$ — At each of the $N - 1$ spaces between digits, we have 4 choices: No-op (extend number), `+`, `-`, or `*`.
- **Space Complexity:** $O(N)$ — Maximum recursion stack depth is $N$.
