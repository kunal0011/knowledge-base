---
date: "2026-08-29"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 227: Basic Calculator II"
tags:
  - leetcode
  - coding
  - stack
  - math
  - string
  - amazon
  - google
---

# LeetCode 227: Basic Calculator II

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg, Apple  
**Difficulty:** Medium  
**Topic:** Stack / Expression Evaluation / Operator Precedence

---

### Problem Statement

Given a string `s` which represents an expression, *evaluate this expression and return its value*. 

The integer division should truncate toward zero.

You may assume that the given expression is always valid. All intermediate results will be in the range of $[-2^{31}, 2^{31} - 1]$.

**Note:** You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as `eval()`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `s`: `str`, where $1 \le \text{len}(s) \le 3 \times 10^5$.
  - `s` consists of integers and operators (`'+'`, `'-'`, `'*'`, `'/'`) separated by some number of spaces.
  - `s` represents a valid arithmetic expression. All integers in the expression are non-negative and in the range $[0, 2^{31} - 1]$.
- **Output:**
  - `int`: The evaluated integer value after truncating all divisions toward zero.
- **Constraints:**
  - Division by zero will not occur.
  - Intermediate and final values fit in a 32-bit signed integer.

---

### Key Idea & Intuition

In standard arithmetic, multiplication (`*`) and division (`/`) have strictly higher precedence than addition (`+`) and subtraction (`-`).

Because there are no parentheses:
- Addition and subtraction can be **delayed** until all higher precedence operations are completed.
- Multiplication and division must be **evaluated immediately** with the preceding operand.

#### Stack-Based Parsing:
1. Maintain `curr_num` accumulating incoming digits.
2. Maintain `prev_op` (initialized to `'+'`). This tracks the operator that came *immediately before* `curr_num`.
3. When we hit an operator or the end of the string:
   - If `prev_op == '+'`: Push `+curr_num` onto stack.
   - If `prev_op == '-'`: Push `-curr_num` onto stack.
   - If `prev_op == '*'`: Pop the top from the stack, multiply by `curr_num`, and push the product back.
   - If `prev_op == '/'`: Pop the top from the stack, divide by `curr_num` (truncating towards zero), and push the quotient back.
   - Update `prev_op` to the current operator character, and reset `curr_num = 0`.
4. After processing the entire string, the stack contains terms that only need to be added together. The answer is simply `sum(stack)`.

#### Follow-Up: Constant Space $\mathcal{O}(1)$
Notice that we only ever pop the most recent value from the stack. We can replace the stack with two variables:
- `last_num`: the evaluated value of the current product/quotient term.
- `total_sum`: the running sum of completed addition/subtraction terms.

---

### Solution Approach (Step-by-Step)

1. Strip whitespace or simply ignore space characters `ch == ' '`.
2. Initialize `stack = []`, `curr_num = 0`, `prev_op = '+'`.
3. Iterate $i$ through $0 \dots \text{len}(s) - 1$:
   - If $s[i]$ is a digit: `curr_num = curr_num * 10 + int(s[i])`.
   - If $s[i]$ is an operator (`+`, `-`, `*`, `/`) or $i == \text{len}(s) - 1$:
     - Apply `prev_op` to `curr_num`:
       - `'+'`: `stack.append(curr_num)`
       - `'-'`: `stack.append(-curr_num)`
       - `'*'`: `stack.append(stack.pop() * curr_num)`
       - `'/'`: `stack.append(int(stack.pop() / curr_num))`
     - `prev_op = s[i]`
     - `curr_num = 0`
4. Return `sum(stack)`.

---

### Visual Algorithm Walkthrough

Evaluate $s = \text{"3 + 2 * 2"}$:

```
Initial: stack = [], curr_num = 0, prev_op = '+'

i = 0, ch = '3':
  curr_num = 3
  End of number not reached yet? Wait, next char is '+'.
  Trigger operator update for prev_op '+':
  Push 3 to stack. Stack = [3]
  prev_op = '+', curr_num = 0

i = 2, ch = '2':
  curr_num = 2
  Next char is '*'.
  Trigger operator update for prev_op '+':
  Push 2 to stack. Stack = [3, 2]
  prev_op = '*', curr_num = 0

i = 4, ch = '2':
  curr_num = 2
  End of string reached!
  Trigger operator update for prev_op '*':
  Pop top (2). Compute 2 * 2 = 4.
  Push 4 to stack. Stack = [3, 4]

Final Sum:
  3 + 4 = 7
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Precedence

- **Input:** `s = " 3/2 "`
- **Execution:**
  - `prev_op = '+'`, `curr_num = 3`. Next is `/`. Push `3`. `prev_op = '/'`.
  - `curr_num = 2`. End of string. Pop `3`, compute `3 / 2 = 1`. Push `1`.
  - Sum: `1`.
- **Output:** `1`

#### Example 2: Negative and Division Truncation

- **Input:** `s = "14 - 3/2"`
- **Execution:**
  - Push `14`. `prev_op = '-'`.
  - Read `3`. Next is `/`. Push `-3`. `prev_op = '/'`.
  - Read `2`. End of string. Pop `-3`. `int(-3 / 2) = -1`. Push `-1`.
  - Sum: `14 + (-1) = 13`.
- **Output:** `13`

#### Example 3: Multiple Consecutive High-Precedence Operations

- **Input:** `s = "2 * 3 * 4"`
- **Execution:**
  - `prev_op = '+'`, push `2`. `prev_op = '*'`.
  - Pop `2`, compute $2 \times 3 = 6$. Push `6`. `prev_op = '*'`.
  - Pop `6`, compute $6 \times 4 = 24$. Push `24`.
  - Sum: `24`.
- **Output:** `24`

---

### Multi-Language Implementations

#### Python 3

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        curr_num = 0
        prev_op = '+'
        n = len(s)

        for i, ch in enumerate(s):
            if ch.isdigit():
                curr_num = curr_num * 10 + int(ch)

            # Check if we hit an operator or reached the end of the string
            if (not ch.isdigit() and ch != ' ') or i == n - 1:
                if prev_op == '+':
                    stack.append(curr_num)
                elif prev_op == '-':
                    stack.append(-curr_num)
                elif prev_op == '*':
                    stack.append(stack.pop() * curr_num)
                elif prev_op == '/':
                    top = stack.pop()
                    # Python floor division // rounds down towards -inf.
                    # int(top / curr_num) properly truncates toward zero.
                    stack.append(int(top / curr_num))
                
                prev_op = ch
                curr_num = 0

        return sum(stack)
```

#### C++17

```cpp
#include <string>
#include <vector>
#include <numeric>
#include <cctype>

class Solution {
public:
    int calculate(const std::string& s) {
        std::vector<long long> stack;
        long long curr_num = 0;
        char prev_op = '+';
        int n = static_cast<int>(s.size());

        for (int i = 0; i < n; ++i) {
            char ch = s[i];
            if (std::isdigit(ch)) {
                curr_num = curr_num * 10 + (ch - '0');
            }

            if ((!std::isdigit(ch) && ch != ' ') || i == n - 1) {
                if (prev_op == '+') {
                    stack.push_back(curr_num);
                } else if (prev_op == '-') {
                    stack.push_back(-curr_num);
                } else if (prev_op == '*') {
                    long long top = stack.back();
                    stack.pop_back();
                    stack.push_back(top * curr_num);
                } else if (prev_op == '/') {
                    long long top = stack.back();
                    stack.pop_back();
                    stack.push_back(top / curr_num); // C++ truncates towards zero
                }
                prev_op = ch;
                curr_num = 0;
            }
        }

        long long total = 0;
        for (long long val : stack) {
            total += val;
        }
        return static_cast<int>(total);
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int calculate(String s) {
        Deque<Integer> stack = new ArrayDeque<>();
        int currNum = 0;
        char prevOp = '+';
        int n = s.length();

        for (int i = 0; i < n; i++) {
            char ch = s.charAt(i);

            if (Character.isDigit(ch)) {
                currNum = currNum * 10 + (ch - '0');
            }

            if ((!Character.isDigit(ch) && ch != ' ') || i == n - 1) {
                if (prevOp == '+') {
                    stack.push(currNum);
                } else if (prevOp == '-') {
                    stack.push(-currNum);
                } else if (prevOp == '*') {
                    stack.push(stack.pop() * currNum);
                } else if (prevOp == '/') {
                    stack.push(stack.pop() / currNum); // Java truncates toward zero
                }
                prevOp = ch;
                currNum = 0;
            }
        }

        int total = 0;
        for (int val : stack) {
            total += val;
        }
        return total;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - A single pass through the string of length $N$.
  - Each character is visited once, and each stack operation takes $\mathcal{O}(1)$ time.
  - Final summation takes $\mathcal{O}(\text{len}(\text{stack})) \le \mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - The stack holds at most $\lceil N / 2 \rceil$ numbers in expressions dominated by addition/subtraction.
  - *Note:* Space can be reduced to $\mathcal{O}(1)$ by keeping only `running_sum` and `last_term`.

---

### Takeaway Pattern & Interview Traps

1. **End-of-String Flush:**
   - The most common pitfall is forgetting to process the final number at $i == n - 1$. The condition `(!isdigit(ch) && ch != ' ') || i == n - 1` ensures the terminal operand is pushed or multiplied/divided.
2. **Delayed Operator Association:**
   - The operator processed at any point is **not** the current operator character, but the **preceding operator** (`prev_op`). The current operator only tells us that the current number has ended!
3. **Division Truncation:**
   - Always remember that Python's `//` rounds toward $-\infty$, whereas `int(a / b)` truncates toward $0$. In C++ and Java, `/` naturally truncates toward zero.