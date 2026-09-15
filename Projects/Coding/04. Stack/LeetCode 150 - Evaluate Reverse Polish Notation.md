---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 150: Evaluate Reverse Polish Notation"
tags:
  - leetcode
  - coding
  - stack
  - math
  - amazon
  - google
---

# LeetCode 150: Evaluate Reverse Polish Notation

**Target Companies:** Amazon, Google, Microsoft, Meta, Bloomberg, LinkedIn  
**Difficulty:** Medium  
**Topic:** Stack / Expression Evaluation / Arithmetic

---

### Problem Statement

You are given an array of strings `tokens` that represents an arithmetic expression in a **Reverse Polish Notation** (Postfix Notation).

Evaluate the expression. Return *an integer that represents the value of the expression*.

**Rules:**
- The valid operators are `'+'`, `'-'`, `'*'`, and `'/'`.
- Each operand may be an integer or another expression.
- The division between two integers always **truncates toward zero**.
- There will not be any division by zero.
- The input represents a valid arithmetic expression in a reverse polish notation.
- The answer and all intermediate calculations can be represented in a **32-bit** signed integer.

---

### Input & Output Formats & Constraints

- **Input:**
  - `tokens`: `List[str]`, where $1 \le \text{tokens.length} \le 10^4$. Each token is either an operator (`'+'`, `'-'`, `'*'`, or `'/'`) or an integer in the range $[-200, 200]$.
- **Output:**
  - `int`: The evaluated 32-bit signed integer result.
- **Constraints:**
  - Division by zero never occurs.
  - Guaranteed valid expression.
  - Intermediate values fit within standard 32-bit signed integer limits.

---

### Key Idea & Intuition

In **Reverse Polish Notation (RPN)** (also known as postfix notation), operators follow their operands. This notation eliminates the need for parentheses and operator precedence rules.

Because operands precede the operator that acts upon them:
- When we encounter numbers, they must be remembered in arrival order.
- When an operator is encountered, it applies immediately to the **two most recently encountered unresolved operands**.
- The result of that operation then becomes an available operand for subsequent operators.

This **Last-In-First-Out (LIFO)** ordering maps directly to a **Stack**:
1. Scan tokens from left to right.
2. If token is a number: push its integer value onto the stack.
3. If token is an operator:
   - Pop the right-hand operand: $b = \text{stack.pop()}$.
   - Pop the left-hand operand: $a = \text{stack.pop()}$.
   - Apply the operation: $result = a \text{ op } b$.
   - Push $result$ back onto the stack.
4. After scanning all tokens, the stack will contain exactly one element: the final evaluated value.

---

### Solution Approach (Step-by-Step)

1. **Stack Initialization:**
   - Create an empty stack of integers.

2. **Iterate Through Tokens:**
   - For each token $t$:
     - Check if $t$ is one of `{"+", "-", "*", "/"}`.
     - If not an operator: convert $t$ to integer and push.
     - If an operator:
       - Pop top element as `b` (second operand).
       - Pop next top element as `a` (first operand).
       - Compute:
         - If $t == \text{"+"}$: push $a + b$
         - If $t == \text{"-"}$: push $a - b$
         - If $t == \text{"*"}$: push $a \times b$
         - If $t == \text{"/"}$: push $\text{trunc}(a / b)$
           *(In Python, use `int(a / b)` to truncate towards zero, avoiding Python's default floor division `//`).*

3. **Return Result:**
   - Return `stack.pop()`.

---

### Visual Algorithm Walkthrough

Let `tokens = ["4", "13", "5", "/", "+"]`:

```
Token: "4"   -> Operand.  Push 4.             Stack: [4]
Token: "13"  -> Operand.  Push 13.            Stack: [4, 13]
Token: "5"   -> Operand.  Push 5.             Stack: [4, 13, 5]
Token: "/"   -> Operator! Pop b=5, Pop a=13.
                Compute a / b = 13 / 5 = 2.
                Push 2.                       Stack: [4, 2]
Token: "+"   -> Operator! Pop b=2, Pop a=4.
                Compute a + b = 4 + 2 = 6.
                Push 6.                       Stack: [6]

End of tokens. Final value = stack.top() = 6.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Evaluation

- **Input:** `tokens = ["2", "1", "+", "3", "*"]`
- **Execution:**
  - `"2"`, `"1"` pushed: Stack = `[2, 1]`
  - `"+"`: `1 + 2 = 3`. Stack = `[3]`
  - `"3"` pushed: Stack = `[3, 3]`
  - `"*"`: `3 * 3 = 9`. Stack = `[9]`
- **Output:** `9` (Expression equivalent to $((2 + 1) \times 3)$).

#### Example 2: Negative Numbers & Truncation Toward Zero

- **Input:** `tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]`
- **Tracing Operator Sub-Steps:**
  - `9 + 3 = 12`
  - `12 * -11 = -132`
  - `6 / -132 = 0` (truncated toward zero!)
  - `10 * 0 = 0`
  - `0 + 17 = 17`
  - `17 + 5 = 22`
- **Output:** `22`

#### Example 3: Single Number

- **Input:** `tokens = ["42"]`
- **Execution:** Stack receives `42`, loop terminates, returns `42`.
- **Output:** `42`

---

### Multi-Language Implementations

#### Python 3

```python
from typing import List

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack: List[int] = []

        for token in tokens:
            if token == "+":
                stack.append(stack.pop() + stack.pop())
            elif token == "-":
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            elif token == "*":
                stack.append(stack.pop() * stack.pop())
            elif token == "/":
                b = stack.pop()
                a = stack.pop()
                # Note: In Python, a // b floors toward -infinity.
                # int(a / b) truncates toward zero as required.
                stack.append(int(a / b))
            else:
                stack.append(int(token))

        return stack[0]
```

#### C++17

```cpp
#include <vector>
#include <string>
#include <stack>

class Solution {
public:
    int evalRPN(const std::vector<std::string>& tokens) {
        std::vector<int> stack;
        stack.reserve(tokens.size());

        for (const std::string& token : tokens) {
            if (token == "+" || token == "-" || token == "*" || token == "/") {
                int b = stack.back(); stack.pop_back();
                int a = stack.back(); stack.pop_back();

                if (token == "+") stack.push_back(a + b);
                else if (token == "-") stack.push_back(a - b);
                else if (token == "*") stack.push_back(a * b);
                else if (token == "/") stack.push_back(a / b); // C++11 truncates toward zero
            } else {
                stack.push_back(std::stoi(token));
            }
        }

        return stack.back();
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class Solution {
    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new ArrayDeque<>();

        for (String token : tokens) {
            if (token.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (token.equals("-")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a - b);
            } else if (token.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (token.equals("/")) {
                int b = stack.pop();
                int a = stack.pop();
                stack.push(a / b); // Java integer division truncates toward zero
            } else {
                stack.push(Integer.parseInt(token));
            }
        }

        return stack.pop();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$
  - Each token is processed once in a single pass of length $N$.
  - Stack pushes, pops, string-to-int conversions, and basic arithmetic take $\mathcal{O}(1)$ time.
  - Overall Time: $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(N)$
  - In the worst case (e.g., all numbers first, followed by operators), the stack stores up to $(N + 1) / 2$ numbers.
  - Overall Space: $\mathcal{O}(N)$.

---

### Takeaway Pattern & Interview Traps

1. **Operand Popping Order:**
   - Because of LIFO order, the **first popped item is the second (right) operand** ($b$), and the **second popped item is the first (left) operand** ($a$).
   - For non-commutative operations (`-` and `/`), failing to assign $a - b$ or $a / b$ correctly produces invalid inverted results.
2. **Python Division Towards Zero Trap:**
   - In Python 3: `-3 // 2 == -2` (floor division), but `-3 / 2 == -1.5` and `int(-3 / 2) == -1` (truncation towards zero).
   - Using `//` in Python fails LeetCode test cases with negative numbers. Always use `int(a / b)` or `math.trunc(a / b)`.
3. **Array as Stack Optimization:**
   - In C++ and Java, using `std::vector` or a flat primitive array with a pointer index is noticeably faster than `std::stack` or `Stack<Integer>` due to cache locality and no dynamic boxing.