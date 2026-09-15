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
---

# LeetCode 150: Evaluate Reverse Polish Notation

Below is a complete, structured explanation of **LeetCode 150 – Evaluate Reverse Polish Notation**, aligned with your usual preference for problem statement, core observations, stack insight, typed Python solution, and a fully worked example.

---

## LeetCode 150: Evaluate Reverse Polish Notation

### Problem Statement

You are given an array of strings `tokens` that represents an arithmetic expression in **Reverse Polish Notation (RPN)**.

Evaluate the expression and return an integer.

**Rules**

* Valid operators are `+`, `-`, `*`, and `/`
* Each operand may be an integer or another expression
* Division between two integers should **truncate toward zero**
* The input is guaranteed to be a valid RPN expression
* All intermediate results fit within 32-bit signed integer range

**Example**

```text
Input: ["2", "1", "+", "3", "*"]
Output: 9
```

---

## Key Observation

Reverse Polish Notation eliminates parentheses and operator precedence by enforcing **execution order** through position.

* Operators always appear **after** their operands
* When an operator is encountered, it must apply to the **most recent unresolved operands**

This naturally maps to a **Last-In-First-Out (LIFO)** structure.

---

## Stack: Key Insight

A **stack** is the optimal data structure because:

1. **Operands** are pushed as they appear
2. When an **operator** appears:

   * Pop the top two operands (right operand first)
   * Apply the operator
   * Push the result back onto the stack
3. At the end, the stack contains exactly **one value**, the final result

### Operand Order Matters

For subtraction and division:

```
a b -  →  a - b
a b /  →  a / b
```

Because of stack popping:

```
right = stack.pop()
left = stack.pop()
```

---

## Algorithm

1. Initialize an empty stack
2. Iterate through each token:

   * If token is a number → push onto stack
   * Else (operator):

     * Pop two operands
     * Apply operation
     * Push result
3. Return the only element left in the stack

---

## Python 3 Solution (with typing)

```python
from typing import List

class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack: List[int] = []

        for token in tokens:
            if token not in {"+", "-", "*", "/"}:
                stack.append(int(token))
            else:
                b = stack.pop()
                a = stack.pop()

                if token == "+":
                    stack.append(a + b)
                elif token == "-":
                    stack.append(a - b)
                elif token == "*":
                    stack.append(a * b)
                else:  # division
                    stack.append(int(a / b))  # truncate toward zero

        return stack[0]
```

---

## Worked Example (Step-by-Step)

### Input

```
tokens = ["4", "13", "5", "/", "+"]
```

### Evaluation Steps

| Token | Action | Stack State |
| --- | --- | --- |
| "4" | push 4 | [4] |
| "13" | push 13 | [4, 13] |
| "5" | push 5 | [4, 13, 5] |
| "/" | 13 / 5 = 2 | [4, 2] |
| "+" | 4 + 2 = 6 | [6] |

### Final Result

```text
Output: 6
```

---

## Complexity Analysis

* **Time Complexity:** `O(n)` — each token is processed once
* **Space Complexity:** `O(n)` — stack may hold up to `n` operands

---

## Why Stack Is the Optimal Choice

* RPN is evaluated strictly in encounter order
* No need for precedence rules or recursion
* Stack guarantees correctness and simplicity

---

If you want, I can also:

* Compare this with **infix evaluation**
* Show **why recursion is unnecessary**
* Extend this to **support variables or custom operators**