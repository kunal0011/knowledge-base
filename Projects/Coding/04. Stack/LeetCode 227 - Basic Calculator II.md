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
---

# LeetCode 227: Basic Calculator II

**Target Companies:** Amazon, Google, Microsoft

---

### Problem Statement

Given a string `s` representing an expression containing non-negative integers, `'+'`, `'-'`, `'*'`, and `'/'`, evaluate the expression (without using `eval()`).

---

### Key Observation

* Operator precedence: `*` and `/` have higher precedence than `+` and `-`.
* Maintain a stack of terms. When a new operator or end of string is encountered:
* For `+`: push `+num`. For `-`: push `-num`.
* For `*`: pop top, multiply, push back. For `/`: pop top, integer divide (truncating toward zero: `int(top / num)`), push back.
* Sum the stack at the end.

---

### Core Technique: Operator Precedence Stack Evaluation

---

### Python 3 Solution (with typing)

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        curr_num = 0
        op = '+'
        s = s.replace(" ", "")
        
        for i, ch in enumerate(s):
            if ch.isdigit():
                curr_num = curr_num * 10 + int(ch)
            if not ch.isdigit() or i == len(s) - 1:
                if op == '+':
                    stack.append(curr_num)
                elif op == '-':
                    stack.append(-curr_num)
                elif op == '*':
                    stack.append(stack.pop() * curr_num)
                elif op == '/':
                    top = stack.pop()
                    stack.append(int(top / curr_num))  # truncate towards zero
                op = ch
                curr_num = 0
                
        return sum(stack)
```

---

### Worked-Out Example

```python
s = "3 + 2 * 2"
op='+', num=3 -> stack=[3]
op='+', num=2 -> stack=[3, 2]
op='*', num=2 -> pop 2, 2*2=4 -> stack=[3, 4]
sum(stack) = 3 + 4 = 7
```

---

### Complexity Analysis

* **Time Complexity:** `O(n)`
* **Space Complexity:** `O(n)`

---

### Takeaway Pattern

Delay addition/subtraction on a stack while immediately evaluating higher-precedence multiplication/division.