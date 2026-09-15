---
date: "2026-09-15"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 155: Min Stack"
tags:
  - leetcode
  - coding
  - stack
  - amazon
  - google
---

# LeetCode 155: Min Stack

**Target Companies:** Amazon (Top Classic), Google, Microsoft, Apple, Meta  
**Difficulty:** Medium  
**Topic:** Stack Design / Prefix Minimum Invariant

---

### Problem Statement

Design a stack that supports push, pop, top, and retrieving the minimum element in **constant time**.

Implement the `MinStack` class:
- `MinStack()`: initializes the stack object.
- `void push(int val)`: pushes the element `val` onto the stack.
- `void pop()`: removes the element on the top of the stack.
- `int top()`: gets the top element of the stack.
- `int getMin()`: retrieves the minimum element in the stack.

You must implement a solution with **$O(1)$ time complexity** for each function.

---

### Input & Output Formats & Constraints

- **Input Operations:** `MinStack()`, `push(val)`, `pop()`, `top()`, `getMin()`
- **Constraints:**
  - $-2^{31} \le \text{val} \le 2^{31} - 1$
  - Methods `pop`, `top` and `getMin` operations will always be called on **non-empty** stacks.
  - At most $3 \times 10^4$ calls will be made to `push`, `pop`, `top`, and `getMin`.

---

### Key Idea & Intuition

- **The Fundamental Challenge:**
  - A standard stack provides $O(1)$ push/pop/top. But tracking the minimum requires finding the smallest across all elements.
- **Prefix Minimum State Invariant:**
  - The minimum of elements currently in the stack depends **only** on the elements below it in the stack.
  - If we store `(val, current_min)` as a pair on the stack, each stack element remembers the exact minimum of the stack up to that height!
  - When an element is popped, the previous minimum is automatically restored with zero overhead.

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class MinStack:
    def __init__(self):
        # stack stores pairs of (val, min_so_far)
        self.stack = []

    def push(self, val: int) -> None:
        curr_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, curr_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

#### 2. C++ (C++17 / STL)
```cpp
#include <stack>
#include <algorithm>

class MinStack {
private:
    std::stack<std::pair<int, int>> st; // pair<val, minSoFar>

public:
    MinStack() {}

    void push(int val) {
        int currMin = st.empty() ? val : std::min(val, st.top().second);
        st.push({val, currMin});
    }

    void pop() {
        st.pop();
    }

    int top() {
        return st.top().first;
    }

    int getMin() {
        return st.top().second;
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class MinStack {
    private static class Element {
        int val, min;
        Element(int val, int min) { this.val = val; this.min = min; }
    }

    private final Deque<Element> stack;

    public MinStack() {
        this.stack = new ArrayDeque<>();
    }

    public void push(int val) {
        int currentMin = stack.isEmpty() ? val : Math.min(val, stack.peek().min);
        stack.push(new Element(val, currentMin));
    }

    public void pop() {
        stack.pop();
    }

    public int top() {
        return stack.peek().val;
    }

    public int getMin() {
        return stack.peek().min;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** Strict $O(1)$ for `push`, `pop`, `top`, and `getMin`.
- **Space Complexity:** $O(N)$ auxiliary space for paired storage.
