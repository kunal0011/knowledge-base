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
  - design
  - amazon
  - google
---

# LeetCode 155: Min Stack

**Target Companies:** Amazon, Google, Microsoft, Apple, Meta, Bloomberg  
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

You must implement a solution with **$\mathcal{O}(1)$ time complexity** for each function.

---

### Input & Output Formats & Constraints

- **Input Operations:** `MinStack()`, `push(val)`, `pop()`, `top()`, `getMin()`
- **Output:** Standard return values (`null`, `int`, or `void`).
- **Constraints:**
  - $-2^{31} \le \text{val} \le 2^{31} - 1$
  - Methods `pop`, `top` and `getMin` operations will always be called on **non-empty** stacks.
  - At most $3 \times 10^4$ calls will be made to `push`, `pop`, `top`, and `getMin`.

---

### Key Idea & Intuition

A standard stack gives $\mathcal{O}(1)$ operations for `push`, `pop`, and `top`. However, finding the minimum across all elements typically takes $\mathcal{O}(N)$.

#### Prefix Minimum Invariant
The minimum of elements currently in the stack depends **only** on the elements below it in the stack.
If we store `(val, min_so_far)` as a pair on the stack:
- Each stack frame records the value of the node alongside the exact minimum of the entire stack at that height.
- When an element is pushed:
  $$\text{current\_min} = \min(val, \text{stack[-1].min})$$
- When an element is popped, the stack naturally reverts to the previous frame's minimum without any search or recalculation.
- Thus, `getMin()` simply reads `stack[-1].min` in strictly $\mathcal{O}(1)$ time.

### Solution Approach (Step-by-Step)

1. **Underlying Storage:**
   - Maintain an underlying stack storing pairs or composite nodes: `(value, min_so_far)`.
2. **`push(val)` Operation:**
   - If the stack is empty, set `current_min = val`.
   - Else, set `current_min = min(val, stack.top().min_val)`.
   - Push `(val, current_min)` onto the stack.
3. **`pop()` Operation:**
   - Pop the top frame off the stack. The new top frame immediately exposes the minimum of all remaining elements below it.
4. **`top()` Operation:**
   - Return the `val` component of the top frame.
5. **`getMin()` Operation:**
   - Return the `min_so_far` component of the top frame in strict $\mathcal{O}(1)$ time.

---

### Visual Algorithm Walkthrough

Suppose we perform: `push(-2) -> push(0) -> push(-3) -> getMin() -> pop() -> top() -> getMin()`

```
1. push(-2):
   Stack is empty. min_so_far = -2.
   Stack: [ (-2, -2) ]

2. push(0):
   min_so_far = min(0, -2) = -2.
   Stack: [ (-2, -2), (0, -2) ]

3. push(-3):
   min_so_far = min(-3, -2) = -3.
   Stack: [ (-2, -2), (0, -2), (-3, -3) ]

4. getMin():
   Read stack.top().min -> returns -3.

5. pop():
   Pop (-3, -3).
   Stack becomes: [ (-2, -2), (0, -2) ]

6. top():
   Read stack.top().val -> returns 0.

7. getMin():
   Read stack.top().min -> returns -2.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Push and Pop Interleaving

- **Calls:** `["MinStack","push","push","push","getMin","pop","top","getMin"]`
- **Arguments:** `[[],[-2],[0],[-3],[],[],[],[]]`
- **Step Tracing:**

| Call | Stack Frame `(val, min)` | Returned Value | Description |
|:---:|:---:|:---:|:---:|
| `push(-2)` | `[(-2, -2)]` | `null` | -2 is first element |
| `push(0)` | `[(-2, -2), (0, -2)]` | `null` | Minimum remains -2 |
| `push(-3)` | `[(-2, -2), (0, -2), (-3, -3)]` | `null` | New minimum is -3 |
| `getMin()` | `[(-2, -2), (0, -2), (-3, -3)]` | `-3` | Top frame min is -3 |
| `pop()` | `[(-2, -2), (0, -2)]` | `null` | Restores previous frame |
| `top()` | `[(-2, -2), (0, -2)]` | `0` | Top value is 0 |
| `getMin()` | `[(-2, -2), (0, -2)]` | `-2` | Restores previous min -2 |

#### Example 2: Monotonically Decreasing Pushes

- `push(5) -> push(4) -> push(3) -> push(2)`
- Each node records itself as the new minimum.
- Each subsequent `pop()` cleanly peels off the corresponding minimum.

---

### Multi-Language Implementations

#### Python 3

```python
class MinStack:
    def __init__(self):
        # stack stores tuples of (value, min_so_far)
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

#### C++17

```cpp
#include <vector>
#include <algorithm>

class MinStack {
private:
    struct Node {
        int val;
        int min_val;
    };
    std::vector<Node> stack;

public:
    MinStack() {}

    void push(int val) {
        int curr_min = stack.empty() ? val : std::min(val, stack.back().min_val);
        stack.push_back({val, curr_min});
    }

    void pop() {
        stack.pop_back();
    }

    int top() {
        return stack.back().val;
    }

    int getMin() {
        return stack.back().min_val;
    }
};
```

#### Java

```java
import java.util.ArrayDeque;
import java.util.Deque;

public class MinStack {
    private static class Element {
        int val;
        int minVal;

        Element(int val, int minVal) {
            this.val = val;
            this.minVal = minVal;
        }
    }

    private final Deque<Element> stack;

    public MinStack() {
        this.stack = new ArrayDeque<>();
    }

    public void push(int val) {
        int currentMin = stack.isEmpty() ? val : Math.min(val, stack.peek().minVal);
        stack.push(new Element(val, currentMin));
    }

    public void pop() {
        stack.pop();
    }

    public int top() {
        return stack.peek().val;
    }

    public int getMin() {
        return stack.peek().minVal;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** Strict $\mathcal{O}(1)$ time for every operation:
  - `push`: $\mathcal{O}(1)$ (one comparison and append)
  - `pop`: $\mathcal{O}(1)$ (one pop)
  - `top`: $\mathcal{O}(1)$ (indexing top)
  - `getMin`: $\mathcal{O}(1)$ (indexing top)
- **Space Complexity:** $\mathcal{O}(N)$ auxiliary space to store $N$ elements and their running minima.

---

### Takeaway Pattern & Interview Traps

1. **Space Optimization Follow-Up (Single Stack with Value Encoding):**
   - Interviewers often ask: *"Can you do this without storing a pair at every node?"*
   - Yes, by storing the difference `val - min` or only pushing to a secondary `min_stack` when `val <= min_stack.top()`. The paired approach, however, avoids 32-bit integer arithmetic overflow (`diff = val - min` can overflow standard signed 32-bit ints when `val = INT_MAX` and `min = INT_MIN`).
2. **Empty Stack Preconditions:**
   - Always clarify with the interviewer whether `pop()` or `getMin()` can be invoked on an empty stack. Per LeetCode constraints, they are always called on non-empty stacks.
