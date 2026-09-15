---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 232: Implement Queue using Stacks"
tags:
  - leetcode
  - coding
  - queue
  - stack
  - amortized-analysis
  - amazon
  - google
---

# LeetCode 232: Implement Queue using Stacks

**Target Companies:** Amazon, Microsoft, Apple, Google, Bloomberg  
**Difficulty:** Easy  
**Topic:** Amortized $O(1)$ Queue via Dual Stacks (In-Stack / Out-Stack)  

---

### Problem Statement

Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (`push`, `peek`, `pop`, and `empty`).

Implement the `MyQueue` class:
- `void push(int x)`: Pushes element `x` to the back of the queue.
- `int pop()`: Removes the element from the front of the queue and returns it.
- `int peek()`: Returns the element at the front of the queue.
- `boolean empty()`: Returns `true` if the queue is empty, `false` otherwise.

**Notes:**
- You must use **only** standard operations of a stack — which means only `push to top`, `peek/pop from top`, `size`, and `is empty` operations are valid.

---

### Input & Output Formats & Constraints

- **Input:** Sequence of operations: `["MyQueue", "push", "push", "peek", "pop", "empty"]`
- **Output:** Returns from functions (`null`, `null`, `null`, `1`, `1`, `false`)
- **Constraints:**
  - $1 \le x \le 9$
  - At most $100$ calls will be made to `push`, `pop`, `peek`, and `empty`.
  - All calls to `pop` and `peek` are guaranteed to be valid (queue is non-empty).

---

### Key Idea & Intuition

A stack is **LIFO** (Last In, First Out), whereas a queue is **FIFO** (First In, First Out).
Pumping elements into a stack and popping them out inverts their order. Reversing an inverted sequence restores the original FIFO order:
$$(A, B, C) \xrightarrow{\text{Stack 1}} (C, B, A) \xrightarrow{\text{Stack 2}} (A, B, C)$$

Rather than shifting all elements on every single operation (which costs $O(N)$ per operation), we use **lazy transfer**:
1. `in_stack`: Receives every new element pushed into the queue.
2. `out_stack`: Holds elements in inverted order ready to be popped or peeked.
3. Whenever `pop()` or `peek()` is requested:
   - If `out_stack` already contains elements, its top element is already the oldest element in the queue. We pop or peek immediately in $O(1)$!
   - If `out_stack` is empty, we transfer **all** elements from `in_stack` to `out_stack` in one go.
4. Each element is pushed to `in_stack` once, moved to `out_stack` once, and popped from `out_stack` once. Hence, each element incurs at most $O(1)$ amortized operations throughout its lifetime.

---

### Solution Approach (Step-by-Step)

1. `__init__()`: Initialize two empty stacks `in_stack` and `out_stack`.
2. `push(x)`: Append `x` directly to `in_stack`.
3. `_shift()`: If `out_stack` is empty, while `in_stack` is not empty, pop the top of `in_stack` and push it onto `out_stack`.
4. `pop()`: Call `_shift()`, then return `out_stack.pop()`.
5. `peek()`: Call `_shift()`, then return the top element of `out_stack`.
6. `empty()`: Return `true` if and only if both `in_stack` and `out_stack` are empty.

---

### Visual Algorithm Walkthrough

```
Operations: push(1), push(2), peek(), pop(), empty()

1. push(1):
   in_stack:  [1] (top=1)
   out_stack: []

2. push(2):
   in_stack:  [1, 2] (top=2)
   out_stack: []

3. peek():
   out_stack is empty -> Transfer in_stack -> out_stack
   pop 2 -> push to out_stack: [2]
   pop 1 -> push to out_stack: [2, 1] (top=1)
   in_stack is now []
   Peek out_stack top -> returns 1.

4. pop():
   out_stack is NOT empty -> directly pop top (1)
   out_stack: [2] (top=2)
   returns 1.

5. empty():
   in_stack is empty, but out_stack has [2].
   Returns False.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Lifecycle
```text
Commands: ["MyQueue", "push", "push", "peek", "pop", "empty"]
Inputs:   [[], [1], [2], [], [], []]
Execution:
- MyQueue()  -> Initialize
- push(1)    -> in_stack: [1]
- push(2)    -> in_stack: [1, 2]
- peek()     -> transfers to out_stack: [2, 1] -> returns 1
- pop()      -> pops 1 from out_stack -> returns 1
- empty()    -> in_stack=[], out_stack=[2] -> returns false
Output: [null, null, null, 1, 1, false]
```

#### Example 2: Interleaved Push and Pop
```text
Commands: push(1), pop() -> 1, push(2), peek() -> 2, pop() -> 2, empty() -> true
Trace:
- push(1): in=[1], out=[]
- pop(): shift -> in=[], out=[1] -> returns 1
- push(2): in=[2], out=[]
- peek(): shift -> in=[], out=[2] -> returns 2
- pop(): returns 2
- empty(): in=[], out=[] -> returns true
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class MyQueue:
    def __init__(self) -> None:
        self.in_stack: list[int] = []
        self.out_stack: list[int] = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self._shift()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._shift()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

    def _shift(self) -> None:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
```

#### 2. C++ (C++17 / STL)
```cpp
#include <stack>

class MyQueue {
private:
    std::stack<int> inStack;
    std::stack<int> outStack;

    void shift() {
        if (outStack.empty()) {
            while (!inStack.empty()) {
                outStack.push(inStack.top());
                inStack.pop();
            }
        }
    }

public:
    MyQueue() {}

    void push(int x) {
        inStack.push(x);
    }

    int pop() {
        shift();
        int val = outStack.top();
        outStack.pop();
        return val;
    }

    int peek() {
        shift();
        return outStack.top();
    }

    bool empty() {
        return inStack.empty() && outStack.empty();
    }
};
```

#### 3. Java (Modern, Typed)
```java
import java.util.ArrayDeque;
import java.util.Deque;

class MyQueue {
    private final Deque<Integer> inStack;
    private final Deque<Integer> outStack;

    public MyQueue() {
        this.inStack = new ArrayDeque<>();
        this.outStack = new ArrayDeque<>();
    }

    public void push(int x) {
        inStack.push(x);
    }

    public int pop() {
        shift();
        return outStack.pop();
    }

    public int peek() {
        shift();
        return outStack.peek();
    }

    public boolean empty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }

    private void shift() {
        if (outStack.isEmpty()) {
            while (!inStack.isEmpty()) {
                outStack.push(inStack.pop());
            }
        }
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `push(x)`: $O(1)$ strictly.
  - `pop()`: $O(1)$ amortized. In the worst-case single call, transferring $N$ elements takes $O(N)$, but across $N$ pushes and $N$ pops, each element moves across stacks exactly once.
  - `peek()`: $O(1)$ amortized.
  - `empty()`: $O(1)$ strictly.
- **Space Complexity:** $O(N)$ auxiliary space to store $N$ elements partitioned between `in_stack` and `out_stack`.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Lazy Batch Inversion with Dual Stacks.
- **Trap:** Prematurely shifting elements back from `out_stack` to `in_stack`. Never push elements back from `out_stack` to `in_stack` after a peek or pop. Leaving them in `out_stack` preserves their exact FIFO order for all subsequent removals!