---
date: "2025-12-19"
type: leetcode-solution
category: "Stack"
folder: "04. Stack"
title: "LeetCode 225: Implement Stack using Queues"
tags:
  - leetcode
  - coding
  - stack
  - queue
  - design
  - amazon
  - google
---

# LeetCode 225: Implement Stack using Queues

**Target Companies:** Amazon, Google, Microsoft, Bloomberg, Apple  
**Difficulty:** Easy  
**Topic:** Stack / Queue / Data Structure Design

---

### Problem Statement

Implement a last-in-first-out (LIFO) stack using only two queues. The implemented stack should support all the functions of a normal stack (`push`, `top`, `pop`, and `empty`).

Implement the `MyStack` class:
- `void push(int x)`: Pushes element `x` to the top of the stack.
- `int pop()`: Removes the element on the top of the stack and returns it.
- `int top()`: Returns the element on the top of the stack.
- `boolean empty()`: Returns `true` if the stack is empty, `false` otherwise.

**Notes:**
- You must use **only** standard operations of a queue — which means only `push to back`, `peek/pop from front`, `size`, and `is empty` operations are valid.
- Depending on your language, the queue may not be supported natively. You may simulate a queue using a list or deque (double-ended queue), as long as you use only standard queue operations.

---

### Input & Output Formats & Constraints

- **Input:**
  - Method calls: `["MyStack", "push", "push", "top", "pop", "empty"]`, with corresponding arguments.
- **Output:**
  - Standard return values for each method call.
- **Constraints:**
  - $1 \le x \le 9$.
  - At most $100$ calls will be made to `push`, `pop`, `top`, and `empty`.
  - All calls to `pop` and `top` are valid (invoked only on non-empty stacks).
- **Follow-up:**
  - Can you implement the stack using only **one queue**?

---

### Key Idea & Intuition

A **stack** is Last-In-First-Out (LIFO), whereas a **queue** is First-In-First-Out (FIFO). In a queue, the earliest inserted element sits at the front. To mimic a stack, we must arrange that the **most recently inserted element always sits at the front** of the queue.

#### The Single-Queue Rotation Invariant
We can achieve a full LIFO stack using **a single queue**:
1. When `push(x)` is called:
   - Enqueue $x$ to the back of the queue.
   - Let $K = \text{queue.size} - 1$ (the number of elements previously present).
   - Dequeue from the front and immediately re-enqueue to the back $K$ times.
2. Invariant maintained:
   - The newly pushed element $x$ is rotated all the way to the **front** of the queue!
   - Elements behind $x$ preserve their exact reverse insertion order.
3. Consequently:
   - `pop()` is simply `queue.dequeue()` in $\mathcal{O}(1)$ time.
   - `top()` is simply `queue.peek()` in $\mathcal{O}(1)$ time.
   - `empty()` is `queue.isEmpty()` in $\mathcal{O}(1)$ time.

---

### Solution Approach (Step-by-Step)

1. **State:**
   - A single FIFO queue `q`.
2. **`push(x)`:**
   - Append `x` to `q`.
   - Rotate the queue: loop `len(q) - 1` times:
     - Pop element from front.
     - Append it back to the tail.
3. **`pop()`:**
   - Dequeue and return front element of `q`.
4. **`top()`:**
   - Return front element of `q` without removing it.
5. **`empty()`:**
   - Return `q.empty()`.

---

### Visual Algorithm Walkthrough

Suppose we perform: `push(1) -> push(2) -> push(3) -> pop()`

```
Initial: q = []

1. push(1):
   - Enqueue 1: [1]
   - Rotation count: len - 1 = 0.
   Queue: Front -> [1] <- Back

2. push(2):
   - Enqueue 2 to back: Front -> [1, 2] <- Back
   - Rotation count: len - 1 = 1.
   - Pop 1 from front, enqueue to back:
   Queue: Front -> [2, 1] <- Back
   (Notice: 2 is now at the FRONT, exactly matching stack top!)

3. push(3):
   - Enqueue 3 to back: Front -> [2, 1, 3] <- Back
   - Rotation count: len - 1 = 2.
   - 1st rotation: pop 2, push 2 -> [1, 3, 2]
   - 2nd rotation: pop 1, push 1 -> [3, 2, 1]
   Queue: Front -> [3, 2, 1] <- Back
   (3 is at the FRONT!)

4. pop():
   - Dequeue front element: 3.
   Queue becomes: Front -> [2, 1] <- Back
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Sequence

- **Input Commands:**
  `["MyStack", "push", "push", "top", "pop", "empty"]`
  `[[], [1], [2], [], [], []]`
- **Step Tracing:**

| Call | Queue State (Front $\to$ Back) | Return Value | Stack Meaning |
|:---:|:---:|:---:|:---:|
| `push(1)` | `[1]` | `null` | Stack: `[1]` |
| `push(2)` | `[2, 1]` | `null` | Stack: `[1, 2]` |
| `top()` | `[2, 1]` | `2` | Peek top element `2` |
| `pop()` | `[1]` | `2` | Remove and return `2` |
| `empty()` | `[1]` | `false` | Stack has element 1 |

#### Example 2: Single Element Multiple Calls

- `push(5)` $\implies$ `q = [5]`
- `empty()` $\implies$ `false`
- `pop()` $\implies$ `5`, `q = []`
- `empty()` $\implies$ `true`

---

### Multi-Language Implementations

#### Python 3

```python
from collections import deque
from typing import Deque

class MyStack:
    def __init__(self) -> None:
        """Initialize data structure using a single queue."""
        self.q: Deque[int] = deque()

    def push(self, x: int) -> None:
        """Push element x onto stack in O(N) time by rotating queue."""
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        """Removes the element on top of the stack and returns it in O(1)."""
        return self.q.popleft()

    def top(self) -> int:
        """Get the top element in O(1)."""
        return self.q[0]

    def empty(self) -> bool:
        """Returns whether the stack is empty in O(1)."""
        return len(self.q) == 0
```

#### C++17

```cpp
#include <queue>

class MyStack {
private:
    std::queue<int> q;

public:
    MyStack() {}

    void push(int x) {
        q.push(x);
        int rotations = static_cast<int>(q.size()) - 1;
        while (rotations-- > 0) {
            q.push(q.front());
            q.pop();
        }
    }

    int pop() {
        int val = q.front();
        q.pop();
        return val;
    }

    int top() {
        return q.front();
    }

    bool empty() {
        return q.empty();
    }
};
```

#### Java

```java
import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
    private Queue<Integer> q;

    public MyStack() {
        q = new LinkedList<>();
    }

    public void push(int x) {
        q.offer(x);
        int rotations = q.size() - 1;
        while (rotations-- > 0) {
            q.offer(q.poll());
        }
    }

    public int pop() {
        return q.poll();
    }

    public int top() {
        return q.peek();
    }

    public boolean empty() {
        return q.isEmpty();
    }
}
```

---

### Complexity Analysis

- **Time Complexity:**
  - `push(x)`: $\mathcal{O}(N)$ where $N$ is the current number of elements in the stack (rotates $N-1$ times).
  - `pop()`: $\mathcal{O}(1)$ directly removes front of the queue.
  - `top()`: $\mathcal{O}(1)$ inspects front of the queue.
  - `empty()`: $\mathcal{O}(1)$ checks size.
- **Space Complexity:** $\mathcal{O}(N)$
  - Stores exactly $N$ elements across a single queue.

---

### Takeaway Pattern & Interview Traps

1. **Trade-off: Costly Push vs Costly Pop:**
   - **Costly Push ($\mathcal{O}(N)$ push, $\mathcal{O}(1)$ pop):** Described above using queue rotation. Preferred when read/pop operations are more frequent.
   - **Costly Pop ($\mathcal{O}(1)$ push, $\mathcal{O}(N)$ pop):** In push, just append to queue. In pop, rotate $N-1$ elements to reach the tail.
2. **Strict Queue Primitives:**
   - In an interview, verify with the interviewer that you cannot use deque-specific bidirectional methods (`pop()`, `peek_back()`). You must strictly use FIFO operations: `enqueue` at tail, `dequeue` at head, `peek` at head.