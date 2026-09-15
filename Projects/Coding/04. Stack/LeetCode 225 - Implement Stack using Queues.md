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
---

# LeetCode 225: Implement Stack using Queues

Below is a complete, structured explanation for **LeetCode 225 – Implement Stack using Queues**, aligned with your usual format.

---

## LeetCode 225: Implement Stack using Queues

### Problem Statement

Implement a **Last-In-First-Out (LIFO) stack** using only **queues**.

You must implement the following operations:

* `push(x)`: Push element `x` onto stack.
* `pop()`: Removes the element on top of the stack and returns it.
* `top()`: Get the top element.
* `empty()`: Returns `true` if the stack is empty, `false` otherwise.

### Constraints

* You may use only standard operations of a queue:

  * `push to back`
  * `pop from front`
  * `peek from front`
  * `size`
* You may assume all operations are valid (e.g., `pop()` will not be called on an empty stack).

---

## Key Observation

A **stack is LIFO**, while a **queue is FIFO**.

To simulate a stack using queues, we must **reorder elements inside the queue** so that:

* The **most recently pushed element always comes to the front** of the queue.

---

## Stack Key Insight (Core Idea)

### Single Queue Rotation Technique

Use **one queue** and enforce this invariant:

> **After every `push`, the newly pushed element must be at the front of the queue.**

#### How?

1. Push the new element to the back.
2. Rotate the queue by moving all previous elements behind it.

This ensures:

* `pop()` → front of queue
* `top()` → front of queue

---

## Algorithm (Single Queue)

### `push(x)`

* Add `x` to the queue.
* Rotate the queue `size - 1` times:

  * Pop front
  * Push it back

### `pop()`

* Pop from front

### `top()`

* Peek front

### `empty()`

* Check if queue is empty

---

## Python 3 Solution (with typing)

```python
from collections import deque
from typing import Deque

class MyStack:
    def __init__(self) -> None:
        self.q: Deque[int] = deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        # Rotate the queue to move x to the front
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q
```

---

## Worked-Out Example

### Operations Sequence

```
push(1)
push(2)
push(3)
top()
pop()
top()
```

---

### Step-by-Step Execution

#### Initial State

```
Queue: []
```

---

### `push(1)`

* Append `1`
* No rotation needed

```
Queue: [1]
Stack view: top → 1
```

---

### `push(2)`

* Append `2`
* Rotate once

```
Before rotation: [1, 2]
After rotation:  [2, 1]

Stack view: top → 2
```

---

### `push(3)`

* Append `3`
* Rotate twice

```
Before rotation: [2, 1, 3]

Rotation 1: [1, 3, 2]
Rotation 2: [3, 2, 1]

Stack view: top → 3
```

---

### `top()`

```
Return: 3
Queue: [3, 2, 1]
```

---

### `pop()`

```
Remove front → 3
Queue: [2, 1]
Stack view: top → 2
```

---

### `top()`

```
Return: 2
```

---

## Time & Space Complexity

| Operation | Time Complexity |
| --- | --- |
| `push` | O(n) |
| `pop` | O(1) |
| `top` | O(1) |
| `empty` | O(1) |

**Space Complexity:** O(n)

---

## Summary

* The trick is **rotating the queue after each push**
* This guarantees stack behavior using FIFO primitives
* Clean and optimal under given constraints
* Frequently tested concept for **data structure simulation**

If you want, I can also explain:

* Two-queue approach
* Why rotation works mathematically
* How this compares to LeetCode 232 (Queue using Stack)