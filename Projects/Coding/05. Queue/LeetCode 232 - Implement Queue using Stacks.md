---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 232: Implement Queue using Stacks"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 232: Implement Queue using Stacks

Below is a structured, interview-grade explanation of **LeetCode 232 – Implement Queue using Stacks**, aligned with how this problem is typically evaluated.

---

## LeetCode 232: Implement Queue using Stacks

### Problem Statement

Implement a **FIFO queue** using only **LIFO stack operations**.

You must implement the following operations of a queue:

* `push(x)` — Push element `x` to the back of the queue
* `pop()` — Remove and return the element from the front of the queue
* `peek()` — Return the element at the front of the queue
* `empty()` — Return `True` if the queue is empty, else `False`

#### Constraints

* You may use **only standard stack operations**:

  * `push`, `pop`, `peek/top`, `isEmpty`
* You may assume all operations are valid (e.g., no `pop` on empty queue)

---

## Key Observation

A **stack reverses order**, while a **queue preserves order**.

To simulate FIFO behavior using LIFO structures:

* **Use two stacks**

  * One stack for **incoming elements**
  * One stack for **outgoing elements**

Reversing elements **once** and reusing them allows us to maintain queue order efficiently.

---

## Core Queue Insight (Amortized Optimization)

Instead of reversing elements on every operation:

* **Push:** Always push to `in_stack`
* **Pop / Peek:**

  * If `out_stack` is empty:

    * Transfer all elements from `in_stack` to `out_stack`
    * This reverses order and exposes the oldest element on top
  * Pop / peek from `out_stack`

### Why this is efficient

* Each element moves **at most twice**:

  1. `in_stack → out_stack`
  2. `out_stack → removed`
* Therefore:

  * **Amortized O(1)** time per operation
  * Worst-case transfer cost is spread across operations

---

## Python 3 Solution (with Typing)

```python
from typing import List

class MyQueue:
    def __init__(self) -> None:
        self.in_stack: List[int] = []
        self.out_stack: List[int] = []

    def push(self, x: int) -> None:
        self.in_stack.append(x)

    def pop(self) -> int:
        self._shift_stacks()
        return self.out_stack.pop()

    def peek(self) -> int:
        self._shift_stacks()
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

    def _shift_stacks(self) -> None:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
```

---

## Worked-Out Example (Step-by-Step)

### Operations Sequence

```
push(1)
push(2)
peek()
pop()
empty()
```

---

### Step 1: `push(1)`

```
in_stack  = [1]
out_stack = []
```

---

### Step 2: `push(2)`

```
in_stack  = [1, 2]
out_stack = []
```

---

### Step 3: `peek()`

* `out_stack` is empty → transfer elements

Transfer process:

```
in_stack.pop() → 2 → out_stack
in_stack.pop() → 1 → out_stack
```

Result:

```
in_stack  = []
out_stack = [2, 1]   # top is 1 (front of queue)
```

Peek returns:

```
1
```

---

### Step 4: `pop()`

* `out_stack` already has elements

```
pop() → 1
```

Stacks now:

```
in_stack  = []
out_stack = [2]
```

---

### Step 5: `empty()`

```
in_stack  = []
out_stack = [2]
```

Result:

```
False
```

---

## Time and Space Complexity

| Operation | Time Complexity |
| --- | --- |
| `push` | O(1) |
| `pop` | Amortized O(1) |
| `peek` | Amortized O(1) |
| `empty` | O(1) |

**Space Complexity:**  
O(n) — total elements stored across two stacks

---

## Interview Takeaway

* This is a **classic amortized analysis problem**
* Key idea: *lazy transfer* instead of eager reversal
* Demonstrates understanding of:

  * Stack vs Queue behavior
  * Amortized complexity
  * Data structure simulation

If you want, I can also:

* Explain **why single-stack solutions fail**
* Compare this with **LeetCode 225 (Stack using Queues)**
* Provide a **dry-run table format** suitable for whiteboard interviews