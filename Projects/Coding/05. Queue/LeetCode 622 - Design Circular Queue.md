---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 622: Design Circular Queue"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 622: Design Circular Queue

Below is a structured, interview-ready explanation of **LeetCode 622 – Design Circular Queue**, aligned with how such problems are typically evaluated.

---

## LeetCode 622: Design Circular Queue

### Problem Statement

Design your implementation of a **circular queue**.  
A circular queue is a linear data structure in which the operations are performed based on the **FIFO (First In First Out)**principle, and the last position is connected back to the first position to make a circle.

Implement the `MyCircularQueue` class:

* `MyCircularQueue(k)` — Initialize the queue with capacity `k`
* `enQueue(value)` — Insert an element into the circular queue. Return `true` if successful
* `deQueue()` — Delete an element from the circular queue. Return `true` if successful
* `Front()` — Get the front item. Return `-1` if empty
* `Rear()` — Get the last item. Return `-1` if empty
* `isEmpty()` — Check whether the circular queue is empty
* `isFull()` — Check whether the circular queue is full

---

## Key Observation

A **circular queue** solves the wasted-space problem of a normal array-based queue by **wrapping indices using modulo arithmetic**.

Key design idea:

* Use **fixed-size array**
* Maintain two pointers:

  * `front` → points to the first valid element
  * `rear` → points to the next insertion position
* Maintain a `size` counter to disambiguate **empty vs full**

---

## Queue Insights (Why Circular Queue Works)

### 1. Index Wrapping

Instead of shifting elements on dequeue:

```
next_index = (current_index + 1) % capacity
```

This ensures:

* O(1) enqueue
* O(1) dequeue
* No memory reallocation

---

### 2. Empty vs Full Condition

Using only `front` and `rear` is ambiguous.  
Therefore, we track **size** explicitly.

| Condition | Check |
| --- | --- |
| Empty | `size == 0` |
| Full | `size == capacity` |

---

### 3. Pointer Semantics

* `front` → index of the **current front element**
* `rear` → index where the **next element will be inserted**

---

## Python 3 Solution (with Typing)

```python
from typing import List

class MyCircularQueue:
    def __init__(self, k: int):
        self.capacity: int = k
        self.queue: List[int] = [0] * k
        self.front: int = 0
        self.rear: int = 0
        self.size: int = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False

        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.capacity
        self.size += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False

        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[self.front]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.queue[(self.rear - 1 + self.capacity) % self.capacity]

    def isEmpty(self) -> bool:
        return self.size == 0

    def isFull(self) -> bool:
        return self.size == self.capacity
```

---

## Worked-Out Example (Step-by-Step)

### Initialize

```
q = MyCircularQueue(3)
```

```
queue = [_, _, _]
front = 0, rear = 0, size = 0
```

---

### enQueue(1)

```
queue = [1, _, _]
front = 0, rear = 1, size = 1
```

---

### enQueue(2)

```
queue = [1, 2, _]
front = 0, rear = 2, size = 2
```

---

### enQueue(3)

```
queue = [1, 2, 3]
front = 0, rear = 0 (wrapped), size = 3
```

Queue is now **full**.

---

### deQueue()

```
front moves from 0 → 1
queue = [1, 2, 3]
size = 2
```

(Logically, `1` is removed.)

---

### enQueue(4)

```
queue = [4, 2, 3]
rear moves from 0 → 1
size = 3
```

**Logical order:** `2 → 3 → 4`

---

### Front() → `2`

### Rear() → `4`

---

## Time & Space Complexity

| Operation | Complexity |
| --- | --- |
| enQueue | O(1) |
| deQueue | O(1) |
| Front / Rear | O(1) |
| Space | O(k) |

---

## Interview Takeaway

* Circular queue = **fixed array + modulo arithmetic**
* Track `size` to avoid ambiguity
* `rear` always points to **next insert position**
* This pattern appears in **ring buffers**, **task schedulers**, and **stream processing systems**

If you want, I can also explain:

* Alternative design without `size`
* Why `(rear + 1) % k == front` works
* Comparison with Deque (LeetCode 641)