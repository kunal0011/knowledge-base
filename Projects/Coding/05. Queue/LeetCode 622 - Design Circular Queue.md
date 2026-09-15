---
date: "2026-09-15"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 622: Design Circular Queue"
tags:
  - leetcode
  - coding
  - queue
  - circular-buffer
  - data-structure-design
  - amazon
  - google
---

# LeetCode 622: Design Circular Queue

**Target Companies:** Amazon, Google, Microsoft, Meta  
**Difficulty:** Medium  
**Topic:** Array-Based Circular FIFO Queue via Modulo Arithmetic  

---

### Problem Statement

Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle, and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Implementation details for `MyCircularQueue`:
- `MyCircularQueue(k)`: Initializes the object with the size of the queue to be $k$.
- `boolean enQueue(int value)`: Inserts an element into the circular queue. Return `true` if the operation is successful.
- `boolean deQueue()`: Deletes an element from the circular queue. Return `true` if the operation is successful.
- `int Front()`: Gets the front item from the queue. If the queue is empty, return `-1`.
- `int Rear()`: Gets the last item from the queue. If the queue is empty, return `-1`.
- `boolean isEmpty()`: Checks whether the circular queue is empty or not.
- `boolean isFull()`: Checks whether the circular queue is full or not.

You must solve the problem without using the built-in queue data structure in your programming language.

---

### Input & Output Formats & Constraints

- **Input:** Sequence of operations: `["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]`
- **Output:** Returns from operations (`null`, `true`, `true`, `true`, `false`, `3`, `true`, `true`, `true`, `4`)
- **Constraints:**
  - $1 \le k \le 1000$
  - $0 \le \text{value} \le 1000$
  - At most $3000$ calls will be made to `enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, and `isFull`.

---

### Key Idea & Intuition

To avoid linear element shifts when dequeuing, we maintain a static array of size $k$ and two logical pointers:
1. `head`: Index of the current front element.
2. `tail`: Index where the next inserted element will be placed.
3. `count`: Explicit count of elements currently present in the queue.
   - Using an explicit `count` cleanly disambiguates the boundary condition where `head == tail` (which occurs in both an empty queue and a full queue without a counter).

#### Modulo Arithmetic
- Enqueue: `data[tail] = value; tail = (tail + 1) % k; count++`.
- Dequeue: `head = (head + 1) % k; count--`.
- Front: `data[head]`.
- Rear: The most recently inserted element is located at index `(tail - 1 + k) % k`.

---

### Solution Approach (Step-by-Step)

1. `__init__(k)`: Initialize array of size $k$, `head = 0`, `tail = 0`, `count = 0`, `capacity = k`.
2. `enQueue(value)`: If `isFull()`, return `False`. Else write to `data[tail]`, update `tail = (tail + 1) % capacity`, increment `count`, return `True`.
3. `deQueue()`: If `isEmpty()`, return `False`. Else advance `head = (head + 1) % capacity`, decrement `count`, return `True`.
4. `Front()`: If `isEmpty()`, return `-1`. Else return `data[head]`.
5. `Rear()`: If `isEmpty()`, return `-1`. Else return `data[(tail - 1 + capacity) % capacity]`.
6. `isEmpty()`: Return `count == 0`.
7. `isFull()`: Return `count == capacity`.

---

### Visual Algorithm Walkthrough

```
Initialize: MyCircularQueue(3)
Array: [ _,  _,  _ ]
head = 0, tail = 0, count = 0

1. enQueue(1):
   Array: [ 1,  _,  _ ], head = 0, tail = 1, count = 1
2. enQueue(2):
   Array: [ 1,  2,  _ ], head = 0, tail = 2, count = 2
3. enQueue(3):
   Array: [ 1,  2,  3 ], head = 0, tail = 0 (wrapped!), count = 3 (isFull() = True)
4. enQueue(4):
   isFull() is True -> Rejected! Returns False.

5. Rear():
   (tail - 1 + 3) % 3 = (0 - 1 + 3) % 3 = 2 -> Array[2] = 3. Returns 3.

6. deQueue():
   head = (0 + 1) % 3 = 1, count = 2. Logically removes 1.
   Array logically: [ _,  2,  3 ]

7. enQueue(4):
   tail was 0. Place 4 at Array[0].
   Array physically: [ 4,  2,  3 ]
   tail = (0 + 1) % 3 = 1, count = 3.
   Logically ordered: Front is at head=1 (val 2), followed by index 2 (val 3), then index 0 (val 4).
   Rear is Array[(1 - 1 + 3) % 3] = Array[0] = 4.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Full Lifecycle Execution
```text
Commands: ["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
Args:     [[3], [1], [2], [3], [4], [], [], [], [4], []]
Trace:
- MyCircularQueue(3): capacity = 3
- enQueue(1) -> true
- enQueue(2) -> true
- enQueue(3) -> true
- enQueue(4) -> false (queue is full)
- Rear()     -> 3
- isFull()   -> true
- deQueue()  -> true (removes 1)
- enQueue(4) -> true (inserts at slot 0)
- Rear()     -> 4
Output: [null, true, true, true, false, 3, true, true, true, 4]
```

#### Example 2: Underflow Protection
```text
Commands: ["MyCircularQueue", "deQueue", "Front", "Rear", "isEmpty"]
Args:     [[2], [], [], [], []]
Trace:
- deQueue() -> false (empty)
- Front()   -> -1
- Rear()    -> -1
- isEmpty() -> true
```

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
class MyCircularQueue:
    def __init__(self, k: int) -> None:
        self.capacity: int = k
        self.data: list[int] = [0] * k
        self.head: int = 0
        self.tail: int = 0
        self.count: int = 0

    def enQueue(self, value: int) -> bool:
        if self.isFull():
            return False
        self.data[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.count += 1
        return True

    def deQueue(self) -> bool:
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return True

    def Front(self) -> int:
        if self.isEmpty():
            return -1
        return self.data[self.head]

    def Rear(self) -> int:
        if self.isEmpty():
            return -1
        return self.data[(self.tail - 1 + self.capacity) % self.capacity]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.capacity
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>

class MyCircularQueue {
private:
    std::vector<int> data;
    int head;
    int tail;
    int count;
    int capacity;

public:
    MyCircularQueue(int k) 
        : data(k), head(0), tail(0), count(0), capacity(k) {}

    bool enQueue(int value) {
        if (isFull()) return false;
        data[tail] = value;
        tail = (tail + 1) % capacity;
        count++;
        return true;
    }

    bool deQueue() {
        if (isEmpty()) return false;
        head = (head + 1) % capacity;
        count--;
        return true;
    }

    int Front() {
        if (isEmpty()) return -1;
        return data[head];
    }

    int Rear() {
        if (isEmpty()) return -1;
        return data[(tail - 1 + capacity) % capacity];
    }

    bool isEmpty() {
        return count == 0;
    }

    bool isFull() {
        return count == capacity;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class MyCircularQueue {
    private final int[] data;
    private int head;
    private int tail;
    private int count;
    private final int capacity;

    public MyCircularQueue(int k) {
        this.data = new int[k];
        this.head = 0;
        this.tail = 0;
        this.count = 0;
        this.capacity = k;
    }

    public boolean enQueue(int value) {
        if (isFull()) return false;
        data[tail] = value;
        tail = (tail + 1) % capacity;
        count++;
        return true;
    }

    public boolean deQueue() {
        if (isEmpty()) return false;
        head = (head + 1) % capacity;
        count--;
        return true;
    }

    public int Front() {
        if (isEmpty()) return -1;
        return data[head];
    }

    public int Rear() {
        if (isEmpty()) return -1;
        return data[(tail - 1 + capacity) % capacity];
    }

    public boolean isEmpty() {
        return count == 0;
    }

    public boolean isFull() {
        return count == capacity;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $O(1)$ strictly for all operations (`enQueue`, `deQueue`, `Front`, `Rear`, `isEmpty`, `isFull`).
- **Space Complexity:** $O(k)$ fixed space allocated once during instantiation.

---

### Takeaway Pattern & Interview Traps

- **Pattern:** Circular Array / Ring Buffer Index Wrapping.
- **Trap:** Calculating `Rear()`: Since `tail` points to the next insertion slot, the rear element is at `tail - 1`. If `tail == 0`, `tail - 1 = -1`. Modulo in C++ and Java with negative numbers produces `-1`. Always add `capacity` before modulo: `(tail - 1 + capacity) % capacity`.