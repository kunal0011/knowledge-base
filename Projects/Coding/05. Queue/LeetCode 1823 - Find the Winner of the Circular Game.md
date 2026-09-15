---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 1823: Find the Winner of the Circular Game"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 1823: Find the Winner of the Circular Game

Below is a structured, interview-ready explanation of **LeetCode 1823 – Find the Winner of the Circular Game**, following the exact format you requested.

---

## LeetCode 1823 – Find the Winner of the Circular Game

---

### **Problem Statement**

There are **n friends** standing in a circle, numbered from **1 to n**.

Starting from friend **1**, you count **k** friends clockwise (including the current one).  
The **k-th friend is eliminated** from the circle.

After elimination, counting resumes from the **next remaining friend**.

This process continues until **only one friend remains**.

**Return the number of the winner.**

---

### **Key Observation**

This is a classic **Josephus Problem** variant.

Important characteristics:

* Players are arranged in a **circular structure**
* Every **k-th** element is removed
* Order must be preserved after each removal

A **queue (FIFO)** is ideal because:

* We rotate the circle by moving elements from front to back
* We eliminate by popping from the front

---

### **Queue Insight (Core Idea)**

For each elimination:

1. Rotate the queue **k − 1 times**

   * Pop front
   * Push it to the back
2. Remove the **k-th** friend (pop once)
3. Repeat until only one element remains

This simulates clockwise movement perfectly.

---

### **Algorithm Steps**

1. Initialize a queue with numbers `1` to `n`
2. While queue size > 1:

   * Rotate `k - 1` times
   * Remove the front element
3. Return the remaining element

---

### **Time & Space Complexity**

* **Time Complexity:** `O(n × k)`
* **Space Complexity:** `O(n)`

Note: Although there is a mathematical `O(n)` solution, this queue simulation is clearer and often preferred in interviews.

---

### **Python 3 Solution (with typing)**

```python
from collections import deque
from typing import Deque

class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        queue: Deque[int] = deque(range(1, n + 1))

        while len(queue) > 1:
            # Rotate k-1 times
            for _ in range(k - 1):
                queue.append(queue.popleft())

            # Eliminate k-th friend
            queue.popleft()

        return queue[0]
```

---

### **Worked Out Example**

#### Input

```
n = 5, k = 2
```

#### Initial Queue

```
[1, 2, 3, 4, 5]
```

---

#### Step-by-Step Simulation

| Round | Queue Before | Rotation (k-1=1) | Eliminated | Queue After |
| --- | --- | --- | --- | --- |
| 1 | [1, 2, 3, 4, 5] | move 1 → back | 2 | [3, 4, 5, 1] |
| 2 | [3, 4, 5, 1] | move 3 → back | 4 | [5, 1, 3] |
| 3 | [5, 1, 3] | move 5 → back | 1 | [3, 5] |
| 4 | [3, 5] | move 3 → back | 5 | [3] |

---

### **Final Answer**

```
Winner = 3
```

---

### **Why Queue Works Best Here**

* Naturally models **circular traversal**
* Clean elimination logic
* No manual index handling
* Easy to reason about and debug

---

If you want next:

* **O(n) mathematical solution (Josephus recurrence)**
* **Dry-run with larger k**
* **Visualization with circular index math**

Tell me how deep you want to go.