---
date: "2025-12-18"
type: leetcode-solution
category: "Heap & Priority Queue"
folder: "09. Heap & Priority Queue"
title: "LeetCode 621: Task Scheduler"
tags:
  - leetcode
  - coding
  - heap-and-priority-queue
---

# LeetCode 621: Task Scheduler

Below is a **complete, structured explanation of LeetCode 621 – Task Scheduler**, aligned with how this problem is typically reasoned about in interviews.

---

## 1. Problem Statement

You are given a list of tasks represented by capital letters `A` to `Z`.  
Each task takes **exactly 1 unit of time** to execute.

You are also given a non-negative integer `n` representing the **cooldown period** between two identical tasks.

**Constraint**  
After executing a task, you must wait for at least `n` units of time before executing the **same task again**.

Idle time is allowed.

**Objective**  
Return the **minimum number of time units** required to finish all tasks.

---

### Example

```text
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
```

---

## 2. Key Observations

### Observation 1: Greedy scheduling

* Tasks with **higher frequency** are more restrictive.
* We should always execute the task that has the **largest remaining count** whenever possible.

This immediately suggests a **max-heap / priority queue**.

---

### Observation 2: Cooling constraint creates “cycles”

Each cycle has length:

```
cycle_length = n + 1
```

Why?

* After executing a task, you need `n` slots before repeating it.
* Those slots can be filled by other tasks or idle time.

---

### Observation 3: Idle time is forced, not optional

Idle time occurs **only if there are fewer distinct tasks than required to fill the cycle**.

We **never choose idle** if a task is available.

---

## 3. Priority Queue Technique

### Data Structures Used

1. **Max Heap** (Priority Queue)

   * Stores task frequencies
   * Always pick the task with the highest remaining count
2. **Temporary list**

   * Holds tasks executed in the current cycle
   * Reinsert them into the heap after decrementing counts

---

### High-Level Algorithm

1. Count task frequencies.
2. Push frequencies into a **max heap**.
3. While heap is not empty:

   * Try to execute up to `n + 1` tasks.
   * For each execution:
   * Pop highest frequency task
   * Decrement and store it temporarily
   * Increase time by 1
   * If heap becomes empty early, stop cycle
4. Push remaining tasks from temp list back into heap.
5. If heap is not empty, add idle time to complete the cycle.

---

## 4. Python 3 Solution (with typing)

```python
from typing import List
import heapq
from collections import Counter

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)

        # Max heap (use negative values)
        max_heap = [-count for count in freq.values()]
        heapq.heapify(max_heap)

        time = 0

        while max_heap:
            temp = []
            cycle = n + 1

            for _ in range(cycle):
                if max_heap:
                    count = heapq.heappop(max_heap)
                    temp.append(count + 1)  # decrement frequency
                    time += 1
                else:
                    break

            for count in temp:
                if count < 0:
                    heapq.heappush(max_heap, count)

            # If tasks still remain, account for idle time
            if max_heap:
                time += cycle - len(temp)

        return time
```

---

## 5. Worked Example (Step-by-Step)

### Example 1

```
tasks = ["A","A","A","B","B","B"], n = 2
```

### Frequency

```
A → 3
B → 3
```

### Initial Heap

```
[-3, -3]
```

---

### Cycle 1 (length = 3)

| Slot | Action |
| --- | --- |
| 1 | A (remaining 2) |
| 2 | B (remaining 2) |
| 3 | idle |

Heap after reinsertion:

```
[-2, -2]
```

Time = 3

---

### Cycle 2

| Slot | Action |
| --- | --- |
| 4 | A (remaining 1) |
| 5 | B (remaining 1) |
| 6 | idle |

Heap:

```
[-1, -1]
```

Time = 6

---

### Cycle 3

| Slot | Action |
| --- | --- |
| 7 | A |
| 8 | B |

Heap empty → stop  
Final time = **8**

---

## 6. Second Example

```
tasks = ["A","A","A","B","B","B"], n = 0
```

No cooldown constraint.

Execution:

```
A B A B A B
```

Result:

```
6
```

---

## 7. Time and Space Complexity

### Time Complexity

```
O(T log K)
```

* `T` = number of tasks
* `K` = number of distinct task types (≤ 26)

### Space Complexity

```
O(K)
```

---

## 8. Why This Approach Is Optimal

* Greedy ensures most constrained tasks are handled first.
* Priority Queue ensures optimal selection at every step.
* Idle time is introduced **only when forced by constraints**, never voluntarily.

---

If you want, I can also:

* Show the **mathematical formula-based solution**
* Compare **heap vs formula approach**
* Draw a **cycle-based execution diagram**
* Explain why this is equivalent to CPU scheduling with cooling constraints

Just tell me how deep you want to go.