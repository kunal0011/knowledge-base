---
date: "2025-12-16"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 621: Task Scheduler"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 621: Task Scheduler

**LeetCode 621 – Task Scheduler**, structured exactly as requested.

---

## 1. Problem Statement

You are given an array of characters `tasks`, where each character represents a CPU task.  
Each task takes **1 unit of time** to execute.

There is a **cooldown period `n`**, meaning **the same task cannot be executed again until `n` time units have passed** since its last execution.

You may insert **idle intervals** if necessary.

### Objective

Return the **minimum number of time units** required to finish all tasks.

---

### Example

```
tasks = ["A","A","A","B","B","B"]
n = 2
Output = 8
```

---

## 2. Key Observations (Critical Insights)

### Observation 1: Order does not matter

Only the **frequency** of tasks matters, not their original order.

---

### Observation 2: Most frequent task dominates the schedule

Let:

* `max_freq` = maximum frequency of any task
* `count_max` = number of tasks with frequency = `max_freq`

These tasks create **forced gaps** because they must be separated by `n` units.

---

### Observation 3: Greedy framing using blocks

Think of the most frequent task as defining a structure:

```
A _ _ A _ _ A
```

If `A` appears `max_freq` times, it creates:

```
(max_freq - 1) blocks
```

Each block must have **at least `n` slots** between same tasks.

---

### Observation 4: Idle time is only needed if tasks are insufficient

If other tasks can fill the gaps → no idle time needed.

---

## 3. Greedy Strategy (Core Trick)

### Step 1: Count frequencies

Use a frequency map.

---

### Step 2: Identify dominant tasks

```
max_freq = maximum task frequency
count_max = number of tasks with max_freq
```

---

### Step 3: Compute minimum required length

The greedy formula:

```
(min_time) = (max_freq - 1) * (n + 1) + count_max
```

Explanation:

* `(max_freq - 1)` → number of gaps
* `(n + 1)` → each block has 1 task + n cooldown slots
* `count_max` → last block’s tasks

---

### Step 4: Compare with total tasks

Final answer is:

```
max(min_time, len(tasks))
```

Why?

* If enough tasks exist to fill idle slots, schedule is compact
* Otherwise, idle time is unavoidable

---

## 4. Python 3 Solution (With Typing)

```python
from typing import List
from collections import Counter

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)

        max_freq = max(freq.values())
        count_max = sum(1 for v in freq.values() if v == max_freq)

        # Greedy formula
        min_time = (max_freq - 1) * (n + 1) + count_max

        return max(min_time, len(tasks))
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```
tasks = ["A","A","A","B","B","B"]
n = 2
```

---

### Step 1: Frequency Count

```
A → 3
B → 3
```

```
max_freq = 3
count_max = 2   # A and B
```

---

### Step 2: Build scheduling blocks

We have `max_freq - 1 = 2` gaps.

```
A _ _ A _ _ A
```

Each `_ _` requires at least `n = 2` slots.

---

### Step 3: Apply formula

```
(min_time) = (3 - 1) * (2 + 1) + 2
           = 2 * 3 + 2
           = 8
```

---

### Step 4: Compare with total tasks

```
len(tasks) = 6
answer = max(8, 6) = 8
```

---

### Step 5: One valid schedule

```
Time:  1  2  3  4  5  6  7  8
Task:  A  B  idle  A  B  idle  A  B
```

✔ Cooldown respected  
✔ Minimum time achieved

---

## 6. Edge Case Example (No Idle Needed)

```
tasks = ["A","A","A","B","B","B","C","C"]
n = 2
```

Here, extra tasks fill the gaps:

```
A B C A B C A B
```

Total tasks = 8  
Greedy formula gives 8  
Final answer = **8**

---

## 7. Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | O(N) |
| Space | O(1) (max 26 tasks) |

---

## 8. Interview Takeaways

* This is **not simulation**
* This is a **math + greedy counting problem**
* Focus on:

  * Dominant frequency
  * Forced idle gaps
  * Filling gaps with remaining tasks

If you want, I can also:

* Derive this formula visually
* Compare with priority-queue simulation
* Show why naive greedy fails

Just tell me.