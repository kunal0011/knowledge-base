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
  - greedy
  - math
  - amazon
  - google
---

# LeetCode 621: Task Scheduler

**Target Companies:** Amazon (Top Classic), Google, Meta, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Priority Queue (Max-Heap Simulation) / Greedy Math Scheduling

---

### Problem Statement

Given a characters array `tasks`, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer `n` that represents the cooldown period between two **same tasks** (the same letter in the CPU). That is, there must be at least `n` units of time between any two occurrences of the same task.

Return *the least number of units of times that the CPU will take to finish all the given tasks*.

---

### Input & Output Formats & Constraints

- **Input:**
  - `tasks`: `List[str]`, where $1 \le \text{tasks.length} \le 10^4$. Each task is an uppercase English letter (`'A'`–`'Z'`).
  - `n`: `int`, cooldown interval ($0 \le n \le 100$).
- **Output:**
  - `int`: Minimum total time units (tasks + idle slots).
- **Constraints:**
  - Cooldown applies strictly between identical tasks. Different tasks require 0 cooldown.

---

### Key Idea & Intuition

The execution bottleneck is entirely determined by the task(s) with the **highest frequency** ($max\_freq$).
Suppose task `'A'` appears $max\_freq = 3$ times, and $n = 2$:
```
A _ _ A _ _ A
```
Between each occurrence of `'A'`, there must be at least $n$ slots filled either by other tasks or by idle periods.

This problem can be understood via two paradigms:

#### Paradigm 1: Max-Heap Simulation ($\mathcal{O}(T \log 26) = \mathcal{O}(T)$ Time)
1. Store frequencies in a **Max-Heap**.
2. Work in rounds of cycle length $cycle = n + 1$:
   - In each cycle, pop up to $n + 1$ most frequent tasks from the heap.
   - Decrement each task's frequency and temporarily hold it in a list `temp`.
   - If tasks still remain after the cycle, pad the cycle with idle slots up to $n + 1$.
   - Push all tasks in `temp` with remaining counts back into the heap.
3. Repeat until the heap is empty.

#### Paradigm 2: Closed-Form Greedy Formula ($\mathcal{O}(T)$ Time, $\mathcal{O}(1)$ Space - Optimal)
- Let $M$ be the maximum frequency of any task: $M = \max(\text{counts})$.
- There are $M - 1$ full chunks of size $n + 1$, plus a final chunk containing all tasks that tie for maximum frequency:
  $$\text{Chunks} = (M - 1) \times (n + 1)$$
- Count how many distinct task types have frequency equal to $M$: let this be $C_{max}$.
- The total slots needed by the most frequent task(s) is:
  $$\text{Required} = (M - 1) \times (n + 1) + C_{max}$$
- If there are enough other tasks to fill all idle slots, no idle time is needed, so the answer is simply the total number of tasks:
  $$\text{Answer} = \max(\text{len}(tasks), (M - 1) \times (n + 1) + C_{max})$$

---

### Solution Approach (Step-by-Step)

#### Approach 1: Closed-Form Formula ($\mathcal{O}(1)$ Auxiliary Space)
1. Count frequencies of tasks across `'A'`–`'Z'`.
2. Find $M = \max(freq)$.
3. Count how many tasks have frequency equal to $M$: $C_{max} = \text{sum}(1 \text{ for } f \text{ in } freq \text{ if } f == M)$.
4. Return $\max(\text{len}(tasks), (M - 1) \times (n + 1) + C_{max})$.

#### Approach 2: Max-Heap Simulation
1. `max_heap = [-count for count in Counter(tasks).values()]`.
2. `heapify(max_heap)`.
3. `time = 0`.
4. While `max_heap`:
   - `temp = []`
   - For $step$ in $0 \dots n$:
     - If `max_heap`:
       - `count = heappop(max_heap)`
       - If `count + 1 < 0`: `temp.append(count + 1)`
       - `time += 1`
     - Else if `not temp`: break
     - Else: `time += 1` (idle slot)
   - For `c` in `temp`: `heappush(max_heap, c)`.
5. Return `time`.

---

### Visual Algorithm Walkthrough

Let `tasks = ["A","A","A","B","B","B"]`, $n = 2$:

```
Frequencies: A: 3, B: 3.
max_freq (M) = 3
Tasks with max_freq (C_max) = 2 (both A and B)

Formula Layout:
  Chunk 1: [A, B, idle] -> length 3 (n + 1 = 3)
  Chunk 2: [A, B, idle] -> length 3
  Chunk 3: [A, B]       -> length 2 (only the 2 max tasks)

Total time = (3 - 1) * (2 + 1) + 2 = 2 * 3 + 2 = 8.
len(tasks) = 6.
Result = max(6, 8) = 8.
```

---

### Solved Examples with Multiple Inputs

#### Example 1: Standard Idle Padding

- **Input:** `tasks = ["A","A","A","B","B","B"]`, `n = 2`
- **Output:** `8` (Execution: `A -> B -> idle -> A -> B -> idle -> A -> B`)

#### Example 2: Zero Cooldown ($n = 0$)

- **Input:** `tasks = ["A","A","A","B","B","B"]`, `n = 0`
- **Execution:** No idle time needed; tasks execute back to back.
- **Output:** `6`

#### Example 3: Abundant Distinct Tasks (No Idle Slots Needed)

- **Input:** `tasks = ["A","A","A","B","B","B","C","C","D","D","E"]`, `n = 2`
- **Execution:** $M = 3, C_{max} = 2 \implies (3 - 1) \times 3 + 2 = 8$. But `len(tasks) = 11 > 8`.
- **Output:** `11`

---

### Multi-Language Implementations

#### Python 3

##### Optimal Closed-Form Formula ($\mathcal{O}(N)$ Time, $\mathcal{O}(1)$ Space)
```python
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)
        max_freq = max(freq.values())
        max_count = sum(1 for f in freq.values() if f == max_freq)

        # Formula: (max_freq - 1) full frames of size (n + 1), plus the final frame
        min_time = (max_freq - 1) * (n + 1) + max_count
        return max(len(tasks), min_time)
```

##### Priority Queue Simulation
```python
import heapq
from collections import Counter
from typing import List

class SolutionHeap:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = Counter(tasks)
        max_heap = [-count for count in freq.values()]
        heapq.heapify(max_heap)

        time = 0

        while max_heap:
            temp = []
            cycle = n + 1

            for _ in range(cycle):
                if max_heap:
                    count = heapq.heappop(max_heap)
                    if count + 1 < 0:
                        temp.append(count + 1)
                    time += 1
                elif not temp:
                    # All tasks completely done; no trailing idle time
                    break
                else:
                    time += 1  # idle slot

            for item in temp:
                heapq.heappush(max_heap, item)

        return time
```

#### C++17

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int leastInterval(const std::vector<char>& tasks, int n) {
        std::vector<int> freq(26, 0);
        for (char t : tasks) {
            freq[t - 'A']++;
        }

        int max_freq = *std::max_element(freq.begin(), freq.end());
        int max_count = 0;
        for (int f : freq) {
            if (f == max_freq) {
                max_count++;
            }
        }

        int min_time = (max_freq - 1) * (n + 1) + max_count;
        return std::max(static_cast<int>(tasks.size()), min_time);
    }
};
```

#### Java

```java
import java.util.Arrays;

public class Solution {
    public int leastInterval(char[] tasks, int n) {
        int[] freq = new int[26];
        for (char c : tasks) {
            freq[c - 'A']++;
        }

        Arrays.sort(freq);
        int maxFreq = freq[25];
        int maxCount = 0;

        for (int i = 25; i >= 0; i--) {
            if (freq[i] == maxFreq) {
                maxCount++;
            } else {
                break;
            }
        }

        int minTime = (maxFreq - 1) * (n + 1) + maxCount;
        return Math.max(tasks.length, minTime);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(T)$
  - Counting task frequencies takes $\mathcal{O}(T)$ where $T$ is the number of tasks.
  - Finding max and summing counts across $26$ English letters takes $\mathcal{O}(26) = \mathcal{O}(1)$ time.
  - Overall Time: $\mathcal{O}(T)$.
- **Space Complexity:** $\mathcal{O}(1)$
  - The frequency table stores counts for at most 26 uppercase letters.

---

### Takeaway Pattern & Interview Traps

1. **Why $\max(len(tasks), min\_time)$?**
   - If there are many distinct tasks with lower frequencies, they can easily fill all the idle slots and even expand beyond the minimum frame without creating any cooldown violations. In that scenario, total time is simply $len(tasks)$ with zero idle time.
2. **Trailing Idle Elimination:**
   - In the heap simulation, never add idle slots after the very last cycle if no remaining tasks need to be reinserted into the heap.