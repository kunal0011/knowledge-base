---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 621: Task Scheduler"
tags:
  - leetcode
  - coding
  - greedy
  - math
  - priority-queue
  - hash-table
  - amazon
  - google
---

# LeetCode 621: Task Scheduler

**Target Companies:** Amazon (Signature Classic), Google, Meta, Microsoft, Apple, Bloomberg  
**Difficulty:** Medium  
**Topic:** Greedy / Math / Priority Queue / Hash Table  

---

### Problem Statement

You are given an array of CPU `tasks`, each represented by letters A to Z, and a cooling time, `n`. Each cycle or interval allows the completion of one task. Tasks can be completed in any order, but there's a constraint: identical tasks must be separated by at least `n` intervals with other tasks or idle cycles.

Return the **minimum number of CPU intervals** required to complete all tasks.

---

### Input & Output Formats & Constraints

- **Input:**
  - `tasks`: `List[str]` / `vector<char>` / `char[]` ($1 \le \text{tasks.length} \le 10^4$).
  - `n`: `int` ($0 \le n \le 100$).
- **Output:**
  - `int` — the minimum total time units (busy + idle) required.
- **Constraints:**
  - $1 \le \text{tasks.length} \le 10^4$
  - `tasks[i]` is an uppercase English letter.
  - $0 \le n \le 100$

---

### Key Idea & Intuition

The execution time is constrained entirely by the task(s) with the **highest frequency**.

#### The Block Partitioning Invariant:
Let:
- $M = \max(\text{frequencies})$ be the maximum frequency of any task.
- $k$ be the number of distinct tasks that share this maximum frequency $M$.

Consider scheduling the most frequent tasks first:
- They must be separated by at least $n$ other slots.
- This creates $M - 1$ full "chunks" or "frames", each of length $n + 1$ (1 slot for the most frequent task + $n$ cooldown/filler slots).
- Following these $M - 1$ frames, the final $M$-th occurrence of the top tasks will sit in the final row, which has length $k$.

Thus, the minimum frame size dictated by cooldowns is:
$$\text{Frame Time} = (M - 1) \times (n + 1) + k$$

#### What if there are more tasks than empty slots?
If there are enough other tasks, we can simply distribute them into the chunks, expanding the chunk sizes beyond $n + 1$ without needing any idle slots at all!
When no idle slots are needed, the total time is simply the total number of tasks: $\text{len}(\text{tasks})$.

Therefore, the global minimum time is given by the closed-form formula:
$$\text{Min Time} = \max(\text{len}(\text{tasks}), (M - 1) \times (n + 1) + k)$$

This computes the optimal answer in strictly $\mathcal{O}(N)$ time and $\mathcal{O}(1)$ space, without any queue simulation!

---

### Solution Approach (Step-by-Step)

1. Compute frequency of each uppercase letter from A to Z using an array of size 26.
2. Find the maximum frequency: `max_freq = max(counts)`.
3. Count how many tasks have this maximum frequency:
   `count_max = sum(1 for c in counts if c == max_freq)`.
4. Calculate theoretical minimum time:
   `frame_time = (max_freq - 1) * (n + 1) + count_max`.
5. Return `max(len(tasks), frame_time)`.

---

### Visual Algorithm Walkthrough

For `tasks = ["A","A","A","B","B","B"]`, `n = 2`:
- $M = 3$ (both A and B appear 3 times)
- $k = 2$ (two tasks: A and B)

Frame construction:
```
Chunk 1: [ A , B , idle ]  (size n + 1 = 3)
Chunk 2: [ A , B , idle ]  (size n + 1 = 3)
Final:   [ A , B ]         (size k = 2)

Timeline:
Cycle:   1   2   3     4   5   6     7   8
Task:    A   B  idle   A   B  idle   A   B

Calculated Time:
(3 - 1) * (2 + 1) + 2 = 2 * 3 + 2 = 8.
max(len(tasks)=6, 8) = 8.
```

For `tasks = ["A","A","A","B","C","D","E"]`, `n = 2`:
- $M = 3$ (task A appears 3 times), $k = 1$ (only A)
- `frame_time` = $(3 - 1) \times (2 + 1) + 1 = 7$.
- Total tasks = 7.
- $\max(7, 7) = 7$. (Schedule: `A B C A D E A`, 0 idle slots).

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `tasks = ["A","A","A","B","B","B"]`, `n = 2`
- **Output:** `8`

#### Example 2:
- **Input:** `tasks = ["A","C","A","B","D","B"]`, `n = 1`
- **Tracing:** $M = 2$ (A:2, B:2), $k = 2$.
  $\text{frame} = (2 - 1) \times 2 + 2 = 4$. Total tasks = 6. $\max(6, 4) = 6$.
- **Output:** `6`

#### Example 3:
- **Input:** `tasks = ["A","A","A","B","B","B"]`, `n = 0`
- **Tracing:** No cooldown ($n = 0$). Can run back-to-back.
- **Output:** `6`

---

### Multi-Language Implementations

#### Python 3
```python
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = Counter(tasks)
        max_freq = max(counts.values())
        
        # Number of tasks that have this maximum frequency
        count_max = sum(1 for count in counts.values() if count == max_freq)
        
        # Theoretical frame size dictated by cooldown
        frame_time = (max_freq - 1) * (n + 1) + count_max
        
        return max(len(tasks), frame_time)
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int leastInterval(const std::vector<char>& tasks, int n) {
        std::vector<int> counts(26, 0);
        for (char c : tasks) {
            counts[c - 'A']++;
        }
        
        int max_freq = *std::max_element(counts.begin(), counts.end());
        int count_max = 0;
        for (int c : counts) {
            if (c == max_freq) {
                count_max++;
            }
        }
        
        int frame_time = (max_freq - 1) * (n + 1) + count_max;
        int total_tasks = static_cast<int>(tasks.size());
        
        return std::max(total_tasks, frame_time);
    }
};
```

#### Java 17
```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        int[] counts = new int[26];
        int maxFreq = 0;
        
        for (char c : tasks) {
            counts[c - 'A']++;
            maxFreq = Math.max(maxFreq, counts[c - 'A']);
        }
        
        int countMax = 0;
        for (int c : counts) {
            if (c == maxFreq) {
                countMax++;
            }
        }
        
        int frameTime = (maxFreq - 1) * (n + 1) + countMax;
        
        return Math.max(tasks.length, frameTime);
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{tasks.length}$
  - Counting task frequencies takes $\mathcal{O}(N)$ time.
  - Finding the maximum and counting max ties across 26 alphabet letters takes $\mathcal{O}(26) = \mathcal{O}(1)$ time.
  - Total time: $\mathcal{O}(N)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space
  - The frequency table size is fixed at 26 elements.

---

### Takeaway Pattern & Interview Traps

- **Closed-Form Math vs Simulation:** While many candidates simulate the schedule with a priority queue ($\mathcal{O}(N \log 26)$), the closed-form math formula `max(len(tasks), (max_freq - 1) * (n + 1) + count_max)` is clean, optimal, and demonstrates deep algorithmic maturity.
- **Multiple Maximum Frequency Tasks ($k > 1$):** If both A and B have frequency 3, the tail has 2 elements (`[A, B]`). Forgetting `count_max` and assuming only 1 task has max frequency is a frequent bug!