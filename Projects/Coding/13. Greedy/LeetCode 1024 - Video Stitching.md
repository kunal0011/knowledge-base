---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1024: Video Stitching"
tags:
  - leetcode
  - coding
  - greedy
  - interval
  - dynamic-programming
  - amazon
  - google
---

# LeetCode 1024: Video Stitching

**Target Companies:** Amazon, Google, Meta, Apple, ByteDance  
**Difficulty:** Medium  
**Topic:** Greedy / Interval Scheduling / Jump Game II Equivalent  

---

### Problem Statement

You are given a series of video clips from a sporting event that lasted `time` seconds. These video clips can be represented with an array `clips` where `clips[i] = [start_i, end_i]` refers to the sporting event starting at `start_i` and ending at `end_i`.

We can cut these clips into smaller segments freely.
- For example, a clip `[0, 7]` can be cut into `[0, 1] + [1, 3] + [3, 7]`.

Return the **minimum number of clips** needed so that we can cut the clips into segments that cover the entire sporting event `[0, time]`. If the task is impossible, return `-1`.

---

### Input & Output Formats & Constraints

- **Input:**
  - `clips`: `List[List[int]]` / `vector<vector<int>>` / `int[][]` — interval list ($1 \le \text{clips.length} \le 100$).
  - `time`: `int` ($1 \le \text{time} \le 100$).
- **Output:**
  - `int` — minimum number of clips required to completely cover `[0, time]`, or `-1` if impossible.
- **Constraints:**
  - $1 \le \text{clips.length} \le 100$
  - $0 \le start_i < end_i \le 100$
  - $1 \le \text{time} \le 100$

---

### Key Idea & Intuition

This is an **Interval Covering Problem** that is isomorphic to **Jump Game II (LeetCode 45)**:
We start at time $t = 0$ and need to reach $t \ge \text{time}$.
At any current reach boundary `cur_end`:
- We can pick any clip that begins at or before `cur_end` ($start \le cur\_end$).
- To minimize the number of clips used, we greedily pick the clip that extends furthest into the future:
  $$next\_end = \max(next\_end, clip.end) \quad \text{for all } clip.start \le cur\_end$$
- Once all clips starting $\le cur\_end$ are considered:
  - If $next\_end \le cur\_end$, we cannot advance further $\rightarrow$ gap detected, return `-1`.
  - Otherwise, we commit to using one clip: `clips_count += 1`, and update `cur_end = next_end`.
  - If $cur\_end \ge time$, we have completely covered the target interval $\rightarrow$ return `clips_count`.

#### Jump Game II Reduction:
We can map each second $t \in [0, \text{time}]$ to the maximum right reach `max_reach[t]` starting at $t$:
$$max\_reach[t] = \max \{end \mid [start, end] \in clips \text{ and } start = t\}$$
Then, running the standard Jump Game II BFS/Greedy across $t \in [0, \text{time}-1]$ solves the problem in $\mathcal{O}(\text{time} + N)$ time!

---

### Solution Approach (Step-by-Step)

1. Create an array `max_reach` of size `time + 1` initialized to 0.
2. For each clip `[start, end]` in `clips`:
   - If `start < time`: `max_reach[start] = max(max_reach[start], end)`.
3. Initialize `clips_count = 0`, `cur_end = 0`, and `next_end = 0`.
4. Loop $t$ from $0$ to `time - 1`:
   - `next_end = max(next_end, max_reach[t])`
   - If $t == cur\_end$:
     - If $next\_end \le cur\_end$, progress is blocked $\rightarrow$ return `-1`.
     - `cur_end = next_end`
     - `clips_count += 1`
     - If `cur_end >= time`: break early.
5. Return `cur_end >= time ? clips_count : -1`.

---

### Visual Algorithm Walkthrough

For `clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]`, `time = 10`:

```
Array max_reach mapping start -> max end:
  0 -> 2
  1 -> 9  (max of 9 and 5)
  4 -> 6
  5 -> 9
  8 -> 10

Simulation:
t = 0:
  next_end = max(0, max_reach[0]=2) = 2
  t == cur_end (0):
    clips_count = 1
    cur_end = 2  (We took clip [0, 2])

t = 1:
  next_end = max(2, max_reach[1]=9) = 9

t = 2:
  next_end = max(9, max_reach[2]=0) = 9
  t == cur_end (2):
    clips_count = 2
    cur_end = 9  (We took clip [1, 9] to extend from 2 to 9)

t = 3..8:
  t = 8:
    next_end = max(9, max_reach[8]=10) = 10

t = 9:
  t == cur_end (9):
    clips_count = 3
    cur_end = 10 (We took clip [8, 10])
    cur_end >= 10 -> Covered! Stop.

Total clips used = 3.
```

---

### Solved Examples with Multiple Inputs

#### Example 1:
- **Input:** `clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]`, `time = 10`
- **Output:** `3` (Clips chosen: `[0,2]`, `[1,9]`, `[8,10]`)

#### Example 2 (Unreachable Start):
- **Input:** `clips = [[0,1],[1,2],[3,4]]`, `time = 5`
- **Tracing:** At $t = 2$, no clip covers $t \in (2, 3)$. Return `-1`.
- **Output:** `-1`

#### Example 3 (Single Clip Covers Entire Interval):
- **Input:** `clips = [[0,4],[2,8]]`, `time = 4`
- **Output:** `1` (Clip `[0,4]` suffices)

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        # max_reach[start] will hold the furthest end time achievable from start
        max_reach = [0] * (time + 1)
        for start, end in clips:
            if start <= time:
                max_reach[start] = max(max_reach[start], end)
                
        clips_count = 0
        cur_end = 0
        next_end = 0
        
        for t in range(time):
            next_end = max(next_end, max_reach[t])
            
            # Reached boundary of current clip's reach
            if t == cur_end:
                if next_end <= cur_end:
                    return -1  # Cannot make forward progress
                cur_end = next_end
                clips_count += 1
                if cur_end >= time:
                    break
                    
        return clips_count if cur_end >= time else -1
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int videoStitching(const std::vector<std::vector<int>>& clips, int time) {
        std::vector<int> max_reach(time + 1, 0);
        for (const auto& clip : clips) {
            if (clip[0] <= time) {
                max_reach[clip[0]] = std::max(max_reach[clip[0]], clip[1]);
            }
        }
        
        int clips_count = 0;
        int cur_end = 0;
        int next_end = 0;
        
        for (int t = 0; t < time; ++t) {
            next_end = std::max(next_end, max_reach[t]);
            
            if (t == cur_end) {
                if (next_end <= cur_end) {
                    return -1;
                }
                cur_end = next_end;
                clips_count++;
                if (cur_end >= time) {
                    break;
                }
            }
        }
        
        return cur_end >= time ? clips_count : -1;
    }
};
```

#### Java 17
```java
class Solution {
    public int videoStitching(int[][] clips, int time) {
        int[] maxReach = new int[time + 1];
        for (int[] clip : clips) {
            if (clip[0] <= time) {
                maxReach[clip[0]] = Math.max(maxReach[clip[0]], clip[1]);
            }
        }
        
        int clipsCount = 0;
        int curEnd = 0;
        int nextEnd = 0;
        
        for (int t = 0; t < time; t++) {
            nextEnd = Math.max(nextEnd, maxReach[t]);
            
            if (t == curEnd) {
                if (nextEnd <= curEnd) {
                    return -1;
                }
                curEnd = nextEnd;
                clipsCount++;
                if (curEnd >= time) {
                    break;
                }
            }
        }
        
        return curEnd >= time ? clipsCount : -1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(\text{time} + N)$
  - Scanning `clips` of size $N$ takes $\mathcal{O}(N)$ time.
  - Traversing time from $0$ to `time - 1` takes $\mathcal{O}(\text{time})$ time.
  - Total time is $\mathcal{O}(\text{time} + N)$. With $\text{time} \le 100$ and $N \le 100$, this executes in $< 1\text{ ms}$.
- **Space Complexity:** $\mathcal{O}(\text{time})$ auxiliary space
  - The `max_reach` array of size `time + 1` takes $\mathcal{O}(\text{time})$ space.

---

### Takeaway Pattern & Interview Traps

- **Equivalence to Jump Game II:** Whenever you need to find the minimum number of overlapping segments to cover $[0, T]$, convert the segments into a reach array `max_reach[start]` and apply the Jump Game II greedy frontier expansion.
- **Trap: Interval Gaps:** If $next\_end \le cur\_end$ at the step $t == cur\_end$, it means no clip extends past $t$, so there is an uncovered gap in time; immediately abort and return `-1`.