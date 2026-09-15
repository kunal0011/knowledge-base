---
date: "2025-12-18"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 1024: Video Stitching"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 1024: Video Stitching

Below is a **complete, interview-grade explanation** of **LeetCode 1024 – Video Stitching**, structured exactly as requested.

---

## 1. Problem Statement

You are given a list of video clips, where each clip is represented as  
`clips[i] = [start, end]`.

Each clip can be cut into smaller segments if needed.

You want to stitch clips together to cover the entire time interval **[0, time]**.

### Objective

Return the **minimum number of clips** needed to cover `[0, time]`.  
If it is impossible, return **-1**.

---

### Example

```
clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]
time = 10
Output = 3
```

---

## 2. Key Observation (Very Important)

This problem is **NOT DP by default**.  
It is a **classic greedy interval covering problem**, very similar to:

* Jump Game II
* Minimum number of intervals to cover a range

### Critical Insight

At any moment:

* You want to **extend coverage as far as possible**
* Using clips that **start before or at the current covered time**

Think of it as:

> “From all clips that can start now, choose the one that pushes the end farthest.”

---

## 3. Greedy Strategy (Core Idea)

### Greedy Rule

1. Sort clips by `start`
2. Maintain:

   * `curr_end`: current covered time
   * `farthest`: farthest reachable end from available clips
3. Iterate through clips:

   * If a clip starts **after** `curr_end`, you are stuck → return `-1`
   * Otherwise, update `farthest`
4. When you exhaust all clips starting ≤ `curr_end`:

   * Take **one clip** (commit)
   * Move `curr_end = farthest`
   * Increase count

---

## 4. Greedy “Trick” to Remember

This is the **interval version of Jump Game II**.

| Jump Game II | Video Stitching |
| --- | --- |
| Index range | Time range |
| nums[i] jump | clip[i] reach |
| farthest index | farthest time |
| steps | clips count |

---

## 5. Python 3 Solution (with Typing)

```python
from typing import List

class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        clips.sort()
        
        res = 0
        curr_end = 0
        farthest = 0
        i = 0
        n = len(clips)
        
        while curr_end < time:
            # Extend coverage as far as possible
            while i < n and clips[i][0] <= curr_end:
                farthest = max(farthest, clips[i][1])
                i += 1
            
            # Cannot extend further
            if farthest == curr_end:
                return -1
            
            # Take one clip
            res += 1
            curr_end = farthest
        
        return res
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]
time = 10
```

### Step 1: Sort Clips

```
[[0,2], [1,5], [1,9], [4,6], [5,9], [8,10]]
```

---

### Step 2: Initialize

```
res = 0
curr_end = 0
farthest = 0
i = 0
```

---

### Step 3: First Iteration (curr\_end = 0)

Eligible clips (`start <= 0`):

```
[0,2] → farthest = 2
```

Commit:

```
res = 1
curr_end = 2
```

---

### Step 4: Second Iteration (curr\_end = 2)

Eligible clips (`start <= 2`):

```
[1,5] → farthest = 5
[1,9] → farthest = 9
```

Commit:

```
res = 2
curr_end = 9
```

---

### Step 5: Third Iteration (curr\_end = 9)

Eligible clips (`start <= 9`):

```
[4,6]  → farthest = 9
[5,9]  → farthest = 9
[8,10] → farthest = 10
```

Commit:

```
res = 3
curr_end = 10
```

---

### Step 6: Termination

```
curr_end >= time
```

✅ **Answer = 3**

---

## 7. Why Greedy Works (Proof Intuition)

* Any solution must start with a clip covering `0`
* Among all such clips, choosing the one with **maximum reach**:

  * Leaves the **largest remaining interval**
  * Minimizes future clips
* Exchange argument guarantees optimality

---

## 8. Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | `O(n log n)` (sorting) |
| Space | `O(1)` (in-place greedy) |

---

## 9. Common Pitfalls

* ❌ Using DP unnecessarily
* ❌ Picking shortest clip instead of farthest
* ❌ Forgetting `farthest == curr_end` dead-end check

---

## 10. Final Interview Summary

> **LeetCode 1024 is a greedy interval covering problem.  
> Treat it like Jump Game II over time intervals.  
> Always extend coverage as far as possible with the fewest clips.**

If you want, I can also:

* Draw the **greedy decision tree**
* Compare this with **DP solution**
* Explain **why sorting by start is mandatory**
* Convert it into **visual timeline explanation**

Just tell me.