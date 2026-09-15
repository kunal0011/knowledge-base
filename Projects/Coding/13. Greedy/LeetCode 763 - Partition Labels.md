---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 763: Partition Labels"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 763: Partition Labels

Below is a **complete, structured, interview-grade explanation** of **LeetCode 763 – Partition Labels**, aligned with your usual learning preference (problem → observation → greedy trick → code → step-by-step dry run).

---

## 1. Problem Statement

**LeetCode 763 – Partition Labels**

You are given a string `s`.  
Your task is to partition the string into as many parts as possible such that:

* Each letter appears in **at most one partition**
* After partitioning, concatenate all parts in order to obtain the original string

Return a list of integers representing the **size of each partition**.

### Example

```text
Input:  s = "ababcbacadefegdehijhklij"
Output: [9, 7, 8]
```

---

## 2. Key Observation (Most Important Insight)

> **If a character appears multiple times, all its occurrences must lie in the same partition.**

Therefore:

* A partition must extend **at least** until the **last occurrence** of every character inside it.
* Once we include a character, we are *forced* to include all characters until its last position.

This directly hints toward a **greedy boundary expansion** strategy.

---

## 3. Greedy Strategy (Why This Works)

### Core Greedy Idea

1. **Precompute the last occurrence index of every character**
2. Traverse the string from left to right
3. Maintain:

   * `start` → start index of current partition
   * `end` → farthest last occurrence of characters seen so far
4. When current index `i == end`:

   * We have safely completed a partition
   * No character inside this segment appears later
   * Cut the partition

### Why Greedy is Optimal

* We always close a partition **as early as possible**
* Closing earlier gives **maximum number of partitions**
* Delaying the cut never increases the count

This is a classic **interval expansion greedy pattern**.

---

## 4. Algorithm Steps

1. Build a map: `last[c] = last index of character c`
2. Initialize:

   * `start = 0`
   * `end = 0`
   * `result = []`
3. Iterate over string with index `i`:

   * Update `end = max(end, last[s[i]])`
   * If `i == end`:

     * Partition size = `end - start + 1`
     * Append to result
     * Move `start = i + 1`

---

## 5. Python 3 Solution (With Typing)

```python
from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # Step 1: Record last occurrence of each character
        last = {}
        for i, ch in enumerate(s):
            last[ch] = i

        result: List[int] = []
        start = 0
        end = 0

        # Step 2: Greedy scan
        for i, ch in enumerate(s):
            end = max(end, last[ch])

            # If current index reaches the end of partition
            if i == end:
                result.append(end - start + 1)
                start = i + 1

        return result
```

---

## 6. Complete Worked Example (Step-by-Step)

### Input

```
s = "ababcbacadefegdehijhklij"
```

### Step 1: Last Occurrence Map

```
a → 8
b → 5
c → 7
d → 14
e → 15
f → 11
g → 13
h → 19
i → 22
j → 23
k → 20
l → 21
```

---

### Step 2: Traverse and Expand Partition

| i | s[i] | last[s[i]] | end | action |
| --- | --- | --- | --- | --- |
| 0 | a | 8 | 8 | expand |
| 1 | b | 5 | 8 | expand |
| 2 | a | 8 | 8 | expand |
| 3 | b | 5 | 8 | expand |
| 4 | c | 7 | 8 | expand |
| 5 | b | 5 | 8 | expand |
| 6 | a | 8 | 8 | expand |
| 7 | c | 7 | 8 | expand |
| 8 | a | 8 | 8 | **cut partition** |

**Partition 1:** indices `[0..8]`, size = `9`

---

| i | s[i] | last[s[i]] | end | action |
| --- | --- | --- | --- | --- |
| 9 | d | 14 | 14 | expand |
| 10 | e | 15 | 15 | expand |
| 11 | f | 11 | 15 | expand |
| 12 | e | 15 | 15 | expand |
| 13 | g | 13 | 15 | expand |
| 14 | d | 14 | 15 | expand |
| 15 | e | 15 | 15 | **cut partition** |

**Partition 2:** indices `[9..15]`, size = `7`

---

| i | s[i] | last[s[i]] | end | action |
| --- | --- | --- | --- | --- |
| 16 | h | 19 | 19 | expand |
| 17 | i | 22 | 22 | expand |
| 18 | j | 23 | 23 | expand |
| 19 | h | 19 | 23 | expand |
| 20 | k | 20 | 23 | expand |
| 21 | l | 21 | 23 | expand |
| 22 | i | 22 | 23 | expand |
| 23 | j | 23 | 23 | **cut partition** |

**Partition 3:** indices `[16..23]`, size = `8`

---

### Final Output

```
[9, 7, 8]
```

---

## 7. Pattern Recognition (Interview Tip)

This problem belongs to the **Greedy – Interval Expansion** pattern.

You will see the same logic in:

* Merge Intervals
* Task Scheduling
* Range covering problems

**Key signal**:

> “A constraint that forces elements to stay together → track last occurrence → greedy cut when safe”

---

If you want, I can also:

* Explain this using **interval merging analogy**
* Show a **wrong greedy approach and why it fails**
* Convert this into a **general reusable greedy template**
* Compare with **two-pointer vs greedy framing**

Just tell me.