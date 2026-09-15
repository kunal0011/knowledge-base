---
date: "2025-12-16"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 466: Count The Repetitions"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 466: Count The Repetitions

**😇 LeetCode 466 – Count The Repetitions**, focusing on **state definition, transition, DP table construction, and a worked example**. This problem is subtle; the DP is not the usual 2D/3D grid but a **state compression + cycle detection DP**, which is why many solutions feel unintuitive.

---

## Problem Statement (Condensed)

You are given:

* `s1`, repeated `n1` times → forms `S1`
* `s2`, repeated `n2` times → forms `S2`

You must find the **maximum integer `M`** such that:

```
(S2 repeated M times) is a subsequence of (S1)
```

Return `M`.

---

## Key Observation

* You **cannot construct S1 explicitly** (length can be 10⁷).
* You only care about **how many times s2 can be matched as a subsequence** while scanning repetitions of `s1`.
* Matching `s2` is **position-based**, not substring-based.

---

## DP Insight (Core Idea)

Instead of classical DP over indices, we model the process as:

> **What happens to our position in `s2` after scanning one full `s1`?**

This leads to a **state machine DP**.

---

## State Definition

Let:

```
dp[i] = (cnt, next_pos)
```

Where:

* `i` = starting index in `s2` (0 ≤ i < len(s2))
* `cnt` = number of times `s2` is fully matched after scanning **one `s1`**
* `next_pos` = index in `s2` where we end after scanning one `s1`

This means:

> If we start matching `s2` from index `i` and scan one `s1`,
>
> * we complete `cnt` full `s2`s
> * and stop at position `next_pos` in `s2`

---

## DP Transition (How dp[i] is computed)

For each `i` in `s2`:

```
cur = i
count = 0

for char in s1:
    if char == s2[cur]:
        cur += 1
        if cur == len(s2):
            count += 1
            cur = 0
```

Finally:

```
dp[i] = (count, cur)
```

This is **pure simulation**, done **len(s2)** times.

---

## DP Table Creation

* DP table size = `len(s2)`
* Each entry computed in `O(len(s1))`
* Total preprocessing: `O(len(s1) * len(s2))`

This is feasible because `len(s1), len(s2) ≤ 100`.

---

## Example Walkthrough

### Input

```
s1 = "acb", n1 = 4
s2 = "ab",  n2 = 2
```

---

### Step 1: Build DP Table

`s2 = "ab"` → indices `{0:'a', 1:'b'}`

---

#### dp[0] → start matching from `'a'`

Scan `"acb"`:

| char | s2[cur] | action |
| --- | --- | --- |
| a | a | cur=1 |
| c | b | skip |
| b | b | cur=2 → complete s2 |

Result:

```
dp[0] = (1, 0)
```

---

#### dp[1] → start from `'b'`

Scan `"acb"`:

| char | s2[cur] | action |
| --- | --- | --- |
| a | b | skip |
| c | b | skip |
| b | b | cur=2 → complete s2 |

Result:

```
dp[1] = (1, 0)
```

---

### Final DP Table

| start\_pos | completed s2 | next\_pos |
| --- | --- | --- |
| 0 | 1 | 0 |
| 1 | 1 | 0 |

---

## Using DP Over n1 Repetitions (Main Loop)

Now simulate scanning `s1` **n1 times**, using DP jumps.

Maintain:

```
cur_pos = 0
total_s2 = 0
```

For each repetition of `s1`:

```
(cnt, next_pos) = dp[cur_pos]
total_s2 += cnt
cur_pos = next_pos
```

---

### Iteration Trace

| s1 repeat | cur\_pos | gained | total\_s2 |
| --- | --- | --- | --- |
| 1 | 0 | +1 | 1 |
| 2 | 0 | +1 | 2 |
| 3 | 0 | +1 | 3 |
| 4 | 0 | +1 | 4 |

---

## Final Answer

We matched `total_s2 = 4` copies of `s2`.

But required unit is `s2` repeated `n2` times:

```
answer = total_s2 // n2 = 4 // 2 = 2
```

---

## Python 3 Implementation (Typed)

```python
from typing import List, Tuple

class Solution:
    def getMaxRepetitions(self, s1: str, n1: int, s2: str, n2: int) -> int:
        l1, l2 = len(s1), len(s2)

        # dp[i] = (count of s2, next position in s2)
        dp: List[Tuple[int, int]] = [(0, 0)] * l2

        # Build DP table
        for i in range(l2):
            cur = i
            cnt = 0
            for ch in s1:
                if ch == s2[cur]:
                    cur += 1
                    if cur == l2:
                        cnt += 1
                        cur = 0
            dp[i] = (cnt, cur)

        # Simulate n1 repetitions of s1
        cur_pos = 0
        total_s2 = 0

        for _ in range(n1):
            cnt, next_pos = dp[cur_pos]
            total_s2 += cnt
            cur_pos = next_pos

        return total_s2 // n2
```

---

## Why This Is DP (Conceptually)

* **State**: current index in `s2`
* **Transition**: consuming one full `s1`
* **DP Table**: precomputed transition results
* **Optimization**: converts repeated scanning into constant-time jumps

This is **state-compression DP**, not grid DP.

---

If you want next:

* Cycle-detection optimization (for very large `n1`)
* Graph interpretation of DP states
* Why greedy or sliding window fails here

Tell me how deep you want to go.