---
date: "2025-12-19"
type: leetcode-solution
category: "Queue"
folder: "05. Queue"
title: "LeetCode 862: Shortest Subarray with Sum at Least K"
tags:
  - leetcode
  - coding
  - queue
---

# LeetCode 862: Shortest Subarray with Sum at Least K

Below is a **complete, structured explanation** of **LeetCode 862 – Shortest Subarray with Sum at Least K**, aligned with how you have been studying advanced DSA patterns.

---

## LeetCode 862 — Shortest Subarray with Sum at Least K

---

## Problem Statement

You are given an integer array `nums` (may contain **negative numbers**) and an integer `k`.

Return the **length of the shortest, non-empty subarray** whose sum is **at least `k`**.  
If no such subarray exists, return `-1`.

### Example

```text
nums = [2, -1, 2]
k = 3
Output: 3
```

---

## Key Observations

1. **Classic sliding window does NOT work**

   * Presence of **negative numbers** breaks monotonicity of window sum.
   * Expanding the window does not guarantee increasing sum.
2. **Prefix Sum Transformation**

   * Let

     ```
     prefix[i] = nums[0] + nums[1] + ... + nums[i-1]
     ```
   * Subarray sum from `j` to `i-1` is:

     ```
     prefix[i] - prefix[j]
     ```
   * We want:

     ```
     prefix[i] - prefix[j] >= k
     ```
3. **Goal Restated**

   * For each `i`, find the **smallest index `j < i`** such that:

     ```
     prefix[j] <= prefix[i] - k
     ```
   * Minimize `(i - j)`

---

## Queue (Deque) Key Insights — *Monotonic Queue*

This is the **core trick** of the problem.

### Why a Deque?

We need:

* Fast access to **smallest prefix values**
* Fast removal of **useless candidates**

### Invariants Maintained

We maintain a deque of **indices of prefix sums**, such that:

1. **Prefix sums are increasing**

   ```
   prefix[dq[0]] < prefix[dq[1]] < ...
   ```
2. **Deque front gives best candidate for shortest subarray**

---

### Two Critical Operations

#### 1️⃣ Pop from Front (Valid Subarray Found)

While:

```
prefix[i] - prefix[dq[0]] >= k
```

* We found a valid subarray
* Update answer: `i - dq[0]`
* Remove front → try to find an even **shorter** subarray

#### 2️⃣ Pop from Back (Maintain Monotonicity)

While:

```
prefix[i] <= prefix[dq[-1]]
```

* The previous prefix is **worse**
* Larger prefix with earlier index is never optimal
* Remove it

---

## Algorithm Steps

1. Build prefix sum array
2. Initialize empty deque
3. Iterate over prefix indices
4. Apply **front popping** for valid subarrays
5. Apply **back popping** to maintain increasing order
6. Append current index
7. Return result or `-1`

---

## Python 3 Solution (With Typing)

```python
from typing import List
from collections import deque

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)

        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        dq = deque()
        ans = float('inf')

        for i in range(n + 1):
            # Step 1: Check if we can form a valid subarray
            while dq and prefix[i] - prefix[dq[0]] >= k:
                ans = min(ans, i - dq.popleft())

            # Step 2: Maintain increasing prefix sums
            while dq and prefix[i] <= prefix[dq[-1]]:
                dq.pop()

            dq.append(i)

        return ans if ans != float('inf') else -1
```

---

## Worked Example (Step-by-Step)

### Input

```text
nums = [2, -1, 2]
k = 3
```

### Prefix Sum

```
index:   0   1   2   3
prefix: [0,  2,  1,  3]
```

---

### Iteration Details

#### i = 0 → prefix = 0

```
deque = [0]
```

---

#### i = 1 → prefix = 2

```
2 - 0 < 3 → no pop front
deque = [0, 1]
```

---

#### i = 2 → prefix = 1

```
prefix[2] <= prefix[1] → pop 1
prefix[2] > prefix[0] → stop
deque = [0, 2]
```

---

#### i = 3 → prefix = 3

```
3 - 0 >= 3 → valid subarray length = 3
ans = 3
pop front → dq = [2]

3 - 1 < 3 → stop
deque = [2, 3]
```

---

### Final Answer

```
Shortest subarray length = 3
```

---

## Time & Space Complexity

| Metric | Complexity |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** |

Each index enters and leaves the deque **once**.

---

## Pattern Recognition

This problem is a **canonical example** of:

* Prefix Sum + Monotonic Deque
* Hardest variation of **Subarray Sum ≥ K**
* Appears in advanced interviews (Google, Meta)

If you want, I can also:

* Compare this with **LeetCode 209**
* Provide a **deque visualization**
* List **all problems using this exact pattern**

Just tell me.