---
date: "2025-12-17"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 659: Split Array into Consecutive Subsequences"
tags:
  - leetcode
  - coding
  - greedy
---

# LeetCode 659: Split Array into Consecutive Subsequences

**LeetCode 659 – Split Array into Consecutive Subsequences**, aligned with interview-level expectations.

---

## 1. Problem Statement

**LeetCode 659 – Split Array into Consecutive Subsequences**

You are given a **sorted** integer array `nums`.

Determine whether it is possible to split `nums` into **one or more subsequences** such that:

1. Each subsequence consists of **consecutive integers**.
2. Each subsequence has a **length of at least 3**.
3. Every element in `nums` is used **exactly once**.

Return `true` if such a split is possible, otherwise return `false`.

---

### Example

```text
Input: nums = [1,2,3,3,4,5]
Output: true
Explanation:
Possible split:
[1,2,3] and [3,4,5]
```

---

## 2. Key Observations

### Observation 1: Greedy, Not Backtracking

* We do **not** need to enumerate all possible subsequences.
* A greedy strategy works because:

  * Shorter subsequences are more fragile.
  * If we do not extend them early, they may fail the length ≥ 3 constraint later.

---

### Observation 2: Two Critical Questions for Each Number `x`

When processing a number `x`, we must decide:

1. **Can I extend an existing subsequence ending at `x-1`?**
2. **If not, can I start a new subsequence `[x, x+1, x+2]`?**

If neither is possible → **return false immediately**.

---

### Observation 3: Priority of Actions (Greedy Rule)

**Always try to extend an existing subsequence first.**

Why?

* Existing subsequences may already be length 1 or 2.
* If we start a new subsequence instead, the old one might remain too short forever.

This is the **core greedy trick** of this problem.

---

## 3. Greedy Strategy (Core Idea)

We maintain **two hash maps**:

### 1. `freq`

* `freq[x]` = how many times `x` is still unused.

### 2. `need`

* `need[x]` = how many subsequences are currently **waiting for `x`** as the next element.

---

### Processing Logic for Each Number `x`

For each `x` in `nums`:

1. **If `freq[x] == 0`**

   * Already used → skip.
2. **If `need[x] > 0`**

   * Extend an existing subsequence:

     * Consume `x`
     * That subsequence will now need `x+1`
3. **Else**

   * Try to start a new subsequence `[x, x+1, x+2]`
   * Requires:

     * `freq[x+1] > 0`
     * `freq[x+2] > 0`
   * If not possible → return `false`

---

## 4. Python 3 Solution (with Typing)

```python
from typing import List
from collections import Counter, defaultdict

class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        freq = Counter(nums)          # count of remaining numbers
        need = defaultdict(int)       # subsequences waiting for a number

        for x in nums:
            if freq[x] == 0:
                continue

            # Case 1: extend an existing subsequence
            if need[x] > 0:
                need[x] -= 1
                need[x + 1] += 1
                freq[x] -= 1

            # Case 2: start a new subsequence of length 3
            else:
                if freq[x + 1] > 0 and freq[x + 2] > 0:
                    freq[x] -= 1
                    freq[x + 1] -= 1
                    freq[x + 2] -= 1
                    need[x + 3] += 1
                else:
                    return False

        return True
```

---

## 5. Complete Worked Example (Step-by-Step)

### Input

```text
nums = [1,2,3,3,4,5]
```

---

### Initial State

```
freq = {1:1, 2:1, 3:2, 4:1, 5:1}
need = {}
```

---

### Step 1: x = 1

* `need[1] == 0`
* Can we start new subsequence?

  * freq[2] > 0, freq[3] > 0 → yes
* Start `[1,2,3]`

```
freq = {1:0, 2:0, 3:1, 4:1, 5:1}
need = {4:1}   # subsequence expects 4 next
```

---

### Step 2: x = 2

* `freq[2] == 0` → skip

---

### Step 3: x = 3 (first occurrence)

* `need[3] == 0`
* Try to start new subsequence:

  * freq[4] > 0, freq[5] > 0 → yes
* Start `[3,4,5]`

```
freq = {1:0, 2:0, 3:0, 4:0, 5:0}
need = {4:1, 6:1}
```

---

### Step 4: x = 3 (second occurrence)

* `freq[3] == 0` → skip

---

### Step 5: x = 4

* `freq[4] == 0` → skip

---

### Step 6: x = 5

* `freq[5] == 0` → skip

---

### Final Result

All numbers used successfully without violating constraints.

```
Return True
```

---

## 6. Why This Greedy Works (Intuition Summary)

* Extending existing subsequences prevents short, invalid leftovers.
* Starting a new subsequence only when extension is impossible ensures:

  * Minimum length of 3 is always satisfied.
* The sorted nature of `nums` ensures correctness of local greedy decisions.

---

## 7. Time and Space Complexity

| Metric | Value |
| --- | --- |
| Time | **O(n)** |
| Space | **O(n)** |

---

If you want, I can also provide:

* Counterexample where naive greedy fails
* Visual subsequence evolution diagram
* Comparison with heap-based solution
* Follow-up variants and interview traps

Just specify.