---
date: "2025-12-15"
type: leetcode-solution
category: "Dynamic Programming"
folder: "12. Dynamic Programming"
title: "LeetCode 55: Jump Game"
tags:
  - leetcode
  - coding
  - dynamic-programming
---

# LeetCode 55: Jump Game

**LeetCode 55 – Jump Game**, with **formal state definition, transition, DP table construction, and a worked example**.  
I will deliberately avoid the greedy shortcut so the DP reasoning is fully clear.

---

## LeetCode 55 – Jump Game

### Problem Statement

You are given an integer array `nums` where `nums[i]` represents the **maximum jump length** from index `i`.

You start at index `0`.  
Determine whether you can reach the **last index**.

---

## 1. DP State Definition (Most Important)

We define a **boolean DP array**:

```
dp[i] = True  → index i is reachable from index 0
dp[i] = False → index i is NOT reachable from index 0
```

### Size of DP Table

```
dp size = n (length of nums)
```

---

## 2. Base Case

```
dp[0] = True
```

Reason:  
You start at index `0`, so it is always reachable.

---

## 3. State Transition

To determine if index `i` is reachable:

Index `i` is reachable **if there exists some previous index `j`** such that:

1. `dp[j] == True` (we can reach `j`)
2. `j + nums[j] >= i` (from `j`, we can jump to or beyond `i`)

### Transition Formula

```
dp[i] = True
if ∃ j in [0, i-1] such that:
    dp[j] == True AND j + nums[j] >= i
```

Otherwise:

```
dp[i] = False
```

---

## 4. Order of Computation

We fill `dp` from **left to right**:

```
i = 1 → n-1
```

Because reachability of later indices depends only on earlier indices.

---

## 5. DP Algorithm (Conceptual)

For each index `i`:

* Check all previous indices `j < i`
* If any `j` can reach `i`, mark `dp[i] = True` and stop checking

---

## 6. Python 3 DP Implementation (With Typing)

```python
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [False] * n
        
        # Base case
        dp[0] = True
        
        # Fill DP table
        for i in range(1, n):
            for j in range(i):
                if dp[j] and j + nums[j] >= i:
                    dp[i] = True
                    break
        
        return dp[n - 1]
```

---

## 7. Example Walkthrough with DP Table

### Example Input

```text
nums = [2, 3, 1, 1, 4]
```

### Index and Values

```
Index:  0  1  2  3  4
nums:   2  3  1  1  4
```

---

### Step-by-Step DP Table Construction

#### Step 0

```
dp = [True, False, False, False, False]
```

---

#### i = 1

Check `j = 0`:

* `dp[0] == True`
* `0 + nums[0] = 2 >= 1` → reachable

```
dp[1] = True
```

```
dp = [True, True, False, False, False]
```

---

#### i = 2

Check `j = 0`:

* `0 + 2 >= 2` → reachable

```
dp[2] = True
```

```
dp = [True, True, True, False, False]
```

---

#### i = 3

Check `j = 0`:

* `0 + 2 < 3` → cannot reach

Check `j = 1`:

* `1 + 3 >= 3` → reachable

```
dp[3] = True
```

```
dp = [True, True, True, True, False]
```

---

#### i = 4

Check `j = 0`:

* `0 + 2 < 4` → cannot reach

Check `j = 1`:

* `1 + 3 >= 4` → reachable

```
dp[4] = True
```

```
dp = [True, True, True, True, True]
```

---

### Final Answer

```
dp[last_index] = True
```

✔ You **can reach the last index**

---

## 8. Time and Space Complexity

### Time Complexity

```
O(n²)
```

* For each index `i`, we scan all previous indices.

### Space Complexity

```
O(n)
```

* DP table of size `n`.

---

## 9. Why This DP Works (Key Insight)

This DP formulation answers a **reachability question**:

> “Is there **any** earlier reachable index that can jump to me?”

It is equivalent to checking **path existence** in a directed graph where edges represent valid jumps.

---

## 10. Important Note (Interview Insight)

* This DP solution is **correct but suboptimal**
* The **greedy solution** reduces time complexity to `O(n)`
* DP is valuable for:

  * Understanding the problem deeply
  * Explaining correctness
  * Variants where greedy fails

If you want, I can next:

* Convert this DP into **greedy step-by-step**
* Show **why DP is redundant**
* Or give a **visual reachability diagram** to link DP → Greedy reasoning