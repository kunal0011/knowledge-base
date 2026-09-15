---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 1052: Grumpy Bookstore Owner"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - amazon
  - google
---

# LeetCode 1052: Grumpy Bookstore Owner

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Optimization Decoupling  

---

### Problem Statement

There is a bookstore owner that has a store open for `n` minutes. Every minute, some number of customers enter the store. You are given an integer array `customers` of length `n` where `customers[i]` is the number of the customer that enters the store at the start of the $i$-th minute and all those customers leave after the end of that minute.

On some minutes, the bookstore owner is grumpy. You are given a binary array `grumpy` where `grumpy[i] == 1` if the bookstore owner is grumpy during the $i$-th minute, and `grumpy[i] == 0` otherwise.

When the bookstore owner is grumpy, the customers of that minute are not satisfied, otherwise they are satisfied.

The bookstore owner knows a secret technique to keep themselves not grumpy for `minutes` consecutive minutes, but can only use it once.

Return *the maximum number of customers that can be satisfied throughout the day*.

---

### Input & Output Formats & Constraints

- **Input:** `customers: List[int]`, `grumpy: List[int]`, `minutes: int`
- **Output:** `int` (maximum possible satisfied customers)
- **Constraints:**
  - $n == \text{customers.length} == \text{grumpy.length}$
  - $1 \le \text{minutes} \le n \le 2 \times 10^4$
  - $0 \le \text{customers}[i] \le 1000$
  - `grumpy[i]` is either `0` or `1`.

---

### Key Idea & Intuition

- **Decoupling Satisfaction into Two Independent Components:**
  - **Component 1: Guaranteed Base Satisfaction:**
    - Customers who arrive when the owner is naturally calm (`grumpy[i] == 0`) are satisfied unconditionally:
      $$\text{base\_satisfied} = \sum_{i=0}^{n-1} \text{customers}[i] \times (1 - \text{grumpy}[i])$$
  - **Component 2: Technique Gain (Fixed Window):**
    - Activating the technique during a window $[i, i + \text{minutes} - 1]$ converts grumpy minutes into satisfied minutes.
    - The *additional* customers gained by choosing this window is:
      $$\text{gain}(i) = \sum_{j=i}^{i + \text{minutes} - 1} \text{customers}[j] \times \text{grumpy}[j]$$
- **Fixed-Size Sliding Window:**
  - We simply need to find the window of length `minutes` that maximizes `gain(i)`.
  - Calculate gain for the first `minutes` elements ($i = 0 \dots \text{minutes} - 1$).
  - Slide the window right by 1 index:
    - Add incoming element: `gain += customers[i] * grumpy[i]`.
    - Subtract outgoing element: `gain -= customers[i - minutes] * grumpy[i - minutes]`.
    - Maintain `max_gain = max(max_gain, gain)`.
  - Final Answer: $\text{base\_satisfied} + \text{max\_gain}$.

---

### Solution Approach (Step-by-Step)

1. Compute `base_satisfied` as the sum of `customers[i]` where `grumpy[i] == 0`.
2. Compute `curr_gain` for the initial window of size `minutes` ($0 \le i < \text{minutes}$):
   - If `grumpy[i] == 1`: `curr_gain += customers[i]`.
3. Set `max_gain = curr_gain`.
4. Slide window from $i = \text{minutes}$ to $n - 1$:
   - Add incoming: if `grumpy[i] == 1`, `curr_gain += customers[i]`.
   - Remove outgoing: if `grumpy[i - minutes] == 1`, `curr_gain -= customers[i - minutes]`.
   - Update `max_gain = max(max_gain, curr_gain)`.
5. Return `base_satisfied + max_gain`.

---

### Visual Algorithm Walkthrough

Let `customers = [1, 0, 1, 2, 1, 1, 7, 5]`, `grumpy = [0, 1, 0, 1, 0, 1, 0, 1]`, `minutes = 3`.

```
Indices:          0  1  2  3  4  5  6  7
Customers:       [1, 0, 1, 2, 1, 1, 7, 5]
Grumpy:          [0, 1, 0, 1, 0, 1, 0, 1]

Base satisfied (grumpy == 0):
customers[0] = 1
customers[2] = 1
customers[4] = 1
customers[6] = 7
Base total = 1 + 1 + 1 + 7 = 10.

Gain array (customers[i] * grumpy[i]):
[0, 0, 0, 2, 0, 1, 0, 5]

Sliding Window of size 3 over Gain Array:
- Window [0..2]: [0, 0, 0]             -> gain = 0
- Window [1..3]: [0, 0, 2]             -> gain = 2
- Window [2..4]: [0, 2, 0]             -> gain = 2
- Window [3..5]: [2, 0, 1]             -> gain = 3
- Window [4..6]: [0, 1, 0]             -> gain = 1
- Window [5..7]: [1, 0, 5]             -> gain = 6 (MAX GAIN!)

Total Maximum Satisfaction = Base (10) + Max Gain (6) = 16!
```

---

### Solved Examples with Multiple Inputs

| Test Case | `customers` | `grumpy` | `minutes` | Base Satisfied | Max Gain | Total Output |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard** | `[1,0,1,2,1,1,7,5]` | `[0,1,0,1,0,1,0,1]` | `3` | 10 | 6 (at end: `1+0+5`) | `16` |
| **Single Minute** | `[1]` | `[0]` | `1` | 1 | 0 | `1` |
| **Always Grumpy** | `[4, 10, 10]` | `[1, 1, 1]` | `2` | 0 | 20 (`10+10`) | `20` |
| **Never Grumpy** | `[5, 5, 5]` | `[0, 0, 0]` | `2` | 15 | 0 | `15` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        """
        Maximizes satisfied customers by applying a secret calm technique of length `minutes`.
        Uses a fixed-size sliding window over potential customer gain.
        """
        n = len(customers)

        # Step 1: Calculate naturally satisfied customers
        base_satisfied = sum(customers[i] for i in range(n) if grumpy[i] == 0)

        # Step 2: Calculate gain for the initial window of size `minutes`
        curr_gain = sum(customers[i] for i in range(minutes) if grumpy[i] == 1)
        max_gain = curr_gain

        # Step 3: Slide window across the array
        for i in range(minutes, n):
            # Add incoming element
            if grumpy[i] == 1:
                curr_gain += customers[i]
            # Remove outgoing element
            if grumpy[i - minutes] == 1:
                curr_gain -= customers[i - minutes]

            if curr_gain > max_gain:
                max_gain = curr_gain

        return base_satisfied + max_gain
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxSatisfied(const std::vector<int>& customers, const std::vector<int>& grumpy, int minutes) {
        int n = static_cast<int>(customers.size());
        int base_satisfied = 0;

        for (int i = 0; i < n; ++i) {
            if (grumpy[i] == 0) {
                base_satisfied += customers[i];
            }
        }

        int curr_gain = 0;
        for (int i = 0; i < minutes; ++i) {
            if (grumpy[i] == 1) {
                curr_gain += customers[i];
            }
        }

        int max_gain = curr_gain;
        for (int i = minutes; i < n; ++i) {
            if (grumpy[i] == 1) {
                curr_gain += customers[i];
            }
            if (grumpy[i - minutes] == 1) {
                curr_gain -= customers[i - minutes];
            }
            max_gain = std::max(max_gain, curr_gain);
        }

        return base_satisfied + max_gain;
    }
};
```

#### Java
```java
class Solution {
    public int maxSatisfied(int[] customers, int[] grumpy, int minutes) {
        int n = customers.length;
        int baseSatisfied = 0;

        for (int i = 0; i < n; i++) {
            if (grumpy[i] == 0) {
                baseSatisfied += customers[i];
            }
        }

        int currGain = 0;
        for (int i = 0; i < minutes; i++) {
            if (grumpy[i] == 1) {
                currGain += customers[i];
            }
        }

        int maxGain = currGain;
        for (int i = minutes; i < n; i++) {
            if (grumpy[i] == 1) {
                currGain += customers[i];
            }
            if (grumpy[i - minutes] == 1) {
                currGain -= customers[i - minutes];
            }
            maxGain = Math.max(maxGain, currGain);
        }

        return baseSatisfied + maxGain;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{customers.length}$.
  - One pass to compute base satisfaction and the initial window of size `minutes`.
  - A second pass to slide the window from index `minutes` to $N - 1$, updating `curr_gain` in $\mathcal{O}(1)$ per step.
  - Total operations $\le 2N$, executing in $< 5 \text{ ms}$ for $N = 2 \times 10^4$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space since we only maintain scalar variables (`base_satisfied`, `curr_gain`, `max_gain`).

---

### Takeaway Pattern & Interview Traps

- **Fixed-Size Window Maintenance:** When window length is fixed at $k$, remember to remove index $i - k$ whenever adding index $i$.
- **Decoupling Gains:** Do not recalculate the entire window's total customers! Only track the *marginal gain* of converting grumpy minutes ($g = 1$) into satisfied minutes, added to the static base satisfaction.