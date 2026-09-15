---
date: "2026-09-15"
type: leetcode-solution
category: "Arrays & Hashing"
folder: "01. Arrays & Hashing"
title: "LeetCode 41: First Missing Positive"
tags:
  - leetcode
  - coding
  - arrays-and-hashing
  - amazon
  - google
  - meta
---

# LeetCode 41: First Missing Positive

**Target Companies:** Google (Signature Hard), Amazon, Meta, Microsoft, Apple  
**Difficulty:** Hard  
**Topic:** In-Place Cyclic Sort / Array as Hash Map

---

### Problem Statement

Given an unsorted integer array `nums`, return the **smallest missing positive integer**.

You must implement an algorithm that runs in **$\mathcal{O}(n)$ time** and uses **$\mathcal{O}(1)$ auxiliary space**.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]`
- **Output:** `int` (the smallest positive integer not present in `nums`)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - $-2^{31} \le \text{nums}[i] \le 2^{31} - 1$

---

### Key Idea & Intuition

#### 1. Pigeonhole Principle
For an array of length $N$, the smallest missing positive integer **must** lie in the range $[1, N + 1]$:
- If all integers $1, 2, \dots, N$ are present in `nums`, then the missing positive is $N + 1$.
- Otherwise, at least one integer in $[1, N]$ is missing, and the answer is the smallest such integer.

Any values that are $\le 0$ or $> N$ are completely irrelevant to our search and can be ignored.

#### 2. Cyclic Placement (Array as Hash Map)
We can achieve $\mathcal{O}(1)$ auxiliary space by using the array itself as a hash map where each number $x \in [1, N]$ is placed at its target index $x - 1$:
- For each index $i$, while $1 \le \text{nums}[i] \le N$ and $\text{nums}[i] \ne \text{nums}[\text{nums}[i] - 1]$:
  - Swap $\text{nums}[i]$ with $\text{nums}[\text{nums}[i] - 1]$.
- After the cyclic placement loop, a second linear scan finds the first index $i$ where $\text{nums}[i] \ne i + 1$. The answer is $i + 1$.
- If all positions $0 \le i < N$ satisfy $\text{nums}[i] == i + 1$, return $N + 1$.

---

### Solution Approach (Step-by-Step)

1. **Cyclic Sort Pass:**
   - Iterate $i$ from $0$ to $N - 1$:
     - Use a `while` loop: as long as $\text{nums}[i]$ is within valid positive range $[1, N]$ and not already at its correct position (i.e. $\text{nums}[i] \ne \text{nums}[\text{nums}[i] - 1]$), swap $\text{nums}[i]$ with $\text{nums}[\text{nums}[i] - 1]$.
     - The `while` condition terminates if $\text{nums}[i]$ is $\le 0$, $> N$, or equal to the element already sitting at its target index (handling duplicates).
2. **Scan for First Missing Value:**
   - Iterate $i$ from $0$ to $N - 1$:
     - If $\text{nums}[i] \ne i + 1$, return $i + 1$.
3. **All Present Fallback:**
   - If every index holds its expected value $i + 1$, return $N + 1$.

---

### Visual Algorithm Walkthrough

#### Example: `nums = [3, 4, -1, 1]` ($N = 4$)

```
Initial:  [ 3,  4, -1,  1 ]
          i=0

Step 1: i = 0, val = 3. Target index for 3 is 3 - 1 = 2.
        Swap nums[0] with nums[2]:
        [ -1,  4,  3,  1 ]
        nums[0] = -1 (<= 0), while loop stops.

Step 2: i = 1, val = 4. Target index for 4 is 4 - 1 = 3.
        Swap nums[1] with nums[3]:
        [ -1,  1,  3,  4 ]
        nums[1] = 1 (in range). Target index for 1 is 1 - 1 = 0.
        Swap nums[1] with nums[0]:
        [  1, -1,  3,  4 ]
        nums[1] = -1 (<= 0), while loop stops.

Step 3: i = 2, val = 3. Already at index 2. Stop.

Step 4: i = 3, val = 4. Already at index 3. Stop.

Array after Cyclic Sort: [ 1, -1, 3, 4 ]

Scan Phase:
  Index 0: nums[0] == 1 (Match)
  Index 1: nums[1] == -1 != 2 (Mismatch!) -> Return 2.
```

---

### Solved Examples with Multiple Inputs

| Input `nums` | Transformed Array After Cyclic Sort | First Mismatch Index | Output | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `[1, 2, 0]` | `[1, 2, 0]` | $i = 2$ (`nums[2] != 3`) | `3` | Missing value within range |
| `[3, 4, -1, 1]` | `[1, -1, 3, 4]` | $i = 1$ (`nums[1] != 2`) | `2` | Negative ignored, 2 missing |
| `[7, 8, 9, 11, 12]` | `[7, 8, 9, 11, 12]` | $i = 0$ (`nums[0] != 1`) | `1` | All elements $> N$, returns 1 |
| `[1]` | `[1]` | None | `2` | All $1..N$ present $\to N+1$ |
| `[1, 1]` | `[1, 1]` | $i = 1$ (`nums[1] != 2`) | `2` | Handles duplicate values safely |

---

### Multi-Language Implementations

#### 1. Python 3 (Clean, Typed)
```python
from typing import List

class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                correct_idx = nums[i] - 1
                nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
                
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
                
        return n + 1
```

#### 2. C++ (C++17 / STL)
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int firstMissingPositive(std::vector<int>& nums) {
        int n = nums.size();

        for (int i = 0; i < n; ++i) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                std::swap(nums[i], nums[nums[i] - 1]);
            }
        }

        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }

        return n + 1;
    }
};
```

#### 3. Java (Modern, Typed)
```java
class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;

        for (int i = 0; i < n; i++) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                int correctIdx = nums[i] - 1;
                int temp = nums[i];
                nums[i] = nums[correctIdx];
                nums[correctIdx] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }

        return n + 1;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(n)$. Although there is a nested `while` loop, each swap places at least one number into its correct and final position. No number is moved more than twice, so the total number of swaps across all iterations is at most $n$. The subsequent verification pass takes $\mathcal{O}(n)$ time. Total time is strictly linear $\mathcal{O}(n)$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space. The swaps are executed entirely in-place modifying the original array without allocating external memory.

---

### Takeaway Pattern & Interview Traps

1. **Duplicate Values Handling in Cyclic Sort:**
   - Always compare `nums[nums[i] - 1] != nums[i]`, NOT `nums[i] - 1 != i`. If two duplicate numbers attempt to occupy the same position (e.g., `nums = [1, 1]`), checking `nums[i] - 1 != i` creates an infinite loop swap, while `nums[nums[i] - 1] != nums[i]` detects the duplicate and stops immediately.
2. **Zero-Based Indexing Offset:**
   - The value $x$ belongs at index $x - 1$. Forgetting the $-1$ offset leads to an out-of-bounds index when $x = N$.
3. **Array Mutation Awareness:**
   - In interviews, clarify if in-place mutation of the input array is permitted. If not, $\mathcal{O}(1)$ space is impossible because storing membership of $N$ values requires at least an $N$-bit bitmap or hash set ($\mathcal{O}(n)$ space).
