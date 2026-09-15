---
date: "2026-09-15"
type: leetcode-solution
category: "Sliding Window"
folder: "03. Sliding Window"
title: "LeetCode 2134: Minimum Swaps to Group All 1’s Together II"
tags:
  - leetcode
  - coding
  - sliding-window
  - array
  - amazon
  - google
---

# LeetCode 2134: Minimum Swaps to Group All 1’s Together II

**Target Companies:** Amazon, Google, Microsoft, Bloomberg  
**Difficulty:** Medium  
**Topic:** Fixed-Size Sliding Window / Circular Array / Swap Optimization  

---

### Problem Statement

A **swap** is defined as taking two distinct positions in an array and swapping the values in them.

A **circular** array is defined as an array where we consider the first element and the last element to be adjacent.

Given a **binary circular** array `nums`, return *the minimum number of swaps required to group all `1`'s present in the array together at any location*.

---

### Input & Output Formats & Constraints

- **Input:** `nums: List[int]` (consisting of only `0` and `1`)
- **Output:** `int` (minimum number of swaps)
- **Constraints:**
  - $1 \le \text{nums.length} \le 10^5$
  - `nums[i]` is either `0` or `1`.

---

### Key Idea & Intuition

- **Target Window Invariant:**
  - Suppose there are $K = \sum \text{nums}[i]$ ones in the entire array.
  - When all $1$s are grouped together, they must occupy a single contiguous block of length $K$.
  - In any window of length $K$, every `0` present inside must be swapped with a `1` from outside the window.
  - Therefore, the number of swaps needed to group all $1$s in a specific window of size $K$ is:
    $$\text{swaps} = \text{count of zeros in the window} = K - \text{count of ones in the window}$$
  - To **minimize swaps**, we simply need to **maximize the count of $1$s in any window of fixed size $K$**:
    $$\min(\text{swaps}) = K - \max_{\text{all windows of size } K}(\text{ones\_in\_window})$$
- **Handling the Circular Property via Modulo ($\mathcal{O}(1)$ Extra Space):**
  - Instead of physically duplicating the array (`nums + nums` which allocates an extra array of size $N$), simulate the circular traversal using index modulo:
    $$\text{nums}[i \pmod N]$$
  - We slide the window from $i = 0$ to $2N - 1$.
- **Edge Cases:**
  - If $K == 0$ or $K == N$, all $1$s are already trivially grouped $\implies$ return `0`.

---

### Solution Approach (Step-by-Step)

1. Let $N = \text{len}(nums)$.
2. Calculate total ones: $K = \sum \text{nums}$.
3. If $K \le 1$ or $K == N$: return `0`.
4. Count ones in the first window of size $K$: `curr_ones = sum(nums[:K])`.
5. Initialize `max_ones = curr_ones`.
6. Slide the window from $i = K$ to $N + K - 1$:
   - Add incoming element: `curr_ones += nums[i % N]`.
   - Remove outgoing element: `curr_ones -= nums[(i - K) % N]`.
   - Update `max_ones = max(max_ones, curr_ones)`.
7. Return $K - \text{max\_ones}$.

---

### Visual Algorithm Walkthrough

Let `nums = [0, 1, 0, 1, 1, 0, 0]`.
- $N = 7$.
- Total ones: $K = 0 + 1 + 0 + 1 + 1 + 0 + 0 = 3$.
- We slide a window of size $K = 3$ across the virtual circular array `nums + nums`:

```
Array:   0  1  0  1  1  0  0 | 0  1  0  1  1  0  0
Index:   0  1  2  3  4  5  6 | 7  8  9 10 11 12 13

Initial Window [0..2]: [0, 1, 0]             -> ones = 1
Slide to [1..3]:           [1, 0, 1]         -> ones = 2
Slide to [2..4]:              [0, 1, 1]      -> ones = 2
Slide to [3..5]:                 [1, 1, 0]   -> ones = 2
Slide to [4..6]:                    [1, 0, 0]-> ones = 1
Slide to [5..7] (wraps!):              [0, 0, 0]-> ones = 0
Slide to [6..8] (wraps!):                 [0, 0, 1]-> ones = 1

Max ones in any window of size 3 = 2.
Min swaps required = K - max_ones = 3 - 2 = 1 swap!
(Swap nums[0]=0 with nums[3]=1 -> [1, 1, 1, 0, 0, 0, 0]).
```

---

### Solved Examples with Multiple Inputs

| Test Case | `nums` | $K$ (Total 1s) | Max 1s in Window $K$ | Min Swaps ($K - \max$) |
| :--- | :--- | :--- | :--- | :--- |
| **Example 1** | `[0,1,0,1,1,0,0]` | 3 | 2 | $3 - 2 = 1$ |
| **Example 2 (Circular Wrap)** | `[0,1,1,1,0,0,1,1,0]` | 5 | 4 | $5 - 4 = 1$ |
| **All Ones** | `[1,1,0,0,1]` | 3 | 3 | $3 - 3 = 0$ |
| **No Ones** | `[0,0,0]` | 0 | 0 | `0` |

---

### Multi-Language Implementations

#### Python 3
```python
from typing import List

class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        """
        Finds the minimum swaps to group all 1's together in a circular binary array.
        Uses a fixed-size sliding window with modulo indexing for O(1) extra space.
        """
        k = sum(nums)
        n = len(nums)

        if k <= 1 or k == n:
            return 0

        # Prime the initial window of size k
        curr_ones = sum(nums[:k])
        max_ones = curr_ones

        # Slide window across circular array of virtual length n + k
        for i in range(k, n + k):
            curr_ones += nums[i % n] - nums[(i - k) % n]
            if curr_ones > max_ones:
                max_ones = curr_ones

        return k - max_ones
```

#### C++17
```cpp
#include <vector>
#include <numeric>
#include <algorithm>

class Solution {
public:
    int minSwaps(const std::vector<int>& nums) {
        int n = static_cast<int>(nums.size());
        int k = std::accumulate(nums.begin(), nums.end(), 0);

        if (k <= 1 || k == n) return 0;

        int curr_ones = 0;
        for (int i = 0; i < k; ++i) {
            curr_ones += nums[i];
        }

        int max_ones = curr_ones;

        for (int i = k; i < n + k; ++i) {
            curr_ones += nums[i % n] - nums[(i - k) % n];
            max_ones = std::max(max_ones, curr_ones);
        }

        return k - max_ones;
    }
};
```

#### Java
```java
class Solution {
    public int minSwaps(int[] nums) {
        int n = nums.length;
        int k = 0;
        for (int x : nums) {
            k += x;
        }

        if (k <= 1 || k == n) {
            return 0;
        }

        int currOnes = 0;
        for (int i = 0; i < k; i++) {
            currOnes += nums[i];
        }

        int maxOnes = currOnes;

        for (int i = k; i < n + k; i++) {
            currOnes += nums[i % n] - nums[(i - k) % n];
            maxOnes = Math.max(maxOnes, currOnes);
        }

        return k - maxOnes;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$ where $N = \text{nums.length}$.
  - Summing elements takes $\mathcal{O}(N)$.
  - Sliding the window runs for $N$ iterations, doing $\mathcal{O}(1)$ operations per step.
  - Total time is $\mathcal{O}(N)$, running in $< 10 \text{ ms}$ for $N = 10^5$.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space because we use modulo indexing rather than concatenating two arrays.

---

### Takeaway Pattern & Interview Traps

- **Fixed Window Size Equals Total Ones ($K$):** The length of the window is predetermined by the problem physics: all $1$s must end up grouped together, which forms a window of length $K$.
- **Modulo Circular Traversal:** Avoiding array duplication (`nums * 2`) preserves memory locality and uses strictly $\mathcal{O}(1)$ auxiliary space.