---
date: "2026-09-15"
type: leetcode-solution
category: "Greedy"
folder: "13. Greedy"
title: "LeetCode 769: Max Chunks To Make Sorted"
tags:
  - leetcode
  - coding
  - greedy
  - array
  - stack
  - google
  - amazon
  - microsoft
---

# LeetCode 769: Max Chunks To Make Sorted

**Target Companies:** Google, Amazon, Adobe, Microsoft  
**Difficulty:** Medium  
**Topic:** Greedy / Array / Stack  

---

### Problem Statement

You are given an integer array `arr` of length `n` that represents a permutation of the integers in the range `[0, n - 1]`.

We split `arr` into some number of **chunks** (i.e., contiguous subarrays), and individually sort each chunk. After concatenating them, the result should equal the sorted array `[0, 1, ..., n - 1]`.

Return *the largest number of chunks we can make to sort the array*.

---

### Input & Output Formats & Constraints

- **Input:** An integer array `arr` representing a permutation of `[0, n - 1]`.
- **Output:** An integer denoting the maximum number of chunks.
- **Constraints:**
  - `n == arr.length`
  - `1 <= n <= 10` (Note: the greedy logic scales seamlessly to $n \le 10^5$)
  - `0 <= arr[i] < n`
  - All integers in `arr` are **unique**.

---

### Key Idea & Intuition

#### Permutation Invariant
In the sorted array, the element at index $i$ is exactly $i$:
$$\text{sorted\_arr}[i] = i \quad \text{for all } 0 \le i < n$$

For a contiguous chunk ending at index $i$ to be sorted independently and end up in its proper place, the prefix `arr[0..i]` must contain exactly the set of numbers $\{0, 1, 2, \dots, i\}$.

#### The Max-So-Far Condition
Because `arr` is a permutation containing unique non-negative integers:
1. The prefix `arr[0..i]` contains $i + 1$ distinct elements.
2. If the maximum element among these $i + 1$ numbers is $i$:
   $$\max(arr[0], arr[1], \dots, arr[i]) == i$$
   then all elements in `arr[0..i]` must be $\le i$.
3. Since there are $i + 1$ unique non-negative integers $\le i$, the set $\{arr[0], \dots, arr[i]\}$ must be **identically equal** to $\{0, 1, \dots, i\}$!
4. Therefore, no element in the suffix $arr[i+1..n-1]$ can have a value $\le i$. The prefix is completely self-contained, allowing us to safely finalize a chunk at index $i$.

#### Why Greedy Cut is Optimal
To maximize the number of chunks, we should finalize a chunk at the **earliest possible opportunity**. Every time $\max_{0 \le k \le i} arr[k] == i$, we immediately close the chunk and increment our counter. Delaying a cut would only merge chunks and decrease the total count.

---

### Solution Approach (Step-by-Step)

1. **Initialize State:**
   - `max_so_far = 0`
   - `chunks = 0`
2. **Linear Scan:**
   - For $i$ from $0$ to $n - 1$:
     - Update `max_so_far = max(max_so_far, arr[i])`.
     - Check if `max_so_far == i`.
     - If true, a chunk boundary is confirmed: increment `chunks += 1`.
3. **Return Result:**
   - Return `chunks`.

---

### Visual Algorithm Walkthrough

#### Trace for `arr = [1, 0, 2, 3, 4]`
```
Index i:        0      1      2      3      4
Value arr[i]:   1      0      2      3      4

i = 0:
  arr[0] = 1
  max_so_far = max(0, 1) = 1
  Is max_so_far == i? (1 == 0) -> False. (Cannot cut yet; 0 is missing)

i = 1:
  arr[1] = 0
  max_so_far = max(1, 0) = 1
  Is max_so_far == i? (1 == 1) -> TRUE! 
  -> Cut Chunk 1! [1, 0] (Contains elements {0, 1})
  -> chunks = 1

i = 2:
  arr[2] = 2
  max_so_far = max(1, 2) = 2
  Is max_so_far == i? (2 == 2) -> TRUE!
  -> Cut Chunk 2! [2]
  -> chunks = 2

i = 3:
  arr[3] = 3
  max_so_far = max(2, 3) = 3
  Is max_so_far == i? (3 == 3) -> TRUE!
  -> Cut Chunk 3! [3]
  -> chunks = 3

i = 4:
  arr[4] = 4
  max_so_far = max(3, 4) = 4
  Is max_so_far == i? (4 == 4) -> TRUE!
  -> Cut Chunk 4! [4]
  -> chunks = 4

Chunks: [1, 0] | [2] | [3] | [4]
Sorted: [0, 1] | [2] | [3] | [4] -> Concatenated: [0, 1, 2, 3, 4]
Total Chunks: 4
```

---

### Solved Examples with Multiple Inputs

| Input `arr` | Step Tracing (`i`, `arr[i]`, `max_so_far`, `== i`) | Chunks Partitioned | Total Chunks |
|---|---|---|---|
| `[4, 3, 2, 1, 0]` | $i=0: 4 \ne 0, i=1: 4 \ne 1, i=2: 4 \ne 2, i=3: 4 \ne 3, i=4: 4 == 4$ | `[4, 3, 2, 1, 0]` | `1` |
| `[1, 0, 2, 3, 4]` | $i=1 (1==1), i=2 (2==2), i=3 (3==3), i=4 (4==4)$ | `[1, 0]`, `[2]`, `[3]`, `[4]` | `4` |
| `[0, 1, 2, 3, 4]` | Every $i$ has $max == i$ | `[0]`, `[1]`, `[2]`, `[3]`, `[4]` | `5` |
| `[2, 0, 1]` | $i=0: 2, i=1: 2, i=2: 2 == 2$ | `[2, 0, 1]` | `1` |

---

### Multi-Language Implementations

#### Python 3
```python
class Solution:
    def maxChunksToSorted(self, arr: list[int]) -> int:
        max_so_far: int = 0
        chunks: int = 0
        
        for i, val in enumerate(arr):
            max_so_far = max(max_so_far, val)
            # Prefix elements arr[0..i] contain all numbers from 0 to i
            if max_so_far == i:
                chunks += 1
                
        return chunks
```

#### C++17
```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxChunksToSorted(const std::vector<int>& arr) {
        int max_so_far = 0;
        int chunks = 0;
        int n = static_cast<int>(arr.size());
        
        for (int i = 0; i < n; ++i) {
            max_so_far = std::max(max_so_far, arr[i]);
            // If the maximum value seen so far equals index i, cut chunk
            if (max_so_far == i) {
                chunks++;
            }
        }
        
        return chunks;
    }
};
```

#### Java 17
```java
class Solution {
    public int maxChunksToSorted(int[] arr) {
        int maxSoFar = 0;
        int chunks = 0;
        
        for (int i = 0; i < arr.length; i++) {
            maxSoFar = Math.max(maxSoFar, arr[i]);
            // When max element in prefix equals current index, we have a valid chunk
            if (maxSoFar == i) {
                chunks++;
            }
        }
        
        return chunks;
    }
}
```

---

### Complexity Analysis

- **Time Complexity:** $\mathcal{O}(N)$, where $N = |arr|$. We make a single linear pass over the array performing constant time comparisons.
- **Space Complexity:** $\mathcal{O}(1)$ auxiliary space, as only two scalar counters (`max_so_far` and `chunks`) are maintained.

---

### Takeaway Pattern & Interview Traps

1. **Difference with LeetCode 768 (Max Chunks To Make Sorted II):**
   - In LC 768, elements are **not** restricted to permutations of $[0, n - 1]$ and duplicates are allowed.
   - For LC 768, the condition becomes: `max_left[i] <= min_right[i + 1]`, which requires monotonic stack or prefix-max/suffix-min arrays.
   - For LC 769, the special property that elements are a permutation of $[0, n-1]$ reduces the check to `max_so_far == i`. Mentioning this distinction in interviews demonstrates deep domain mastery.
2. **Early Cuts Guarantee Maximum Chunks:** Because chunk boundaries are purely independent conditions, greedily cutting every time `max_so_far == i` is mathematically optimal.